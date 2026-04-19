import asyncio
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    from transformers.cache_utils import Cache
except Exception:  # pragma: no cover - compatibility with older transformers
    Cache = Any

from config.settings import (
    DEFAULT_PRODUCT_CONTEXT,
    SALES_AGENT_SYSTEM_PROMPT,
    AppSettings,
)
from saletech.models.schemas import ConversationMessage, MessageRole
from saletech.utils.errors import SaleTechException
from saletech.utils.logger import get_logger


logger = get_logger("saletech.llm")


class LLMServiceError(SaleTechException):
    """Raised when the LLM service cannot load or generate safely."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(
            message=message,
            error_code="LLM_SERVICE_ERROR",
            status_code=500,
            original_exception=original_exception,
        )


class LLMWithKVCache:
    """
    Qwen chat service with per-session generation cache.

    The service mirrors the ASR service shape:
    - one global model instance
    - async initialization
    - streaming token output
    - per-session cleanup

    KV-cache reuse is attempted only when the new prompt is an exact token-prefix
    extension of the previous prompt/response. If that invariant does not hold,
    the service falls back to normal full-context generation.
    """

    def __init__(self):
        self.settings = AppSettings()
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.model_name = self.settings.qwen_model_path
        self.device = self._resolve_device()
        self.torch_dtype = self._resolve_dtype()

        self._session_caches: Dict[str, Cache] = {}
        self._session_input_ids: Dict[str, torch.Tensor] = {}
        self._generation_errors: Dict[str, Exception] = {}

        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._generation_lock = asyncio.Lock()

        logger.info(
            "llm_service_initialized",
            model=self.model_name,
            device=self.device,
            dtype=str(self.torch_dtype),
        )

    def _resolve_device(self) -> str:
        if self.settings.llm_device != "auto":
            return self.settings.llm_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self) -> torch.dtype:
        dtype = self.settings.llm_torch_dtype
        if dtype == "auto":
            return torch.float16 if self.device == "cuda" else torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    async def initialize(self) -> None:
        """Load tokenizer and Qwen model."""
        if self._initialized:
            return

        try:
            logger.info("loading_qwen_model", model=self.model_name)
            start_time = time.time()
            loop = asyncio.get_running_loop()

            self.tokenizer = await loop.run_in_executor(
                self._executor,
                self._load_tokenizer_blocking,
            )
            self.model = await loop.run_in_executor(
                self._executor,
                self._load_model_blocking,
            )

            if self.settings.llm_warmup_enabled:
                await self._generate_blocking(
                    [{"role": "user", "content": "Hi"}],
                    max_tokens=2,
                )

            self._initialized = True
            logger.info(
                "qwen_model_loaded",
                load_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("qwen_load_failed", error=str(e), exc_info=True)
            raise LLMServiceError(
                "LLM model initialization failed",
                original_exception=e,
            )

    def _load_tokenizer_blocking(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model_blocking(self):
        kwargs: dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "use_cache": True,
        }

        if self.device == "cuda":
            kwargs["device_map"] = "auto"
            if self.settings.llm_load_in_8bit:
                kwargs["load_in_8bit"] = True
            if self.settings.llm_attn_implementation:
                kwargs["attn_implementation"] = self.settings.llm_attn_implementation

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if self.device == "cpu":
            model = model.to(self.device)

        model.eval()
        return model

    async def generate_response(
        self,
        session_id: str,
        conversation_history: List[ConversationMessage],
        customer_name: Optional[str] = None,
        product_context: Optional[str] = None,
    ) -> str:
        """Collect a streamed response into one string."""
        chunks = []
        async for token in self.generate_with_cache(
            session_id=session_id,
            conversation_history=conversation_history,
            customer_name=customer_name,
            product_context=product_context,
        ):
            chunks.append(token)
        return "".join(chunks).strip()

    async def generate_with_cache(
        self,
        session_id: str,
        conversation_history: List[ConversationMessage],
        customer_name: Optional[str] = None,
        product_context: Optional[str] = None,
        past_kv_cache: Optional[bytes] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response for one conversation turn.

        `past_kv_cache` is optional serialized state from external storage.
        In-memory cache is preferred when available because model cache objects
        can be large and may not serialize reliably across process versions.
        """
        if not self._initialized or self.model is None or self.tokenizer is None:
            raise LLMServiceError("LLM service not initialized")

        start_time = time.time()
        token_count = 0
        cache_reused = False

        try:
            if past_kv_cache and session_id not in self._session_caches:
                self._load_serialized_cache(session_id, past_kv_cache)

            messages = self._build_messages(
                conversation_history=conversation_history,
                customer_name=customer_name,
                product_context=product_context,
            )

            async with self._generation_lock:
                async for token, cache_reused in self._generate_streaming_with_cache(
                    messages=messages,
                    session_id=session_id,
                ):
                    token_count += 1
                    yield token

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "llm_generation_complete",
                session_id=session_id,
                tokens=token_count,
                latency_ms=latency_ms,
                tokens_per_second=token_count / max(latency_ms / 1000, 0.001),
                cache_reused=cache_reused,
            )

        except Exception as e:
            logger.error(
                "llm_generation_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            raise LLMServiceError("LLM generation failed", original_exception=e)

    async def _generate_streaming_with_cache(
        self,
        messages: List[dict],
        session_id: str,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer([prompt], return_tensors="pt")
        full_input_ids = encoded.input_ids.to(self.model.device)
        full_attention_mask = encoded.attention_mask.to(self.model.device)

        input_ids = full_input_ids
        attention_mask = full_attention_mask
        past_key_values = None
        cache_reused = False

        cached_ids = self._session_input_ids.get(session_id)
        cached_cache = self._session_caches.get(session_id)

        if cached_ids is not None and cached_cache is not None:
            cached_length = cached_ids.shape[1]
            can_reuse = (
                full_input_ids.shape[1] > cached_length
                and torch.equal(full_input_ids[:, :cached_length].cpu(), cached_ids.cpu())
            )
            if can_reuse:
                input_ids = full_input_ids[:, cached_length:]
                attention_mask = full_attention_mask
                past_key_values = cached_cache
                cache_reused = True
                logger.info(
                    "kv_cache_reuse",
                    session_id=session_id,
                    cached_tokens=cached_length,
                    new_tokens=input_ids.shape[1],
                )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.settings.llm_max_tokens,
            "temperature": self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "do_sample": self.settings.llm_temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": True,
        }

        if past_key_values is not None:
            generation_kwargs["past_key_values"] = past_key_values

        thread = Thread(
            target=self._generate_and_update_cache,
            args=(generation_kwargs, session_id, full_input_ids, streamer),
            daemon=True,
        )
        thread.start()

        for token in streamer:
            yield token, cache_reused
            await asyncio.sleep(0)

        thread.join()

        if session_id in self._generation_errors:
            error = self._generation_errors.pop(session_id)

            if cache_reused:
                logger.warning(
                    "kv_cache_generation_failed_retrying_without_cache",
                    session_id=session_id,
                    error=str(error),
                )
                self.clear_session_cache(session_id)
                async for token, _ in self._generate_streaming_with_cache(
                    messages=messages,
                    session_id=session_id,
                ):
                    yield token, False
                return

            raise error

    def _generate_and_update_cache(
        self,
        generation_kwargs: dict,
        session_id: str,
        prompt_input_ids: torch.Tensor,
        streamer: TextIteratorStreamer,
    ) -> None:
        try:
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)

            sequences = outputs.sequences.detach().cpu()
            past_key_values = getattr(outputs, "past_key_values", None)

            if past_key_values is not None:
                self._session_caches[session_id] = past_key_values
                self._session_input_ids[session_id] = sequences
            else:
                self._session_input_ids[session_id] = prompt_input_ids.detach().cpu()

        except Exception as e:
            self._generation_errors[session_id] = e
            streamer.on_finalized_text("", stream_end=True)

    def _build_messages(
        self,
        conversation_history: List[ConversationMessage],
        customer_name: Optional[str],
        product_context: Optional[str],
    ) -> List[dict]:
        system_prompt = SALES_AGENT_SYSTEM_PROMPT
        system_prompt += f"\n\nPRODUCT:\n{product_context or DEFAULT_PRODUCT_CONTEXT}"

        if customer_name:
            system_prompt += f"\n\nCustomer: {customer_name}"

        messages = [{"role": "system", "content": system_prompt}]
        recent_history = conversation_history[-self.settings.llm_max_conversation_history:]

        for msg in recent_history:
            if msg.role == MessageRole.SYSTEM:
                continue
            role = "user" if msg.role == MessageRole.USER else "assistant"
            messages.append({"role": role, "content": msg.content})

        return messages

    def _load_serialized_cache(self, session_id: str, payload: bytes) -> None:
        try:
            state = pickle.loads(payload)
            cache = state.get("cache")
            input_ids = state.get("input_ids")
            if cache is not None and input_ids is not None:
                self._session_caches[session_id] = cache
                self._session_input_ids[session_id] = input_ids
                logger.info("kv_cache_loaded", session_id=session_id)
        except Exception as e:
            logger.warning(
                "kv_cache_deserialize_failed",
                session_id=session_id,
                error=str(e),
            )

    async def serialize_cache(self, session_id: str) -> Optional[bytes]:
        """Serialize per-session cache for external storage such as Redis."""
        try:
            cache = self._session_caches.get(session_id)
            input_ids = self._session_input_ids.get(session_id)
            if cache is None or input_ids is None:
                return None
            return pickle.dumps({"cache": cache, "input_ids": input_ids})
        except Exception as e:
            logger.warning(
                "kv_cache_serialize_failed",
                session_id=session_id,
                error=str(e),
            )
            return None

    def clear_session_cache(self, session_id: str) -> None:
        self._session_caches.pop(session_id, None)
        self._session_input_ids.pop(session_id, None)
        self._generation_errors.pop(session_id, None)
        logger.info("kv_cache_cleared", session_id=session_id)

    async def _generate_blocking(
        self,
        messages: List[dict],
        max_tokens: int = 50,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

    async def cleanup(self) -> None:
        logger.info("llm_cleanup_started")

        self.clear_all_session_caches()

        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer

        self.model = None
        self.tokenizer = None
        self._initialized = False

        self._executor.shutdown(wait=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("llm_service_cleaned_up")

    def clear_all_session_caches(self) -> None:
        self._session_caches.clear()
        self._session_input_ids.clear()
        self._generation_errors.clear()


_llm_service: Optional[LLMWithKVCache] = None
_llm_init_lock = asyncio.Lock()


async def get_llm_kvcache_service() -> LLMWithKVCache:
    """Get or create the global LLM service singleton."""
    global _llm_service

    async with _llm_init_lock:
        if _llm_service is None:
            instance = LLMWithKVCache()
            await instance.initialize()
            _llm_service = instance

    return _llm_service
