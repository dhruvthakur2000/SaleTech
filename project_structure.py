import os

PROJECT_ROOT = "."  # current dir

folders = [
    "src/api",# FastAPI routes (WebRTC, health, etc.)
    "src/core",# Core logic (sessions, orchestration)
    "src/services",# ASR, LLM, TTS services
    "src/models",# Pydantic models / schemas
    "src/utils",# Helpers, logging, config
    "src/media",# Audio handling, VAD, codecs
    "config",
    "scripts",# Dev & ops scripts
    "docker",
    "tests",
]

files = [
    "src/main.py",
    "config/settings.py",
    "config/logging.yaml",
    "docker/Dockerfile",
    "docker/docker-compose.yml",
    ".gitignore",
    "pyproject.toml",
    ".env.example",
    "src/api/__init__.py",
    "src/core/__init__.py",
    "src/services/__init__.py",
    "src/models/__init__.py",
    "src/utils/__init__.py",
    "src/media/__init__.py",
    "tests/__init__.py",
]

def create_structure():
    print(" Setting up your project structure...\n")
    for folder in folders:
        path = os.path.join(PROJECT_ROOT, folder)
        os.makedirs(path, exist_ok=True)
        print(f" Created folder: {path}")

    for file in files:
        file_path = os.path.join(PROJECT_ROOT, file)
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, "w") as f:
            f.write("")  # Empty file
        print(f" Created file: {file_path}")

    print("\n Project structure created successfully.")

if __name__ == "__main__":
    create_structure()