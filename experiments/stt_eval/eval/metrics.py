from jiwer import wer

def calculate_wer(reference: str, hypothesis: str) -> float:
    reference = reference.lower().strip()
    hypothesis = hypothesis.lower().strip()
    return wer(reference, hypothesis)
