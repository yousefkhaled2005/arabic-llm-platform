from typing import List, Dict
import re


class ArabicInstructionCurator:
    """Cleans, decontaminates, and formats Arabic instruction datasets."""

    MIN_PROMPT_CHARS = 10
    MAX_PROMPT_CHARS = 2000

    def clean_text(self, text: str) -> str:
        # Strip excessive whitespace and unwanted control characters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def filter_sample(self, sample: Dict[str, str]) -> bool:
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")

        if len(instruction) < self.MIN_PROMPT_CHARS or len(instruction) > self.MAX_PROMPT_CHARS:
            return False
        if not response or len(response) < 5:
            return False

        # Filter out common English contamination in Arabic datasets
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", instruction))
        if arabic_chars / (len(instruction) + 1) < 0.4:
            return False

        return True

    def format_chatml(self, instruction: str, response: str, system: str = "") -> Dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": response})
        return {"messages": messages}
