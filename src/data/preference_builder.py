from typing import List, Dict


class DPOPreferenceDatasetBuilder:
    """Structures preference data (prompt, chosen, rejected) for DPO alignment."""

    def build_pair(self, prompt: str, chosen: str, rejected: str, domain: str = "general") -> Dict:
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "domain": domain
        }

    def validate_pair(self, pair: Dict) -> bool:
        # Chosen response should ideally be helpful, factually correct, and natural Arabic
        if pair["chosen"] == pair["rejected"]:
            return False
        if len(pair["chosen"].strip()) == 0 or len(pair["rejected"].strip()) == 0:
            return False
        return True
