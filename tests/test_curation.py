from src.data.curator import ArabicInstructionCurator
from src.data.preference_builder import DPOPreferenceDatasetBuilder


def test_arabic_cleaning():
    curator = ArabicInstructionCurator()
    sample = {
        "instruction": "اشرح مفهوم التعلم المعزز من التغذية الراجعة البشرية بالتفصيل.",
        "response": "التعلم المعزز من التغذية الراجعة البشرية (RLHF) هو أسلوب لمواءمة النماذج اللغوية."
    }
    assert curator.filter_sample(sample) is True


def test_dpo_pair_builder():
    builder = DPOPreferenceDatasetBuilder()
    pair = builder.build_pair(
        prompt="ما هي عاصمة مصر؟",
        chosen="عاصمة مصر هي مدينة القاهرة.",
        rejected="عاصمة مصر هي الإسكندرية."
    )
    assert builder.validate_pair(pair) is True
