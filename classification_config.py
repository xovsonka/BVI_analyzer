from dataclasses import dataclass

MULTICLASS_LABELS = [
    "legit",
    "spam",
    "phishing",
    "financial_fraud",
]

LABEL_TO_ID = {label: i for i, label in enumerate(MULTICLASS_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class ClassificationConfig:
    mode: str = "multiclass"
    labels: tuple[str, ...] = tuple(MULTICLASS_LABELS)


CONFIG = ClassificationConfig()