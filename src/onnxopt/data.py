"""The official 10kGNAD test split (1,028 articles), the same one Day 5 reported on."""

DATASET = "community-datasets/gnad10"


def load_test() -> tuple[list[str], list[int], list[str]]:
    from datasets import load_dataset

    test = load_dataset(DATASET, split="test")
    return list(test["text"]), list(test["label"]), test.features["label"].names
