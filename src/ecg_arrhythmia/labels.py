"""Label definitions used across training and inference."""

LABEL_TO_ID = {"N": 0, "V": 1, "A": 2, "a": 2}
ID_TO_LABEL = {0: "N", 1: "V", 2: "a"}
TARGET_SYMBOLS = tuple(LABEL_TO_ID.keys())
