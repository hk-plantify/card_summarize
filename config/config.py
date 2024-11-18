from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data Config
# -----------------------------------------------------------------------------

_C.DATASET.PATH = ".tmp/preprocessed_card_data.csv"
_C.MAX_LEN = "1024"

_C.BATCH_SIZE = "4"
_C.VAL_SIZE = "0.1"
_C.TEST_SIZE = "0.1"

# -----------------------------------------------------------------------------
# Model Config
# -----------------------------------------------------------------------------
_C.LR = "1e-4"
_C.MAX_EPOCHS = "10"
_C.WARMUP_RATIO = "0.1"

# -----------------------------------------------------------------------------
# Experiment Config
# -----------------------------------------------------------------------------
_C.MODEL_PATH = ".tmp/best_model.pt"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()