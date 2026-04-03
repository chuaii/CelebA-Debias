import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(ROOT, "datasets")
IMG_DIR = os.path.join(DATA_ROOT, "img_align_celeba")
ATTR_CSV = os.path.join(DATA_ROOT, "list_attr_celeba.csv")
PARTITION_CSV = os.path.join(DATA_ROOT, "list_eval_partition.csv")

# group = target * 2 + sensitive
TARGET_ATTR = "Blond_Hair"
SENSITIVE_ATTR = "Male"
GROUP_NAMES = {0: "NonBlond_Female", 1: "NonBlond_Male", 2: "Blond_Female", 3: "Blond_Male"}

# Mouth_Slightly_Open × Smiling 
# TARGET_ATTR = "Mouth_Slightly_Open"
# SENSITIVE_ATTR = "Smiling"
# GROUP_NAMES = {0: "MouthClosed_NonSmiling", 1: "MouthClosed_Smiling", 2: "MouthOpen_NonSmiling", 3: "MouthOpen_Smiling"}

BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_EPOCHS = 10
WARMUP_EPOCHS = 1  # linear warmup for first N epochs, 0 to disable
LR = 1e-5
LR_BACKBONE = 1e-6
WD = 1e-4
EMBED_DIM = 128
TEMPERATURE = 0.07 # SupCon default: 0.07
LAMBDA_CON = 1.5  # 0 = baseline, >0 = debias

DEVICE = "cuda"
SEED = 42
CKPT_DIR = os.path.join(ROOT, "checkpoints")
