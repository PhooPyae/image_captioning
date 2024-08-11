import torch
import multiprocessing as mp

class Config:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 50
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    EPOCHS = 1
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95
    EARLY_STOPPING = True

class MOEConfig:
    n_embd: int = 768 # model.config.decoder.n_embd
    n_expert: int = 8
    n_expert_per_token: int = 4
    intermediate_size: int = 3072 // 8
    bias: bool = True
    
