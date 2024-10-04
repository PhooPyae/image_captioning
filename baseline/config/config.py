import torch
class Config:
    #sample hyperparameters
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False
    
    batch_size = 256

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    # vocab_size = len(dataset.vocab)
    num_layers = 3
    learning_rate = 3e-4
    num_epochs = 10

class CocoConfig:
    ann_root = '/projects/bdfr/plinn/image_captioning/baseline/coco'
    image_root = '/projects/bdfr/plinn/image_captioning/baseline/coco/images'
    train = 'coco_karpathy_train.json'
    val = 'coco_karpathy_val.json'
    test = 'coco_karpathy_test.json'