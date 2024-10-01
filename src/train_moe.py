from torchvision import io, transforms
from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import default_data_collator

import sys
sys.path.append('/projects/bdfr/plinn/image_captioning/')
from config.config import Config, MOEConfig
from utils import metrics, util
from dataset.flickr8k import load_data, ImgDataset
from model.moe_model import VisionEncoderDecoderMoE
import wandb

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config = Config()
moe_config = MOEConfig()

transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
   ]
)

feature_extractor, tokenizer, model = util.load_pretrained(config)
logger.debug('Loaded Pretrained !')

train_df, val_df = load_data(data_path = '/projects/bdfr/plinn/image_captioning/data/Flickr8k.token.txt')
train_dataset = ImgDataset(train_df, root_dir = "/projects/bdfr/plinn/image_captioning/data/Flicker8k_Dataset",tokenizer=tokenizer,feature_extractor = feature_extractor ,transform = transform)
val_dataset = ImgDataset(val_df , root_dir = "/projects/bdfr/plinn/image_captioning/data/Flicker8k_Dataset",tokenizer=tokenizer,feature_extractor = feature_extractor , transform  = transform)
logger.debug('Loaded Dataset !')
logger.debug(f'Train data: {train_df.shape} Val data: {val_df.shape}')
logger.debug(train_dataset.__getitem__(0))
vl_model = VisionEncoderDecoderMoE(model, moe_config)
model = vl_model.model
logger.debug(f'Decoder Model {model.decoder}')

#update model parameters
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.max_length = 50
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

total_params, trainable_params = util.compute_parameters(model)
metrics = metrics.Metrics(tokenizer)

wandb.init(
    project="image_captioning_vit_gpt2",
    config={
        "encoder": config.ENCODER,
        "decoder": config.DECODER,
        "vocab_size": model.config.vocab_size,
        "epochs": config.EPOCHS,
        "val_epochs": config.VAL_EPOCHS,
        "train_batch_size": config.TRAIN_BATCH_SIZE,
        "val_batch_size": config.VAL_BATCH_SIZE,
        "learning_rate": config.LR,
        "seed": config.SEED,
        "max_len": config.MAX_LEN,
        "weight_decay": config.WEIGHT_DECAY,
        "num_workers": config.NUM_WORKERS,
        "img_size": config.IMG_SIZE,
        "label_mask": config.LABEL_MASK,
        "top_k_experts": moe_config.n_expert_per_token,
        "num_experts": moe_config.n_expert,
        "trainable_params": trainable_params,
        "total_params": total_params

    }
)

training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,  
    save_steps=2048, 
    warmup_steps=1024,  
    learning_rate = 5e-5,
    # max_steps=10, # delete for full training
    num_train_epochs = 5, #TRAIN_EPOCHS
    overwrite_output_dir=False,
    save_total_limit=1,
    report_to = 'wandb'
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    compute_metrics=metrics.compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)
trainer.train()
trainer.save_model('VIT_large_gpt2_moe')
wandb.finish()