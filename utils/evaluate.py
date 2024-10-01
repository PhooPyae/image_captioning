from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator
from PIL import Image
import sys
import pandas as pd
from datasets import load_metric
from tqdm import tqdm
import numpy as np

sys.path.append('/projects/bdfr/plinn/image_captioning/')
from config.config import Config

config = Config()

# Load the saved model
model_path = '/projects/bdfr/plinn/image_captioning/VIT_large_gpt2_test'
model = VisionEncoderDecoderModel.from_pretrained(model_path)
model = model.to("cuda")

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)

tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

def generate_caption(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    caption = tokenizer.decode(model.generate(feature_extractor(image, return_tensors="pt").pixel_values.to("cuda"), max_length=50)[0])

    return caption

df = pd.read_csv('/projects/bdfr/plinn/image_captioning/train_df.csv')

rouge = load_metric("rouge")

def compute_rouge(caption, reference):
    score = rouge.compute(predictions=[caption], references=reference, rouge_types=["rouge2"])["rouge2"].mid
    return {
        "rouge2_precision": round(score.precision, 4),
        "rouge2_recall": round(score.recall, 4),
        "rouge2_fmeasure": round(score.fmeasure, 4),
    }

dataset_dir = '/projects/bdfr/plinn/image_captioning/data/Flicker8k_Dataset/'
# Compute ROUGE scores for the entire dataset
rouge_precisions = []
rouge_recalls = []
rouge_fmeasures = []

temp = {}
images = df['image'].unique()
print(f'total number of images {len(images)}')
images = np.random.choice(images, 5)
for image_pth in tqdm(images):
    gt_captions = df[df['image'] == image_pth]['caption']
    print('GrouthTruth')
    print(gt_captions)
    generated_caption = generate_caption(dataset_dir +'/'+ image_pth)
    print('Generated')
    print(generated_caption)
    print('-----------')
    # temp[image] = generated_caption

# print(f'total number of caption {len(temp)}')
# print(temp)

# for index, row in tqdm(df.iterrows()):
#     image_path = row['image']
#     reference_caption = row['caption']

#     generated_caption = temp[image_path]
#     rouge_scores = compute_rouge(generated_caption, reference_caption)

#     rouge_precisions.append(rouge_scores["rouge2_precision"])
#     rouge_recalls.append(rouge_scores["rouge2_recall"])
#     rouge_fmeasures.append(rouge_scores["rouge2_fmeasure"])


# print("Average ROUGE Precision:", sum(rouge_precisions) / len(rouge_precisions))
# print("Average ROUGE Recall:", sum(rouge_recalls) / len(rouge_recalls))
# print("Average ROUGE F-Measure:", sum(rouge_fmeasures) / len(rouge_fmeasures))