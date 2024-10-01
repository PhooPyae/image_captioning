import datasets
from nltk.translate.bleu_score import sentence_bleu
from pycocoevalcap.cider.cider import Cider
import numpy as np

class Metrics:
    def __init__(self, tokenizer):
        self.rouge = datasets.load_metric("rouge")
        self.tokenizer = tokenizer

    def compute_bleu(self, pred_str, label_str):
        bleu_scores = []
        for pred, label in zip(pred_str, label_str):
            pred_tokens = pred.split()
            label_tokens = [label.split()]  # List of reference sentences
            bleu_scores.append(sentence_bleu(label_tokens, pred_tokens))

        return np.mean(bleu_scores)
            
    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = self.rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        # mean_bleu = self.compute_bleu(pred_str, label_str)
        
        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
            # "bleu": round(mean_bleu, 4)
        }