import datasets
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pdb

class Metrics:
    def __init__(self, tokenizer):
        self.rouge = datasets.load_metric("rouge")
        self.cider = Cider()
        # self.meteor = Meteor()
        self.tokenizer = tokenizer

    def compute_metrics(self, pred):
        pdb.set_trace()
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute ROUGE
        # Compute ROUGE
        rouge_output = self.rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
        rouge_scores = {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }
        return rouge_scores

        # # Compute BLEU
        bleu_scores = []
        for pred, label in zip(pred_str, label_str):
            pred_tokens = pred.split()
            label_tokens = [label.split()]  # List of reference sentences
            bleu_scores.append(sentence_bleu(label_tokens, pred_tokens))
        bleu_score = {"bleu": np.mean(bleu_scores)}

        # # Prepare data for pycocoevalcap (references should be a list of possible captions)
        # references = {i: [label_str[i]] for i in range(len(label_str))}
        # hypotheses = {i: [pred_str[i]] for i in range(len(pred_str))}

        # # Compute CIDEr
        # cider_score, _ = self.cider.compute_score(references, hypotheses)
        # cider_scores = {"cider": round(cider_score, 4)}

        # # Compute METEOR
        # meteor_score, _ = self.meteor.compute_score(references, hypotheses)
        # meteor_scores = {"meteor": round(meteor_score, 4)}

        # # Combine all metrics
        # return {
        #     "bleu": round(bleu_score["bleu"], 4),
        #     **rouge_scores,
        #     **cider_scores,
        #     **meteor_scores
        # }

if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs

    AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.unk_token
    print('tokenizer is loaded')

    # Assuming your compute_metrics function is defined in a class called Metrics
    # and that you have already instantiated it like this:
    metrics = Metrics(tokenizer=tokenizer)  # Make sure tokenizer is available

    # Mock predictions and labels (IDs of tokens)
    mock_predictions = torch.tensor([
        [1169, 2415, 318, 14080, 257, 1720, 50256, 50256, 50256],  # Example tokenized sentence
        [64, 2415, 1838, 257, 5001, 379, 281, 15162, 18371]   # Another tokenized sentence
    ])

    # Labels (IDs of actual references, usually padded with -100 in training, but here with actual values)
    mock_labels = torch.tensor([
        [1169, 2415, 318, 14080, 257, 1720, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],  # Reference sentence
        [64, 2415, 1838, 257, 5001, 379, 281, 15162, 18371, 764, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]   # Another reference sentence
    ])

    # Wrap predictions and labels in a dictionary like how the Trainer would pass them
    mock_pred = {
        'predictions': mock_predictions,
        'label_ids': mock_labels
    }

    # Now you can call your compute_metrics function
    result = metrics.compute_metrics(mock_pred)

    # Print the result to see the computed metrics
    print(result)

