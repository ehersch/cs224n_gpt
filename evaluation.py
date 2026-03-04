# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
      'labels'].flatten()

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_true.extend(labels)
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)

if __name__ == "__main__":
    gold_path = "data/TRUE_sonnets_held_out_dev.txt"  
    # alpha_scores = [8, 16, 32]
    lora_scores = [8, 16, 32, 64, 128]
    lr_scores = ['1e_4', '2e_4', '3e_4']
    # for alpha in alpha_scores:
    #     test_path = f"predictions/generated_sonnets_dev_alpha{alpha}.txt"
    #     chrf_score = test_sonnet(test_path, gold_path)
    #     print(f"CHRF score for {test_path}: {chrf_score}")
    for lora in lora_scores:
        test_path = f"predictions/generated_sonnets_dev_lora{lora}.txt"
        chrf_score = test_sonnet(test_path, gold_path)
        print(f"CHRF score for {test_path}: {chrf_score}")
    # for lr in lr_scores:
    #     test_path = f"predictions/generated_sonnets_dev_lora4_{lr}.txt"
    #     chrf_score = test_sonnet(test_path, gold_path)
    #     print(f"CHRF score for {test_path}: {chrf_score}")
    test_path = f"predictions/generated_sonnets_dev.txt"
    chrf_score = test_sonnet(test_path, gold_path)
    print(f"CHRF score for {test_path}: {chrf_score}")
