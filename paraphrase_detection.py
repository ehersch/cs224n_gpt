'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import os
import random
import sys
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    gpt_output = self.gpt(input_ids, attention_mask)
    hidden_states = gpt_output['last_hidden_state']
    last_hidden_state = hidden_states[:, -1, :]
    logits = self.paraphrase_detection_head(last_hidden_state)
    return logits



def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath, weights_only=False)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


# ---------------------------------------------------------------------------
# Modal training/inference for paraphrase detection
# ---------------------------------------------------------------------------
try:
  import modal
except ImportError:
  modal = None

if modal is not None:
  MODAL_WORKSPACE = "/workspace"
  MODAL_CHECKPOINTS = "/checkpoints"

  para_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
      "torch",
      "transformers",
      "einops",
      "tqdm",
      "requests",
      "filelock",
      "importlib-metadata",
      "sacrebleu",
      "scikit-learn",
      "numpy",
      "pandas",
    )
    .workdir(MODAL_WORKSPACE)
    .add_local_dir(".", remote_path=MODAL_WORKSPACE, ignore=[".git", ".git/*"])
  )

  para_volume = modal.Volume.from_name("paraphrase-checkpoints", create_if_missing=True)
  para_volumes = {MODAL_CHECKPOINTS: para_volume}
  para_app = modal.App("paraphrase-detection")

  @para_app.function(
    image=para_image,
    gpu="L4",
    timeout=3600 * 2,
    volumes=para_volumes,
  )
  def train_and_test_modal(
    run_name: str = "paraphrase_modal",
    para_train: str = "data/quora-train.csv",
    para_dev: str = "data/quora-dev.csv",
    para_test: str = "data/quora-test-student.csv",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-5,
    model_size: str = "gpt2",
    seed: int = 11711,
  ):
    sys.path.insert(0, MODAL_WORKSPACE)
    seed_everything(seed)

    run_dir = os.path.join(MODAL_CHECKPOINTS, run_name)
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    args = argparse.Namespace(
      para_train=os.path.join(MODAL_WORKSPACE, para_train),
      para_dev=os.path.join(MODAL_WORKSPACE, para_dev),
      para_test=os.path.join(MODAL_WORKSPACE, para_test),
      para_dev_out=os.path.join(pred_dir, "para-dev-output.csv"),
      para_test_out=os.path.join(pred_dir, "para-test-output.csv"),
      seed=seed,
      epochs=epochs,
      use_gpu=True,
      batch_size=batch_size,
      lr=lr,
      model_size=model_size,
    )
    args.filepath = os.path.join(run_dir, f"{epochs}-{lr}-paraphrase.pt")

    train(args)
    test(args)
    para_volume.commit()

    return {
      "checkpoint": os.path.relpath(args.filepath, MODAL_CHECKPOINTS),
      "dev_out": os.path.relpath(args.para_dev_out, MODAL_CHECKPOINTS),
      "test_out": os.path.relpath(args.para_test_out, MODAL_CHECKPOINTS),
    }

  @para_app.function(image=para_image, volumes=para_volumes)
  def read_volume_file(relative_path: str) -> bytes:
    para_volume.reload()
    path = os.path.join(MODAL_CHECKPOINTS, relative_path)
    with open(path, "rb") as f:
      return f.read()

  @para_app.local_entrypoint()
  def main_modal(
    run_name: str = "paraphrase_modal",
    para_train: str = "data/quora-train.csv",
    para_dev: str = "data/quora-dev.csv",
    para_test: str = "data/quora-test-student.csv",
    para_dev_out: str = "predictions/para-dev-output.csv",
    para_test_out: str = "predictions/para-test-output.csv",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-5,
    model_size: str = "gpt2",
    seed: int = 11711,
  ):
    print("Running paraphrase detection train+test on Modal...")
    result = train_and_test_modal.remote(
      run_name=run_name,
      para_train=para_train,
      para_dev=para_dev,
      para_test=para_test,
      epochs=epochs,
      batch_size=batch_size,
      lr=lr,
      model_size=model_size,
      seed=seed,
    )
    print(f"Checkpoint saved in Volume: {result['checkpoint']}")

    dev_bytes = read_volume_file.remote(result["dev_out"])
    test_bytes = read_volume_file.remote(result["test_out"])

    os.makedirs(os.path.dirname(para_dev_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(para_test_out) or ".", exist_ok=True)
    with open(para_dev_out, "wb") as f:
      f.write(dev_bytes)
    with open(para_test_out, "wb") as f:
      f.write(test_bytes)
    print(f"Saved {para_dev_out}")
    print(f"Saved {para_test_out}")


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
