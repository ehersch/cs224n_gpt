'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.

With LoRA (fewer params, often better for small datasets):
  `python sonnet_generation.py --use_gpu --use_lora`

SFT training on Modal (GPU in the cloud):
  `modal run sonnet_generation.py`           # single run
  `modal run sonnet_generation.py::main_lora_sweep`   # LoRA sweep over ranks 4,8,16,32,64
  Optional: --epochs, --batch-size, --lr, --use-lora, --lora-rank, etc.
  Sweep: --lora-ranks "4,8,16,32,64" (custom list).
'''

import argparse
import os
import random
import sys
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from modules.lora import LoRALinear

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


def _apply_lora_to_gpt(gpt, rank, alpha):
  """Freeze base GPT and wrap target linear layers with LoRA."""
  for param in gpt.parameters():
    param.requires_grad = False

  for layer in gpt.gpt_layers:
    # Attention: Q, K, V
    layer.self_attention.query = LoRALinear(layer.self_attention.query, rank=rank, alpha=alpha)
    layer.self_attention.key = LoRALinear(layer.self_attention.key, rank=rank, alpha=alpha)
    layer.self_attention.value = LoRALinear(layer.self_attention.value, rank=rank, alpha=alpha)
    # Attention output
    layer.attention_dense = LoRALinear(layer.attention_dense, rank=rank, alpha=alpha)
    # MLP
    layer.interm_dense = LoRALinear(layer.interm_dense, rank=rank, alpha=alpha)
    layer.out_dense = LoRALinear(layer.out_dense, rank=rank, alpha=alpha)


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for sonnet generation."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.use_lora = getattr(args, 'use_lora', False)

    if self.use_lora:
      _apply_lora_to_gpt(self.gpt, rank=args.lora_rank, alpha=args.lora_alpha)
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    gpt_output = self.gpt(input_ids, attention_mask)
    last_hidden_state = gpt_output['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(last_hidden_state)
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  # @torch.no_grad()
  # def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
  #   """
  #   Generates an original sonnet using top-p sampling and softmax temperature.

  #   TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
  #   In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
  #   there are many.
  #   """
  #   token_ids = encoding.to(self.get_device())
  #   attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


  #   for _ in range(max_length):
  #     # Forward pass to get logits
  #     logits_sequence = self.forward(token_ids, attention_mask)
  #     logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

  #     # Convert logits to probabilities
  #     probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

  #     # Top-p (nucleus) sampling
  #     sorted_probs, sorted_indices = torch.sort(probs, descending=True)
  #     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
  #     top_p_mask = cumulative_probs <= top_p
  #     top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
  #     top_p_mask[..., 0] = True  # Always include the highest probability token
  #     filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
  #     filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

  #     # Sample from filtered distribution
  #     sampled_index = torch.multinomial(filtered_probs, 1)
  #     sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

  #     # Stop if end-of-sequence token is reached
  #     if sampled_token.item() == self.tokenizer.eos_token_id:
  #       break

  #     # Append sampled token
  #     token_ids = torch.cat([token_ids, sampled_token], dim=1)
  #     attention_mask = torch.cat(
  #       [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
  #     )

  #   generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
  #   return token_ids, generated_output

  @torch.no_grad()
  def generate(self, encoding, temperature=0.8, top_p=0.9, top_k=40, max_length=128):
    """
    Generates a sonnet using optimized Top-K and Top-P (Nucleus) sampling.
    """
    model_device = self.get_device()
    token_ids = encoding.to(model_device)
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    current_ids = token_ids

    for _ in range(max_length):
        attention_mask = torch.ones_like(current_ids).to(model_device)
        logits = self.forward(current_ids, attention_mask)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Top-P (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        if next_token.item() == self.tokenizer.eos_token_id:
            break

    generated_output = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
    return current_ids, generated_output

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
  """Train GPT-2 for sonnet generation with padding mask, grad clip, LR schedule, early stopping."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(
    sonnet_dataset,
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=sonnet_dataset.collate_fn,
    num_workers=args.num_workers,
  )
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  if args.use_lora:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=args.weight_decay)
    print(f"LoRA enabled: training {sum(p.numel() for p in params):,} parameters")
  else:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

  scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
  pad_id = model.tokenizer.pad_token_id
  best_loss = float('inf')
  best_epoch = -1

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(
        logits, labels, reduction='mean', ignore_index=pad_id
      )
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        model.parameters() if not args.use_lora else params,
        max_norm=args.grad_clip_max_norm,
      )
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    scheduler.step()
    print(f"Epoch {epoch}: train loss :: {train_loss:.3f} (lr={scheduler.get_last_lr()[0]:.2e}).")
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')
    if train_loss < best_loss:
      best_loss = train_loss
      best_epoch = epoch
      save_model(model, optimizer, args, f'{args.epochs-1}_{args.filepath}')
      print(f"  New best loss {best_loss:.3f}, saved as final checkpoint.")

    if epoch - best_epoch >= args.early_stopping_patience:
      print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stopping_patience} epochs).")
      break


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(
    f'{args.epochs-1}_{args.filepath}',
    weights_only=False,
    map_location=device,
  )

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output, skip_special_tokens=True)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2-xl')

  # Training improvements
  parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
  parser.add_argument("--grad_clip_max_norm", type=float, default=1.0, help="Max norm for gradient clipping")
  parser.add_argument("--early_stopping_patience", type=int, default=3, help="Stop after N epochs without improvement")
  parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers (0 to disable)")

  # LoRA (Low-Rank Adaptation) - fewer trainable params, often better for small datasets
  parser.add_argument("--use_lora", action='store_true', help="Use LoRA instead of full fine-tuning")
  parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (typical: 4, 8, 16)")
  parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling (2*rank)")

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
# Modal SFT training: run with `modal run sonnet_generation.py`
# ---------------------------------------------------------------------------
try:
  import modal
except ImportError:
  modal = None

if modal is not None:
  MODAL_WORKSPACE = "/workspace"
  MODAL_CHECKPOINTS = "/checkpoints"

  sonnet_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
      "torch", "transformers", "einops", "tqdm", "requests", "filelock", "importlib-metadata"
    )
    .workdir(MODAL_WORKSPACE)
    .add_local_dir(".", remote_path=MODAL_WORKSPACE, ignore=[".git", ".git/*"])
  )

  volume = modal.Volume.from_name("sonnet-checkpoints", create_if_missing=True)
  volumes = {MODAL_CHECKPOINTS: volume}

  app = modal.App("sonnet-lora-rank-sweep")

  @app.function(
    image=sonnet_image,
    gpu="T4",
    timeout=3600 * 2,  # 2 hours
    volumes=volumes,
  )
  def train_sft(
    sonnet_path: str = "data/sonnets.txt",
    held_out_sonnet_path: str = "data/sonnets_held_out.txt",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-5,
    model_size: str = "gpt2",
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    seed: int = 11711,
    temperature: float = 1.2,
    top_p: float = 0.9,
    weight_decay: float = 0.01,
    grad_clip_max_norm: float = 1.0,
    early_stopping_patience: int = 3,
    num_workers: int = 2,
  ):
    """Run SFT training on Modal with padding mask, grad clip, LR schedule, early stopping."""
    sys.path.insert(0, MODAL_WORKSPACE)
    seed_everything(seed)

    args = argparse.Namespace(
      sonnet_path=os.path.join(MODAL_WORKSPACE, sonnet_path),
      held_out_sonnet_path=os.path.join(MODAL_WORKSPACE, held_out_sonnet_path),
      sonnet_out="predictions/generated_sonnets.txt",
      epochs=epochs,
      batch_size=batch_size,
      lr=lr,
      model_size=model_size,
      use_lora=use_lora,
      lora_rank=lora_rank,
      lora_alpha=lora_alpha,
      temperature=temperature,
      top_p=top_p,
      weight_decay=weight_decay,
      grad_clip_max_norm=grad_clip_max_norm,
      early_stopping_patience=early_stopping_patience,
      num_workers=num_workers,
    )
    args.filepath = (
      f"{epochs}-{lr}-sonnet-lora{lora_rank}-alpha{lora_alpha}.pt"
      if use_lora
      else f"{epochs}-{lr}-sonnet.pt"
    )
    args.use_gpu = True

    device = torch.device("cuda")
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(
      sonnet_dataset,
      shuffle=True,
      batch_size=args.batch_size,
      collate_fn=sonnet_dataset.collate_fn,
      num_workers=num_workers,
    )
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    if args.use_lora:
      params = [p for p in model.parameters() if p.requires_grad]
      optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
      print(f"LoRA enabled: training {sum(p.numel() for p in params):,} parameters")
    else:
      optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    pad_id = model.tokenizer.pad_token_id
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
      model.train()
      train_loss = 0
      num_batches = 0
      for batch in tqdm(sonnet_dataloader, desc=f"train-{epoch}", disable=False):
        b_ids = batch["token_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(
          logits, labels, reduction="mean", ignore_index=pad_id
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
          params if args.use_lora else model.parameters(),
          max_norm=args.grad_clip_max_norm,
        )
        optimizer.step()
        train_loss += loss.item()
        num_batches += 1

      train_loss /= num_batches
      scheduler.step()
      print(f"Epoch {epoch}: train loss :: {train_loss:.3f} (lr={scheduler.get_last_lr()[0]:.2e})")
      model.eval()
      for batch in held_out_sonnet_dataset:
        encoding = model.tokenizer(
          batch[1], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        output = model.generate(
          encoding["input_ids"],
          temperature=args.temperature,
          top_p=args.top_p,
        )
        print(f"{batch[1]}{output[1]}\n\n")

      checkpoint_name = f"{epoch}_{args.filepath}"
      checkpoint_path = os.path.join(MODAL_CHECKPOINTS, checkpoint_name)
      save_model(model, optimizer, args, checkpoint_path)
      if train_loss < best_loss:
        best_loss = train_loss
        best_epoch = epoch
        final_name = f"{args.epochs - 1}_{args.filepath}"
        final_path = os.path.join(MODAL_CHECKPOINTS, final_name)
        save_model(model, optimizer, args, final_path)
        print(f"  New best loss {best_loss:.3f}, saved as final checkpoint.")
      if epoch - best_epoch >= args.early_stopping_patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stopping_patience} epochs).")
        break
    volume.commit()

    # Read checkpoint in same container and return bytes (avoids separate download step)
    final_name = f"{args.epochs - 1}_{args.filepath}"
    final_path = os.path.join(MODAL_CHECKPOINTS, final_name)
    with open(final_path, "rb") as f:
      checkpoint_data = f.read()
    return {"final_checkpoint": final_name, "checkpoint_data": checkpoint_data}

  @app.function(image=sonnet_image, volumes=volumes)
  def download_checkpoint(filename: str) -> bytes:
    """Read checkpoint from Volume (must run remotely)."""
    volume.reload()
    path = os.path.join(MODAL_CHECKPOINTS, filename)
    with open(path, "rb") as f:
      return f.read()

  @app.local_entrypoint()
  def main(
    sonnet_path: str = "data/sonnets.txt",
    held_out_sonnet_path: str = "data/sonnets_held_out.txt",
    sonnet_out: str = "predictions/generated_sonnets.txt",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 5e-6,
    model_size: str = "gpt2",
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    temperature: float = 1.2,
    top_p: float = 0.9,
  ):
    """Run SFT training on Modal, download checkpoint, run generation locally."""
    print("Running SFT training on Modal...")
    result = train_sft.remote(
      sonnet_path=sonnet_path,
      held_out_sonnet_path=held_out_sonnet_path,
      epochs=epochs,
      batch_size=batch_size,
      lr=lr,
      model_size=model_size,
      use_lora=use_lora,
      lora_rank=lora_rank,
      lora_alpha=lora_alpha,
      temperature=temperature,
      top_p=top_p,
    )
    final_checkpoint = result["final_checkpoint"]
    print(f"Training complete. Saving checkpoint {final_checkpoint}...")
    with open(final_checkpoint, "wb") as f:
      f.write(result["checkpoint_data"])
    print(f"Checkpoint saved to {final_checkpoint}")

    # Run generation locally
    args = argparse.Namespace(
      sonnet_path=sonnet_path,
      held_out_sonnet_path=held_out_sonnet_path,
      sonnet_out=sonnet_out,
      epochs=epochs,
      batch_size=batch_size,
      lr=lr,
      model_size=model_size,
      use_lora=use_lora,
      use_gpu=torch.cuda.is_available(),
      temperature=temperature,
      top_p=top_p,
    )
    args.filepath = f"{epochs}-{lr}-sonnet.pt"
    generate_submission_sonnets(args)

  LORA_RANKS = [4, 8, 16, 32, 64]
  # LORA_RANKS = [64]

  @app.local_entrypoint()
  def main_lora_sweep(
    sonnet_path: str = "data/sonnets.txt",
    held_out_sonnet_path: str = "data/sonnets_held_out.txt",
    sonnet_out: str = "predictions/generated_sonnets.txt",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 5e-6,
    model_size: str = "gpt2",
    lora_alpha: float = 16.0,
    temperature: float = 1.2,
    top_p: float = 0.9,
    lora_ranks: str = "4,8,16,32,64",
  ):
    """Run SFT with LoRA sequentially for multiple ranks (default: 4,8,16,32,64)."""
    ranks = [int(r.strip()) for r in lora_ranks.split(",")]
    for i, rank in enumerate(ranks):
      print(f"\n{'='*60}")
      print(f"Experiment {i+1}/{len(ranks)}: LoRA rank = {rank}")
      print(f"{'='*60}\n")
      result = train_sft.remote(
        sonnet_path=sonnet_path,
        held_out_sonnet_path=held_out_sonnet_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_size=model_size,
        use_lora=True,
        lora_rank=rank,
        lora_alpha=lora_alpha,
        temperature=temperature,
        top_p=top_p,
      )
      final_checkpoint = result["final_checkpoint"]
      print(f"Saving checkpoint {final_checkpoint}...")
      with open(final_checkpoint, "wb") as f:
        f.write(result["checkpoint_data"])
      print(f"Checkpoint saved to {final_checkpoint}")

      base, ext = os.path.splitext(sonnet_out)
      run_sonnet_out = f"{base}_lora{rank}{ext}"
      args = argparse.Namespace(
        sonnet_path=sonnet_path,
        held_out_sonnet_path=held_out_sonnet_path,
        sonnet_out=run_sonnet_out,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_size=model_size,
        use_lora=True,
        use_gpu=torch.cuda.is_available(),
        temperature=temperature,
        top_p=top_p,
      )
      args.filepath = f"{epochs}-{lr}-sonnet-lora{rank}-alpha{lora_alpha}.pt"
      generate_submission_sonnets(args)
      print(f"Submission file written to {run_sonnet_out}")

    print(f"\nDone. Ran {len(ranks)} experiments for LoRA ranks: {ranks}")

  @app.local_entrypoint()
  def main_lora_alpha_sweep(
    sonnet_path: str = "data/sonnets.txt",
    held_out_sonnet_path: str = "data/sonnets_held_out.txt",
    sonnet_out: str = "predictions/generated_sonnets.txt",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 5e-6,
    model_size: str = "gpt2",
    lora_rank: int = 8,
    temperature: float = 1.2,
    top_p: float = 0.9,
    lora_alphas: str = "8,16,32",
  ):
    """Run SFT with LoRA sequentially for multiple alphas (rank fixed at 8). Default alphas: 8,16,32."""
    alphas = [int(a.strip()) for a in lora_alphas.split(",")]
    for i, alpha in enumerate(alphas):
      print(f"\n{'='*60}")
      print(f"Experiment {i+1}/{len(alphas)}: LoRA alpha = {alpha} (rank = {lora_rank})")
      print(f"{'='*60}\n")
      result = train_sft.remote(
        sonnet_path=sonnet_path,
        held_out_sonnet_path=held_out_sonnet_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_size=model_size,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=alpha,
        temperature=temperature,
        top_p=top_p,
      )
      final_checkpoint = result["final_checkpoint"]
      print(f"Saving checkpoint {final_checkpoint}...")
      with open(final_checkpoint, "wb") as f:
        f.write(result["checkpoint_data"])
      print(f"Checkpoint saved to {final_checkpoint}")

      base, ext = os.path.splitext(sonnet_out)
      run_sonnet_out = f"{base}_alpha{alpha}{ext}"
      args = argparse.Namespace(
        sonnet_path=sonnet_path,
        held_out_sonnet_path=held_out_sonnet_path,
        sonnet_out=run_sonnet_out,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_size=model_size,
        use_lora=True,
        use_gpu=torch.cuda.is_available(),
        temperature=temperature,
        top_p=top_p,
      )
      args.filepath = f"{epochs}-{lr}-sonnet-lora{lora_rank}-alpha{alpha}.pt"
      generate_submission_sonnets(args)
      print(f"Submission file written to {run_sonnet_out}")

    print(f"\nDone. Ran {len(alphas)} experiments for LoRA alphas: {alphas} (rank={lora_rank})")


if __name__ == "__main__":
  # With Modal installed, "modal run" should only run inference. Use "train" subcommand to train.
  if len(sys.argv) >= 2 and sys.argv[1] == "train":
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    generate_submission_sonnets(args)
  elif modal is not None:
    pass
  else:
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    generate_submission_sonnets(args)