import argparse
import copy
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT, add_arguments


def _quantize_weight_tensor(w: torch.Tensor, bits: int):
    if bits == 8:
        qmax, qmin = 127, -128
    elif bits == 4:
        qmax, qmin = 7, -8
    else:
        raise ValueError(f"Unsupported bits={bits}")
    max_abs = w.detach().abs().max()
    scale = torch.clamp(max_abs / max(qmax, 1), min=1e-8)
    q = torch.clamp(torch.round(w / scale), qmin, qmax).to(torch.int8)
    return q, scale


def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    q_u = (q + 8).to(torch.uint8).reshape(-1)
    if q_u.numel() % 2 == 1:
        q_u = torch.cat([q_u, torch.zeros(1, dtype=torch.uint8, device=q_u.device)])
    lo = q_u[0::2] & 0x0F
    hi = (q_u[1::2] & 0x0F) << 4
    return lo | hi


def _unpack_int4(packed: torch.Tensor, shape) -> torch.Tensor:
    p = packed.reshape(-1)
    lo = (p & 0x0F).to(torch.int16)
    hi = ((p >> 4) & 0x0F).to(torch.int16)
    vals = torch.empty(p.numel() * 2, dtype=torch.int16, device=p.device)
    vals[0::2] = lo
    vals[1::2] = hi
    n = int(torch.tensor(shape).prod().item())
    vals = vals[:n]
    vals = (vals - 8).to(torch.int8)
    return vals.reshape(*shape)


class WeightOnlyQuantLinear(nn.Module):
    def __init__(self, base: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        # Use actual loaded tensor shapes (can differ from module metadata in this codebase).
        w = base.weight.detach().cpu()
        self.weight_shape = tuple(w.shape)
        self.out_features, self.in_features = self.weight_shape
        q, scale = _quantize_weight_tensor(w, bits=bits)
        self.register_buffer("scale", scale.detach().cpu())
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach().cpu())
        else:
            self.bias = None
        if bits == 8:
            self.register_buffer("qweight_int8", q.contiguous())
            self.qweight_packed = None
        else:
            self.register_buffer("qweight_packed", _pack_int4(q).contiguous())
            self.qweight_int8 = None

    def _dequant_weight(self, device):
        if self.bits == 8:
            q = self.qweight_int8.to(device)
        else:
            q = _unpack_int4(
                self.qweight_packed.to(device),
                self.weight_shape,
            )
        return q.float() * self.scale.to(device)

    def forward(self, x):
        w = self._dequant_weight(x.device)
        b = self.bias.to(x.device) if self.bias is not None else None
        return F.linear(x, w, b)


def _replace_linear_with_weight_only_quant(module: nn.Module, bits: int):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyQuantLinear(child, bits=bits))
        else:
            _replace_linear_with_weight_only_quant(child, bits=bits)


def get_tokenizer(model):
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
    return tok


def model_forward_logits(model, input_ids, attention_mask):
    out = model(input_ids, attention_mask)
    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "last_hidden_state" in out and hasattr(model, "gpt"):
            return model.gpt.hidden_state_to_token(out["last_hidden_state"])
    if hasattr(out, "logits"):
        return out.logits
    return out


@torch.no_grad()
def generate_sequence(
    model, input_ids, attention_mask, temperature: float, top_p: float
):
    if hasattr(model, "tokenizer") and hasattr(model, "generate"):
        out = model.generate(input_ids, temperature=temperature, top_p=top_p)
        if isinstance(out, tuple):
            return out[0][0]
        return out[0]
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=128,
        pad_token_id=get_tokenizer(model).eos_token_id,
    )
    return gen_ids[0]


def estimate_flops_per_token_with_seqlen(model, seq_len: int) -> float:
    d = model.gpt.config.hidden_size if hasattr(model.gpt, "config") else None
    l = model.gpt.config.num_hidden_layers if hasattr(model.gpt, "config") else None
    if d is None or l is None:
        d = model.gpt.embed_layer.weight.shape[1]
        l = len(model.gpt.gpt_layers)
    return float(l * ((12 * d * d) + (2 * seq_len * d)))


def serialized_model_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        num_bytes = os.path.getsize(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return num_bytes / (1024**2)


@torch.no_grad()
def eval_dev_metrics(
    model, dev_gold_path: str, batch_size: int = 8, device: str = "cpu"
):
    model = model.to(device)
    model.eval()

    dev_dataset = SonnetsDataset(dev_gold_path)
    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=dev_dataset.collate_fn,
    )

    pad_id = dev_dataset.tokenizer.pad_token_id
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    for batch in dev_loader:
        b_ids = batch["token_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        logits = model_forward_logits(model, b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
        labels = b_ids[:, 1:].contiguous().flatten()

        loss = F.cross_entropy(logits, labels, reduction="mean", ignore_index=pad_id)
        total_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=-1)
        valid = labels != pad_id
        correct += (preds[valid] == labels[valid]).sum().item()
        total += valid.sum().item()

    ce = total_loss / num_batches if num_batches else float("inf")
    token_acc = (correct / total) if total else 0.0
    ppl = float(torch.exp(torch.tensor(ce)).item()) if ce < 50 else float("inf")
    return {"dev_ce": ce, "token_acc": token_acc, "ppl": ppl}


@torch.no_grad()
def eval_dev_chrf(
    model,
    dev_held_out_path: str,
    dev_gold_path: str,
    device: str = "cpu",
    temperature: float = 1.2,
    top_p: float = 0.9,
):
    from sacrebleu.metrics import CHRF

    model = model.to(device)
    model.eval()
    held_out_dataset = SonnetsDataset(dev_held_out_path)
    gold_dataset = SonnetsDataset(dev_gold_path)
    tokenizer = get_tokenizer(model)

    generated_sonnets = []
    total_gen_time = 0.0
    total_samples = 0
    total_new_tokens = 0

    for batch in held_out_dataset:
        encoding = tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = generate_sequence(
            model,
            input_ids=encoding["input_ids"],
            attention_mask=encoding.get("attention_mask", None),
            temperature=temperature,
            top_p=top_p,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        total_gen_time += time.perf_counter() - t0

        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        generated_sonnets.append(f"{decoded_output}\n\n")

        total_samples += 1
        new_tokens = max(0, int(output.shape[0] - encoding["input_ids"].shape[1]))
        total_new_tokens += new_tokens

    true_sonnets = [x[1] for x in gold_dataset]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    chrf = float(CHRF().corpus_score(generated_sonnets, [true_sonnets]).score)
    avg_sec_per_sample = (
        total_gen_time / total_samples if total_samples else float("inf")
    )
    toks_per_sec = total_new_tokens / total_gen_time if total_gen_time > 0 else 0.0

    return {
        "dev_chrf": chrf,
        "avg_gen_sec_per_sample": avg_sec_per_sample,
        "gen_tokens_per_sec": toks_per_sec,
    }


def load_checkpoint_model(checkpoint_path: str):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = add_arguments(saved["args"])
    model = SonnetGPT(args)
    model.load_state_dict(saved["model"], strict=False)
    model.eval()
    return model


def make_variant(model, variant: str):
    variant = variant.lower().strip()
    m = copy.deepcopy(model).cpu().eval()

    if variant == "fp32":
        return m.float(), "fp32"

    if variant == "fp16":
        return m.to(dtype=torch.float16), "fp16"

    if variant == "fp64":
        return m.to(dtype=torch.float64), "fp64"

    if variant == "bf16":
        return m.to(dtype=torch.bfloat16), "bf16"

    # True weight-only int8 quantization for all nn.Linear modules.
    if variant == "int8":
        _replace_linear_with_weight_only_quant(m, bits=8)
        return m, "int8_weight_only"

    # True weight-only packed int4 quantization for all nn.Linear modules.
    if variant == "int4":
        _replace_linear_with_weight_only_quant(m, bits=4)
        return m, "int4_weight_only"

    raise ValueError(f"Unknown variant: {variant}")


def _run_benchmark(
    checkpoint: str,
    dev_held_out_path: str,
    dev_gold_path: str,
    batch_size: int,
    seq_len: int,
    variants: str,
    device: str,
    temperature: float,
    top_p: float,
):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    base = load_checkpoint_model(checkpoint)
    flops_per_token = estimate_flops_per_token_with_seqlen(base, seq_len=seq_len)

    lines = [
        f"checkpoint: {checkpoint}",
        f"dev_held_out_path: {dev_held_out_path}",
        f"dev_gold_path: {dev_gold_path}",
        f"seq_len for FLOPs estimate: {seq_len}",
        "",
        "variant,device,dev_chrf,dev_ce,token_acc,ppl,avg_gen_sec_per_sample,gen_tokens_per_sec,size_mb,approx_flops_per_token",
    ]

    for v in [x.strip() for x in variants.split(",") if x.strip()]:
        try:
            model_v, label = make_variant(base, v)
            eval_device = device
            metrics = eval_dev_metrics(
                model_v,
                dev_gold_path=dev_gold_path,
                batch_size=batch_size,
                device=eval_device,
            )
            gen_eval = eval_dev_chrf(
                model_v,
                dev_held_out_path=dev_held_out_path,
                dev_gold_path=dev_gold_path,
                device=eval_device,
                temperature=temperature,
                top_p=top_p,
            )
            size_mb = serialized_model_size_mb(model_v)
            lines.append(
                f"{label},{eval_device},{gen_eval['dev_chrf']:.6f},{metrics['dev_ce']:.6f},{metrics['token_acc']:.6f},"
                f"{metrics['ppl']:.6f},{gen_eval['avg_gen_sec_per_sample']:.6f},{gen_eval['gen_tokens_per_sec']:.3f},"
                f"{size_mb:.2f},{flops_per_token:.0f}"
            )
        except Exception as e:
            lines.append(f"{v},ERROR,{e}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dev_held_out_path", type=str, default="data/sonnets_held_out_dev.txt"
    )
    parser.add_argument(
        "--dev_gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument(
        "--variants",
        type=str,
        default="fp32,bf16,fp16,int8,int4",
        help="Comma-separated: fp32,bf16,fp16,fp64,int8,int4",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    report = _run_benchmark(
        checkpoint=args.checkpoint,
        dev_held_out_path=args.dev_held_out_path,
        dev_gold_path=args.dev_gold_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        variants=args.variants,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(report)


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    MODAL_WORKSPACE = "/workspace"
    MODAL_CHECKPOINTS = "/checkpoints"

    quant_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements("requirements.txt")
        .pip_install("numpy", "pandas")
        .workdir(MODAL_WORKSPACE)
        .add_local_dir(".", remote_path=MODAL_WORKSPACE, ignore=[".git", ".git/*"])
    )

    volume = modal.Volume.from_name("sonnet-checkpoints", create_if_missing=True)
    app = modal.App("sonnet-quantization-benchmark")

    @app.function(
        image=quant_image,
        gpu="L4:4",
        timeout=3600 * 2,
        volumes={MODAL_CHECKPOINTS: volume},
    )
    def run_quant_benchmark_modal(
        checkpoint_in_volume: str,
        dev_held_out_path: str = "data/sonnets_held_out_dev.txt",
        dev_gold_path: str = "data/TRUE_sonnets_held_out_dev.txt",
        batch_size: int = 8,
        seq_len: int = 128,
        variants: str = "fp32,bf16,fp16,int8,int4",
        device: str = "cuda",
        temperature: float = 1.2,
        top_p: float = 0.9,
    ) -> str:
        sys.path.insert(0, MODAL_WORKSPACE)
        checkpoint_path = os.path.join(MODAL_CHECKPOINTS, checkpoint_in_volume)
        dev_held_out_full = os.path.join(MODAL_WORKSPACE, dev_held_out_path)
        dev_gold_full = os.path.join(MODAL_WORKSPACE, dev_gold_path)
        return _run_benchmark(
            checkpoint=checkpoint_path,
            dev_held_out_path=dev_held_out_full,
            dev_gold_path=dev_gold_full,
            batch_size=batch_size,
            seq_len=seq_len,
            variants=variants,
            device=device,
            temperature=temperature,
            top_p=top_p,
        )

    @app.local_entrypoint()
    def main_modal(
        checkpoint_in_volume: str,
        dev_held_out_path: str = "data/sonnets_held_out_dev.txt",
        dev_gold_path: str = "data/TRUE_sonnets_held_out_dev.txt",
        batch_size: int = 8,
        seq_len: int = 128,
        variants: str = "fp32,bf16,fp16,int8,int4",
        device: str = "cuda",
        temperature: float = 1.2,
        top_p: float = 0.9,
        out: str = "predictions/quantization/quant_report.csv",
    ):
        report = run_quant_benchmark_modal.remote(
            checkpoint_in_volume=checkpoint_in_volume,
            dev_held_out_path=dev_held_out_path,
            dev_gold_path=dev_gold_path,
            batch_size=batch_size,
            seq_len=seq_len,
            variants=variants,
            device=device,
            temperature=temperature,
            top_p=top_p,
        )
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
