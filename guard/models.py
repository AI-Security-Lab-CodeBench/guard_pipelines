"""Классификатор «бэкбон + пулинг + линейная голова»: энкодеры (полное дообучение) и 8B-декодеры (NF4 + LoRA)."""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .data import LABELS


def load_tokenizer(spec: dict):
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=spec.get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode(tok, texts: list[str], spec: dict, max_len: int) -> list[list[int]]:
    tmpl = spec.get("template")
    texts = [tmpl.format(cmd=t) if tmpl else t for t in texts]
    prefix = spec.get("prefix_tokens")
    if prefix:  # UniXcoder: <s> <encoder-only> </s> code </s>
        pid = tok.convert_tokens_to_ids(prefix)
        out = []
        for t in texts:
            body = tok(t, add_special_tokens=False)["input_ids"][: max_len - 3 - len(pid)]
            out.append([tok.cls_token_id, *pid, tok.sep_token_id, *body, tok.sep_token_id])
        return out
    return tok(texts, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]


def collate(batch_ids: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), n), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    return ids.to(device), mask.to(device)


class SeqClassifier(nn.Module):
    """Для декодеров всегда строится та же обёртка NF4 + LoRA, что при обучении: имена параметров совпадают
    с чекпойнтом, а численно инференс идентичен обучению."""

    def __init__(self, spec: dict, num_labels: int = len(LABELS)):
        super().__init__()
        self.spec = spec
        self.pooling = spec.get("pooling", "mean")
        kw = {"trust_remote_code": spec.get("trust_remote_code", False)}
        if spec["kind"] == "decoder":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type=spec.get("quant", "nf4"), bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
            kw["dtype"] = torch.bfloat16
            kw["device_map"] = {"": 0}
        else:
            kw["dtype"] = torch.float32
        backbone = AutoModel.from_pretrained(spec["hf_id"], **kw)
        if spec["kind"] == "decoder":
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True,
                                                       gradient_checkpointing_kwargs={"use_reentrant": False})
            lc = spec.get("lora", {})
            backbone = get_peft_model(backbone, LoraConfig(
                r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32), lora_dropout=lc.get("dropout", 0.05), bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.backbone = backbone
        self.drop = nn.Dropout(spec.get("dropout", 0.1))
        self.head = nn.Linear(backbone.config.hidden_size, num_labels)
        if spec["kind"] == "decoder":  # бэкбон уже на GPU через device_map
            self.head.to("cuda")

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return h[:, 0]
        if self.pooling == "last":  # правый паддинг: последний не-pad токен
            idx = mask.sum(1) - 1
            return h[torch.arange(h.size(0), device=h.device), idx]
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.head(self.drop(self.pool(h, attention_mask).to(self.head.weight.dtype)))

    def trainable_state(self) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.named_parameters() if v.requires_grad}


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() * (2 if p.__class__.__name__ == "Params4bit" else 1) for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
