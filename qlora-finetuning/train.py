#!/usr/bin/env python3
"""
Gemma 4 31B — QLoRA Fine-Tuning
Task: Multi-label classification of children's health self-reports (German)
"""

import argparse
import json
import logging
import os
import re

import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    pipeline,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 31B with QLoRA")
    p.add_argument("--train_pkl",    default="../data/train_df_new_interviews.pkl")
    p.add_argument("--test_pkl",     default="../data/test_df_new_interviews.pkl")
    p.add_argument("--output_dir",   default="./gemma4-31b-health")
    p.add_argument("--model_id",     default="google/gemma-4-31B-it")
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_seq_len",  type=int,   default=256)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--eval_steps",   type=int,   default=10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--hf_token",     default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--skip_eval",    action="store_true",
                   help="Skip final evaluation on test set")
    return p.parse_args()

# ── Label setup ───────────────────────────────────────────────────────────────

codes_health   = ["physical_health", "mental_health", "daily_functioning",
                  "health_unspecific", "health_none"]
codes_freq     = ["freq_mentioned", "freq_not_mentioned"]
codes_neg      = ["health_none", "freq_not_mentioned"]
codes_combined = codes_health + codes_freq

code_pos     = [c for c in codes_combined if c not in codes_neg]
code_pos_new = {label: i for i, label in enumerate(code_pos)}
idx_to_label = {v: k for k, v in code_pos_new.items()}
num_classes  = len(code_pos_new)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Klassifiziere die Aussage eines Kindes. "
    "Antworte ausschliesslich im JSON-Format:\n"
    '{"health": ["<label>", ...], "frequency": "<label>"}\n'
    "health-Labels (mehrere moeglich): physical_health, mental_health, "
    "daily_functioning, health_unspecific, health_none.\n"
    "frequency-Label (genau eines): freq_mentioned, freq_not_mentioned."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def indices_to_labels(lst):
    return [idx_to_label[i] for i in lst]


def labels_to_json(label_indices):
    names         = indices_to_labels(label_indices)
    health_labels = [l for l in names if l in codes_health] or ["health_none"]
    freq_label    = next((l for l in names if l in codes_freq), "freq_not_mentioned")
    return json.dumps({"health": health_labels, "frequency": freq_label}, ensure_ascii=False)


def make_messages(text, label_json=None):
    """
    Build a conversation in the standard roles format.
    Gemma 4 natively supports system/user/assistant roles (unlike Gemma 3).
    SFTTrainer calls tokenizer.apply_chat_template() on the "messages" column.

    Training:  label_json provided  → system + user + assistant
    Inference: label_json=None      → system + user only; pipeline adds generation prompt
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": text},
    ]
    if label_json is not None:
        messages.append({"role": "assistant", "content": label_json})
    return messages


def safe_parse_json(raw):
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        raw = raw[raw.find("{"):]
        raw = re.sub(r'("frequency"\s*:\s*"[a-z_]+)$', r'\1"}', raw)
        if not raw.strip().endswith("}"):
            raw = raw.strip() + '"}'
        return json.loads(raw)
    except Exception:
        return None


def classify(text, pipe, tokenizer):
    """
    Inference using the same chat template path as training.
    pipeline internally calls apply_chat_template(add_generation_prompt=True).
    """
    messages = make_messages(text)  # no assistant turn → generation mode
    outputs  = pipe(messages, max_new_tokens=60, do_sample=False)
    raw      = outputs[0]["generated_text"][-1]["content"].strip()

    parsed = safe_parse_json(raw)
    if parsed is None:
        log.warning("Parse failed: %r", raw)
        return []

    labels = []
    for h in parsed.get("health", []):
        if h in code_pos_new:
            labels.append(code_pos_new[h])
    f = parsed.get("frequency")
    if f in code_pos_new:
        labels.append(code_pos_new[f])
    return list(set(labels))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        log.info("Logged in to HuggingFace Hub")
    else:
        log.warning("No HF_TOKEN set — will fail on gated models like Gemma")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    log.info("Loading data ...")
    train_df = pd.read_pickle(args.train_pkl)
    train_df = train_df[train_df["annotator"] != ""].copy().reset_index(drop=True)

    test_df = pd.read_pickle(args.test_pkl)
    test_df = test_df[test_df["peter"].notna()].copy().reset_index(drop=True)
    test_df["label"] = test_df["peter"]

    log.info("Train: %d | Test: %d", len(train_df), len(test_df))

    all_conversations = [
        {"messages": make_messages(row["childPart"], labels_to_json(row["label"]))}
        for _, row in train_df.iterrows()
    ]

    train_convs, val_convs = train_test_split(
        all_conversations, test_size=0.2, random_state=args.seed, shuffle=True
    )
    train_dataset = Dataset.from_list(train_convs)
    val_dataset   = Dataset.from_list(val_convs)
    log.info("Train split: %d | Val split: %d", len(train_dataset), len(val_dataset))

    # ── Model ─────────────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Gemma 4 PEFT compatibility patch ─────────────────────────────────────
    # Gemma4ClippableLinear wraps nn.Linear with input/output clamping for the
    # vision/audio encoders. It inherits from nn.Module, not nn.Linear, so
    # PEFT rejects it. We walk the loaded model and replace every instance with
    # a plain nn.Linear that copies the quantized weight in-place.
    # This must be done AFTER from_pretrained() but BEFORE get_peft_model().

    log.info("Loading %s in 4-bit NF4 ...", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        # flash_attention_2 is incompatible with Gemma 4's global attention
        # layers (head_dim=512 > FA2 max of 256). Use "eager" or "sdpa".
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    log.info(
        "Model loaded. Params: %s | GPU: %.1f GB",
        f"{model.num_parameters():,}",
        torch.cuda.memory_allocated() / 1e9,
    )

    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    def _unwrap_clippable(model):
        """
        Replace every Gemma4ClippableLinear wrapper with its inner .linear
        module in-place. The inner layer is already a quantized Linear4bit,
        which PEFT recognises. The clamping behaviour is dropped — it only
        affects the vision/audio encoder paths which we exclude from LoRA
        anyway and which don't participate in text fine-tuning.
        """
        for name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, Gemma4ClippableLinear):
                    setattr(module, child_name, child.linear)
        return model

    model = _unwrap_clippable(model)
    log.info("Unwrapped Gemma4ClippableLinear wrappers for PEFT compatibility")


    # ── LoRA ──────────────────────────────────────────────────────────────────
    # exclude_modules skips the vision/audio towers entirely — we only want to
    # fine-tune the language model layers for this text classification task.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"],
    )
    # NOTE: do NOT call get_peft_model() here — peft_config is passed to
    # SFTTrainer below which applies LoRA internally. Calling both would apply it twice.

    # Patch Gemma 4's chat template to include {% generation %} markers.
    # The generation block must directly wrap the content output expression.
    # We replace the bare output with a role-conditional wrapped version.
    if "{% generation %}" not in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{{- captured_content -}}",
            "{%- if role == 'model' %}{% generation %}{{- captured_content -}}{% endgeneration %}{%- else %}{{- captured_content -}}{%- endif %}",
        )
        ok = "{% generation %}" in tokenizer.chat_template
        log.info("Chat template patch: %s", "OK" if ok else "FAILED — loss will cover full sequence")

    sft_config = SFTConfig(
        output_dir=args.output_dir,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        bf16=True,
        tf32=True,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=10,
        report_to="none",

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        seed=args.seed,

        max_length=args.max_seq_len,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info(
        "Starting training. Steps/epoch: ~%d",
        len(train_dataset) // (args.batch_size * args.grad_accum),
    )
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = os.path.join(args.output_dir, "adapter_final")
    trainer.save_model(adapter_path)   # saves both adapter weights and tokenizer
    log.info("Adapter saved to: %s", adapter_path)

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.skip_eval:
        log.info("Skipping evaluation (--skip_eval)")
        return

    try:
        from small_text.utils.labels import list_to_csr
    except ImportError:
        log.warning("small_text not installed — skipping evaluation")
        return

    log.info("Running inference on %d test examples ...", len(test_df))
    model.eval()

    infer_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    ft_results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        ft_results.append(classify(row["childPart"], infer_pipe, tokenizer))

    y_true = list_to_csr(test_df["label"].tolist(), shape=(len(test_df), num_classes))
    y_pred = list_to_csr(ft_results,                shape=(len(ft_results), num_classes))

    report = classification_report(
        y_true.toarray(),
        y_pred.toarray(),
        target_names=list(code_pos_new.keys()),
        zero_division=0,
    )
    log.info("\n=== Fine-Tuned Gemma 4 31B ===\n%s", report)

    report_path = os.path.join(args.output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to: %s", report_path)


if __name__ == "__main__":
    main()
