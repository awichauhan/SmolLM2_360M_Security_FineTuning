import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

PROJECT_ROOT = Path(__file__).resolve().parent


VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "instruction"
    / "security_sft_validation.jsonl"
)


DOMAIN_ADAPTER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "security_lora_adapter"
)


SFT_ADAPTER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "security_sft_adapter"
)


MAX_LENGTH = 256

BATCH_SIZE = 1


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print(
    "========== SFT VALIDATION EVALUATION =========="
)

print(
    f"Using device: {device}"
)


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


# SmolLM2 has no dedicated padding token.
#
# PAD and EOS therefore share token ID 0.
tokenizer.pad_token = (
    tokenizer.eos_token
)


print(
    f"EOS token ID: "
    f"{tokenizer.eos_token_id}"
)

print(
    f"PAD token ID: "
    f"{tokenizer.pad_token_id}"
)


# =========================================================
# LOAD JSONL
# =========================================================

def load_jsonl(file_path):

    examples = []

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    return examples


# =========================================================
# FORMAT EXAMPLE
# =========================================================
#
# Must EXACTLY match the training format.
#
# Otherwise validation would measure a different
# distribution from the one we trained on.
#
# =========================================================

def format_example(example):

    instruction = (
        example["instruction"]
        .strip()
    )

    response = (
        example["response"]
        .strip()
    )


    return (

        "### Instruction:\n"

        f"{instruction}"

        "\n\n"

        "### Response:\n"

        f"{response}"

        f"{tokenizer.eos_token}"
    )


# =========================================================
# TOKENIZE + FILTER
# =========================================================
#
# Same rule as training:
#
# <= 256 tokens → keep
# > 256 tokens  → skip
#
# We do NOT truncate.
#
# =========================================================

def prepare_examples(
    raw_examples
):

    prepared_examples = []

    skipped = 0


    for example in raw_examples:

        text = format_example(
            example
        )


        token_ids = (
            tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )


        if len(token_ids) > MAX_LENGTH:

            skipped += 1

            continue


        prepared_examples.append(

            torch.tensor(
                token_ids,
                dtype=torch.long
            )
        )


    return (
        prepared_examples,
        skipped
    )


# =========================================================
# COLLATOR
# =========================================================
#
# Same EOS/PAD fix as training.
#
# Real EOS:
#
# attention mask = 1
# label          = EOS ID
#
#
# Padding:
#
# attention mask = 0
# label          = -100
#
# =========================================================

def collate_batch(
    batch
):

    input_ids = pad_sequence(

        batch,

        batch_first=True,

        padding_value=(
            tokenizer.pad_token_id
        )
    )


    attention_sequences = [

        torch.ones_like(
            sequence
        )

        for sequence
        in batch
    ]


    attention_mask = pad_sequence(

        attention_sequences,

        batch_first=True,

        padding_value=0
    )


    labels = pad_sequence(

        batch,

        batch_first=True,

        padding_value=-100
    )


    return {

        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "labels":
            labels,
    }


# =========================================================
# PREPARE VALIDATION DATASET
# =========================================================

raw_validation_examples = (
    load_jsonl(
        VALIDATION_DATA_PATH
    )
)


(
    validation_examples,
    skipped_validation_examples

) = prepare_examples(
    raw_validation_examples
)


print(
    "\n========== VALIDATION DATASET =========="
)

print(
    f"Raw validation examples: "
    f"{len(raw_validation_examples):,}"
)

print(
    f"Usable <= {MAX_LENGTH}: "
    f"{len(validation_examples):,}"
)

print(
    f"Skipped > {MAX_LENGTH}: "
    f"{skipped_validation_examples:,}"
)


validation_loader = DataLoader(

    validation_examples,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_batch
)


# =========================================================
# MODEL LOADER
# =========================================================
#
# adapter_path = None
#     → Base SmolLM2
#
# adapter_path = security_lora_adapter
#     → Domain model
#
# adapter_path = security_sft_adapter
#     → SFT model
#
# =========================================================

def load_model(
    adapter_path=None
):

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME
        )
    )


    base_model.config.use_cache = False

    base_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )


    if adapter_path is None:

        model = base_model

    else:

        model = (
            PeftModel
            .from_pretrained(
                base_model,
                adapter_path
            )
        )


    model = model.to(
        device
    )

    model.eval()


    return model


# =========================================================
# EVALUATION FUNCTION
# =========================================================
#
# We calculate summed token-level cross entropy manually.
#
# Why?
#
# Different examples have different lengths.
#
# We want:
#
# total negative log likelihood
# --------------------------------
# total number of target tokens
#
# rather than treating every differently-sized batch
# as equally important.
#
# =========================================================

def evaluate_model(
    model,
    model_name
):

    total_negative_log_likelihood = 0.0

    total_target_tokens = 0


    print(
        f"\nEvaluating: {model_name}"
    )


    with torch.inference_mode():


        for batch in validation_loader:


            input_ids = (
                batch["input_ids"]
                .to(device)
            )


            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )


            labels = (
                batch["labels"]
                .to(device)
            )


            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask
            )


            logits = (
                outputs.logits
            )


            # ---------------------------------------------
            # Causal language modeling shift
            #
            # Token position t predicts token t+1.
            # ---------------------------------------------

            shift_logits = (
                logits[:, :-1, :]
                .contiguous()
            )


            shift_labels = (
                labels[:, 1:]
                .contiguous()
            )


            # ---------------------------------------------
            # Sum cross entropy over all REAL target tokens.
            #
            # Padding labels = -100 are automatically
            # ignored.
            # ---------------------------------------------

            batch_negative_log_likelihood = (
                F.cross_entropy(

                    shift_logits.view(
                        -1,
                        shift_logits.size(-1)
                    ),

                    shift_labels.view(-1),

                    ignore_index=-100,

                    reduction="sum"
                )
            )


            target_token_count = (

                shift_labels
                .ne(-100)
                .sum()
                .item()
            )


            total_negative_log_likelihood += (

                batch_negative_log_likelihood
                .item()
            )


            total_target_tokens += (
                target_token_count
            )


    average_loss = (

        total_negative_log_likelihood
        / total_target_tokens
    )


    perplexity = math.exp(
        average_loss
    )


    print(
        f"{model_name} validation loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"{model_name} perplexity: "
        f"{perplexity:.2f}"
    )

    print(
        f"Target tokens evaluated: "
        f"{total_target_tokens:,}"
    )


    return (
        average_loss,
        perplexity
    )


# =========================================================
# MEMORY CLEANUP
# =========================================================

def cleanup_model(
    model
):

    del model

    gc.collect()


    if torch.backends.mps.is_available():

        torch.mps.empty_cache()


# =========================================================
# 1. BASE MODEL
# =========================================================

print(
    "\n========== BASE MODEL =========="
)

base_model = load_model()


(
    base_loss,
    base_perplexity

) = evaluate_model(

    base_model,

    "BASE"
)


cleanup_model(
    base_model
)


# =========================================================
# 2. DOMAIN LoRA MODEL
# =========================================================

print(
    "\n========== DOMAIN MODEL =========="
)

domain_model = load_model(
    DOMAIN_ADAPTER_PATH
)


(
    domain_loss,
    domain_perplexity

) = evaluate_model(

    domain_model,

    "DOMAIN"
)


cleanup_model(
    domain_model
)


# =========================================================
# 3. SFT MODEL
# =========================================================

print(
    "\n========== SFT MODEL =========="
)

sft_model = load_model(
    SFT_ADAPTER_PATH
)


(
    sft_loss,
    sft_perplexity

) = evaluate_model(

    sft_model,

    "SFT"
)


cleanup_model(
    sft_model
)


# =========================================================
# COMPARISON
# =========================================================

print(
    "\n========== VALIDATION COMPARISON =========="
)


domain_vs_base_change = (

    (
        domain_loss
        - base_loss
    )

    / base_loss

) * 100


sft_vs_domain_change = (

    (
        sft_loss
        - domain_loss
    )

    / domain_loss

) * 100


sft_vs_base_change = (

    (
        sft_loss
        - base_loss
    )

    / base_loss

) * 100


print(
    f"BASE loss:   "
    f"{base_loss:.4f}"
)

print(
    f"DOMAIN loss: "
    f"{domain_loss:.4f}"
)

print(
    f"SFT loss:    "
    f"{sft_loss:.4f}"
)


print(
    "\nLoss change:"
)

print(
    f"DOMAIN vs BASE: "
    f"{domain_vs_base_change:+.2f}%"
)

print(
    f"SFT vs DOMAIN: "
    f"{sft_vs_domain_change:+.2f}%"
)

print(
    f"SFT vs BASE: "
    f"{sft_vs_base_change:+.2f}%"
)


print(
    "\nPerplexity:"
)

print(
    f"BASE:   "
    f"{base_perplexity:.2f}"
)

print(
    f"DOMAIN: "
    f"{domain_perplexity:.2f}"
)

print(
    f"SFT:    "
    f"{sft_perplexity:.2f}"
)


print(
    "\nSFT validation evaluation completed."
)