import json
from pathlib import Path

import torch

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


# -------------------------
# Dataset
# -------------------------

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "instruction"
    / "security_sft_train.jsonl"
)


# -------------------------
# Existing domain adapter
# -------------------------

DOMAIN_ADAPTER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "security_lora_adapter"
)


# -------------------------
# New SFT adapter
# -------------------------

SFT_ADAPTER_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "security_sft_adapter"
)


# -------------------------
# Training configuration
# -------------------------

MAX_LENGTH = 256

BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-4

MAX_OPTIMIZER_STEPS = 300

REPORT_INTERVAL = 25

RANDOM_SEED = 42


torch.manual_seed(
    RANDOM_SEED
)


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():

    device = torch.device(
        "mps"
    )

else:

    device = torch.device(
        "cpu"
    )


print(
    "========== STANDARD SECURITY SFT =========="
)

print(
    f"Using device: {device}"
)

print(
    f"MAX_LENGTH: {MAX_LENGTH}"
)

print(
    f"BATCH_SIZE: {BATCH_SIZE}"
)

print(
    f"GRADIENT_ACCUMULATION_STEPS: "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"EFFECTIVE_BATCH_SIZE: "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"LEARNING_RATE: {LEARNING_RATE}"
)

print(
    f"MAX_OPTIMIZER_STEPS: "
    f"{MAX_OPTIMIZER_STEPS}"
)

print(
    f"REPORT_INTERVAL: "
    f"{REPORT_INTERVAL}"
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


# SmolLM2 does not have a separate PAD token.
#
# We reuse EOS as PAD.
#
# IMPORTANT:
# Because EOS and PAD now share the same token ID,
# we must NOT create the attention mask by checking
# token IDs.
#
# We handle this correctly later in collate_batch().
tokenizer.pad_token = (
    tokenizer.eos_token
)


print(
    "\n========== TOKENIZER =========="
)

print(
    f"Vocabulary size: "
    f"{len(tokenizer):,}"
)

print(
    f"EOS token: "
    f"{tokenizer.eos_token}"
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
# LOAD JSONL DATASET
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

            example = json.loads(
                line
            )

            examples.append(
                example
            )

    return examples


# =========================================================
# FORMAT ONE SFT EXAMPLE
# =========================================================
#
# Original JSON:
#
# {
#     "instruction": "What is malware?",
#     "response": "Malware is..."
# }
#
#
# Converted into:
#
# ### Instruction:
# What is malware?
#
# ### Response:
# Malware is...
# <|endoftext|>
#
# =========================================================

def format_example(example):

    instruction = (
        example[
            "instruction"
        ]
        .strip()
    )

    response = (
        example[
            "response"
        ]
        .strip()
    )


    formatted_text = (

        "### Instruction:\n"

        f"{instruction}"

        "\n\n"

        "### Response:\n"

        f"{response}"

        f"{tokenizer.eos_token}"
    )


    return formatted_text


# =========================================================
# TOKENIZE + FILTER DATASET
# =========================================================
#
# Decision from our previous experiment:
#
# 96.87% of examples fit inside 256 tokens.
#
# Instead of truncating long responses, we skip them.
#
# This guarantees every retained training example has:
#
# complete instruction
# complete response
# EOS
#
# =========================================================

def prepare_training_examples(
    raw_examples
):

    prepared_examples = []

    skipped_long_examples = 0


    for example in raw_examples:

        formatted_text = (
            format_example(
                example
            )
        )


        input_ids = (
            tokenizer.encode(
                formatted_text,
                add_special_tokens=False
            )
        )


        # ---------------------------------------------
        # Skip examples longer than 256 tokens.
        #
        # We intentionally do NOT truncate them.
        # ---------------------------------------------

        if (
            len(input_ids)
            > MAX_LENGTH
        ):

            skipped_long_examples += 1

            continue


        input_tensor = (
            torch.tensor(
                input_ids,
                dtype=torch.long
            )
        )


        prepared_examples.append(
            input_tensor
        )


    return (
        prepared_examples,
        skipped_long_examples
    )


# =========================================================
# DYNAMIC PADDING COLLATOR
# =========================================================
#
# This function prepares a batch for the Transformer.
#
#
# Example:
#
# sequence A = 40 tokens
# sequence B = 55 tokens
#
# We pad only to 55.
#
# We do NOT pad everything permanently to 256.
#
#
# IMPORTANT EOS/PAD DETAIL:
#
# EOS ID = 0
# PAD ID = 0
#
# Therefore:
#
# input_ids != pad_token_id
#
# CANNOT be used to build the attention mask.
#
# That would incorrectly classify the REAL EOS token
# as padding.
#
# Instead, we build attention masks from the original
# sequence lengths before padding.
#
# =========================================================

def collate_batch(
    batch
):

    # -----------------------------------------------------
    # INPUT IDS
    #
    # Pad all examples to the longest example in this batch.
    # -----------------------------------------------------

    input_ids = (
        pad_sequence(
            batch,
            batch_first=True,
            padding_value=(
                tokenizer.pad_token_id
            )
        )
    )


    # -----------------------------------------------------
    # ATTENTION MASK
    #
    # Every original token gets 1.
    #
    # This includes the REAL EOS token.
    #
    # Only newly-added padding gets 0.
    # -----------------------------------------------------

    attention_mask_sequences = [

        torch.ones_like(
            sequence
        )

        for sequence
        in batch
    ]


    attention_mask = (
        pad_sequence(
            attention_mask_sequences,
            batch_first=True,
            padding_value=0
        )
    )


    # -----------------------------------------------------
    # STANDARD SFT LABELS
    #
    # Every real token contributes to loss:
    #
    # Instruction tokens → loss
    # Response tokens    → loss
    # EOS token          → loss
    #
    # Padding uses -100.
    #
    # PyTorch CrossEntropyLoss ignores -100.
    # -----------------------------------------------------

    labels = (
        pad_sequence(
            batch,
            batch_first=True,
            padding_value=-100
        )
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
# PREPARE TRAINING DATA
# =========================================================

raw_train_examples = (
    load_jsonl(
        TRAIN_DATA_PATH
    )
)


(
    train_examples,
    skipped_examples

) = prepare_training_examples(
    raw_train_examples
)


print(
    "\n========== SFT DATASET =========="
)

print(
    f"Raw train examples: "
    f"{len(raw_train_examples):,}"
)

print(
    f"Usable <= {MAX_LENGTH}: "
    f"{len(train_examples):,}"
)

print(
    f"Skipped > {MAX_LENGTH}: "
    f"{skipped_examples:,}"
)


# =========================================================
# SANITY CHECK
# =========================================================

first_example = (
    train_examples[0]
)


print(
    "\n========== DATA SANITY CHECK =========="
)

print(
    f"First example length: "
    f"{len(first_example)}"
)

print(
    f"Last token ID: "
    f"{first_example[-1].item()}"
)

print(
    f"Expected EOS ID: "
    f"{tokenizer.eos_token_id}"
)


if (
    first_example[-1].item()
    != tokenizer.eos_token_id
):

    raise RuntimeError(
        "Training example does not end "
        "with the EOS token."
    )


print(
    "EOS check: PASSED"
)


# =========================================================
# DATALOADER
# =========================================================
#
# shuffle=True:
#
# Training examples appear in a different random order.
#
#
# Generator gives reproducible shuffling.
#
# =========================================================

data_generator = (
    torch.Generator()
)

data_generator.manual_seed(
    RANDOM_SEED
)


train_loader = (
    DataLoader(

        train_examples,

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=collate_batch,

        generator=data_generator
    )
)


# =========================================================
# LOAD BASE MODEL
# =========================================================

print(
    "\nLoading base SmolLM2..."
)


base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)


# During training we do not need KV cache.
#
# KV cache is mainly an inference optimization.
base_model.config.use_cache = False


# Tell model config which token ID acts as padding.
base_model.config.pad_token_id = (
    tokenizer.pad_token_id
)


# =========================================================
# LOAD EXISTING SECURITY DOMAIN LoRA
# =========================================================
#
# Our progression:
#
# Base SmolLM2
#
#       +
#
# security_lora_adapter
#       ↓
# security-domain adapted model
#
#
# is_trainable=True means:
#
# reload LoRA matrices
# AND allow gradients to modify them again.
#
# Base SmolLM2 remains frozen.
#
# =========================================================

print(
    "\nLoading security domain adapter..."
)


model = (
    PeftModel
    .from_pretrained(

        base_model,

        DOMAIN_ADAPTER_PATH,

        is_trainable=True
    )
)


model = (
    model.to(
        device
    )
)


# =========================================================
# TRAINABLE PARAMETERS
# =========================================================

print(
    "\n========== TRAINABLE PARAMETERS =========="
)


model.print_trainable_parameters()


trainable_parameters = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad
]


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = (
    torch.optim.AdamW(

        trainable_parameters,

        lr=LEARNING_RATE
    )
)


# =========================================================
# TRAINING STATE
# =========================================================

model.train()


optimizer.zero_grad(
    set_to_none=True
)


optimizer_step = 0

micro_step = 0


# ---------------------------------------------
# Loss accumulated across 4 micro-batches.
# ---------------------------------------------

accumulated_loss = 0.0


# ---------------------------------------------
# Used for smooth 25-step reporting.
# ---------------------------------------------

report_loss = 0.0

report_steps = 0


print(
    "\n========== STANDARD SFT TRAINING =========="
)


# =========================================================
# TRAINING LOOP
# =========================================================

for batch in train_loader:


    # -----------------------------------------------------
    # Move tensors to MPS / CPU
    # -----------------------------------------------------

    input_ids = (
        batch[
            "input_ids"
        ]
        .to(device)
    )


    attention_mask = (
        batch[
            "attention_mask"
        ]
        .to(device)
    )


    labels = (
        batch[
            "labels"
        ]
        .to(device)
    )


    # -----------------------------------------------------
    # FORWARD PASS
    #
    # Because labels are provided,
    # Hugging Face automatically calculates
    # causal language-model cross entropy.
    # -----------------------------------------------------

    outputs = (
        model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            labels=labels
        )
    )


    original_loss = (
        outputs.loss
    )


    # -----------------------------------------------------
    # GRADIENT ACCUMULATION
    #
    # Effective batch:
    #
    # batch_size 1
    # × 4 micro-batches
    #
    # = 4 examples per optimizer update
    #
    # Divide loss by 4 so accumulated gradients have
    # approximately the correct scale.
    # -----------------------------------------------------

    scaled_loss = (

        original_loss

        / GRADIENT_ACCUMULATION_STEPS
    )


    # -----------------------------------------------------
    # BACKPROPAGATION
    # -----------------------------------------------------

    scaled_loss.backward()


    accumulated_loss += (
        original_loss.item()
    )


    micro_step += 1


    # =====================================================
    # OPTIMIZER UPDATE
    # =====================================================

    if (
        micro_step
        % GRADIENT_ACCUMULATION_STEPS
        == 0
    ):


        # -------------------------------------------------
        # Gradient clipping
        #
        # Prevent unusually large gradients from causing
        # unstable updates.
        # -------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            trainable_parameters,

            max_norm=1.0
        )


        # -------------------------------------------------
        # Update LoRA parameters
        # -------------------------------------------------

        optimizer.step()


        # -------------------------------------------------
        # Clear old gradients
        # -------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        optimizer_step += 1


        # -------------------------------------------------
        # Average loss across the 4 micro-batches
        # that produced this optimizer update.
        # -------------------------------------------------

        optimizer_average_loss = (

            accumulated_loss

            / GRADIENT_ACCUMULATION_STEPS
        )


        # -------------------------------------------------
        # Add this optimizer step's loss into
        # the reporting window.
        # -------------------------------------------------

        report_loss += (
            optimizer_average_loss
        )

        report_steps += 1


        # -------------------------------------------------
        # Print step 1 separately.
        #
        # Do NOT reset report_loss here.
        #
        # This allows step 25 to represent the actual
        # mean of steps 1 → 25.
        # -------------------------------------------------

        if optimizer_step == 1:

            print(

                f"Optimizer step "
                f"{optimizer_step:03d} | "
                f"SFT loss: "
                f"{optimizer_average_loss:.4f}"
            )


        # -------------------------------------------------
        # Every 25 optimizer steps:
        #
        # calculate average training loss over the
        # complete reporting window.
        # -------------------------------------------------

        if (
            optimizer_step
            % REPORT_INTERVAL
            == 0
        ):


            mean_report_loss = (

                report_loss

                / report_steps
            )


            print(

                f"Optimizer step "
                f"{optimizer_step:03d} | "
                f"Average SFT loss: "
                f"{mean_report_loss:.4f}"
            )


            # Start new reporting window.

            report_loss = 0.0

            report_steps = 0


        # -------------------------------------------------
        # Reset gradient-accumulation loss.
        # -------------------------------------------------

        accumulated_loss = 0.0


        # -------------------------------------------------
        # Stop after exactly 300 optimizer updates.
        # -------------------------------------------------

        if (
            optimizer_step
            >= MAX_OPTIMIZER_STEPS
        ):

            break


# =========================================================
# VERIFY TRAINING COMPLETED
# =========================================================

print(
    "\n========== TRAINING COMPLETE =========="
)

print(
    f"Optimizer steps completed: "
    f"{optimizer_step}"
)


if (
    optimizer_step
    != MAX_OPTIMIZER_STEPS
):

    raise RuntimeError(

        "Training ended before reaching "
        f"{MAX_OPTIMIZER_STEPS} optimizer steps."
    )


# =========================================================
# SAVE SFT ADAPTER
# =========================================================
#
# IMPORTANT:
#
# We do NOT overwrite:
#
# artifacts/security_lora_adapter
#
#
# The new checkpoint is:
#
# artifacts/security_sft_adapter
#
# =========================================================

print(
    "\n========== SAVING SFT ADAPTER =========="
)


SFT_ADAPTER_OUTPUT_PATH.mkdir(

    parents=True,

    exist_ok=True
)


model.save_pretrained(
    SFT_ADAPTER_OUTPUT_PATH
)


tokenizer.save_pretrained(
    SFT_ADAPTER_OUTPUT_PATH
)


print(
    "Saved SFT adapter to:"
)

print(
    SFT_ADAPTER_OUTPUT_PATH
)


print(
    "\nStandard SFT training completed successfully."
)