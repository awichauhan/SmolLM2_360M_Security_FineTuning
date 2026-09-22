import gc
import math

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

ADAPTER_PATH = (
    "artifacts/security_lora_adapter"
)

BLOCK_SIZE = 128

# Match our security validation experiment exactly.
NUMBER_OF_BLOCKS = 504

BATCH_SIZE = 2

RANDOM_SEED = 42


# =========================================================
# Device
# =========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =========================================================
# Tokenizer
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# Load general-domain validation dataset
# =========================================================

print(
    "\nLoading WikiText-2 validation split..."
)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="validation"
)

print(
    f"WikiText rows: {len(dataset):,}"
)


# =========================================================
# Tokenize the general-domain text
# =========================================================

token_ids = []


for row in dataset:

    text = row["text"]

    # WikiText contains some blank rows.
    if not text.strip():
        continue

    # Add newline so separate rows do not get
    # glued together unnaturally.
    row_ids = tokenizer.encode(
        text + "\n",
        add_special_tokens=False
    )

    token_ids.extend(
        row_ids
    )


print(
    f"WikiText tokens: "
    f"{len(token_ids):,}"
)


# =========================================================
# Create fixed 128-token blocks
# =========================================================

blocks = []


for start_index in range(
    0,
    len(token_ids) - BLOCK_SIZE + 1,
    BLOCK_SIZE
):

    end_index = (
        start_index + BLOCK_SIZE
    )

    block = token_ids[
        start_index:end_index
    ]

    blocks.append(
        block
    )


blocks = torch.tensor(
    blocks,
    dtype=torch.long
)


print(
    f"Available blocks: "
    f"{blocks.shape[0]:,}"
)


# =========================================================
# Deterministically select 504 blocks
# =========================================================

generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)


random_indices = torch.randperm(
    len(blocks),
    generator=generator
)


selected_indices = random_indices[
    :NUMBER_OF_BLOCKS
]


general_validation_blocks = blocks[
    selected_indices
]


print(
    f"Selected validation blocks: "
    f"{general_validation_blocks.shape}"
)


# =========================================================
# Generic evaluation function
# =========================================================

@torch.inference_mode()
def evaluate_model(
    model,
    validation_blocks
):

    model.eval()

    total_loss = 0.0
    number_of_batches = 0

    for start_index in range(
        0,
        len(validation_blocks),
        BATCH_SIZE
    ):

        batch = validation_blocks[
            start_index:
            start_index + BATCH_SIZE
        ].to(device)

        outputs = model(
            input_ids=batch,
            labels=batch
        )

        total_loss += (
            outputs.loss.item()
        )

        number_of_batches += 1

        if (
            number_of_batches % 50
            == 0
        ):
            print(
                f"Processed "
                f"{number_of_batches} batches..."
            )


    average_loss = (
        total_loss
        / number_of_batches
    )

    perplexity = math.exp(
        average_loss
    )

    return (
        average_loss,
        perplexity
    )


# =========================================================
# 1. Evaluate untouched BASE model
# =========================================================

print(
    "\n========== BASE MODEL =========="
)


base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)

base_model.config.use_cache = False

base_model = base_model.to(
    device
)


base_loss, base_perplexity = (
    evaluate_model(
        model=base_model,
        validation_blocks=general_validation_blocks
    )
)


print(
    f"\nBase general loss: "
    f"{base_loss:.4f}"
)

print(
    f"Base general perplexity: "
    f"{base_perplexity:.2f}"
)


# =========================================================
# Free memory before loading LoRA model
# =========================================================

del base_model

gc.collect()

if device.type == "mps":
    torch.mps.empty_cache()


# =========================================================
# 2. Evaluate SECURITY LoRA model
# =========================================================

print(
    "\n========== SECURITY LoRA MODEL =========="
)


lora_base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)

lora_base_model.config.use_cache = False


lora_model = PeftModel.from_pretrained(
    lora_base_model,
    ADAPTER_PATH
)

lora_model = lora_model.to(
    device
)


lora_loss, lora_perplexity = (
    evaluate_model(
        model=lora_model,
        validation_blocks=general_validation_blocks
    )
)


print(
    f"\nLoRA general loss: "
    f"{lora_loss:.4f}"
)

print(
    f"LoRA general perplexity: "
    f"{lora_perplexity:.2f}"
)


# =========================================================
# Compare
# =========================================================

loss_change = (
    lora_loss
    - base_loss
)


loss_change_percent = (
    loss_change
    / base_loss
) * 100


print(
    "\n========== GENERAL-DOMAIN COMPARISON =========="
)

print(
    f"Base loss: "
    f"{base_loss:.4f}"
)

print(
    f"Security LoRA loss: "
    f"{lora_loss:.4f}"
)

print(
    f"Loss change: "
    f"{loss_change:+.4f} "
    f"({loss_change_percent:+.2f}%)"
)

print()

print(
    f"Base perplexity: "
    f"{base_perplexity:.2f}"
)

print(
    f"Security LoRA perplexity: "
    f"{lora_perplexity:.2f}"
)