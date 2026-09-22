import re
from pathlib import Path

import torch
from transformers import AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

TRAIN_TEXT_PATH = Path(
    "data/processed/security_train.txt"
)

VALIDATION_TEXT_PATH = Path(
    "data/processed/security_validation.txt"
)

TRAIN_BLOCKS_PATH = Path(
    "data/processed/smollm2_security_train_blocks.pt"
)

VALIDATION_BLOCKS_PATH = Path(
    "data/processed/smollm2_security_validation_blocks.pt"
)


# ---------------------------------------------------------
# Training sequence length
# ---------------------------------------------------------

# SmolLM2 supports up to 8192 tokens.
#
# We deliberately start with only 128 because training
# memory depends strongly on sequence length.
#
# M2 + 8 GB unified memory -> start conservatively.
BLOCK_SIZE = 128


# ---------------------------------------------------------
# Load SmolLM2 tokenizer
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# Read security records
# ---------------------------------------------------------

def tokenize_text(file_path):

    text = file_path.read_text(
        encoding="utf-8"
    )

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    return token_ids


# ---------------------------------------------------------
# Split token stream into fixed training sequences
# ---------------------------------------------------------

def create_blocks(token_ids):

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

        blocks.append(block)

    return torch.tensor(
        blocks,
        dtype=torch.long
    )


# ---------------------------------------------------------
# Prepare one dataset split
# ---------------------------------------------------------

def prepare_split(
    source_path,
    output_path,
    split_name
):
    token_ids = tokenize_text(
        source_path
    )

    print(
        f"\n{split_name} tokens: "
        f"{len(token_ids):,}"
    )

    print(
        f"{split_name} tokens: "
        f"{len(token_ids):,}"
    )

    blocks = create_blocks(
        token_ids
    )

    print(
        f"{split_name} blocks: "
        f"{blocks.shape[0]:,}"
    )

    print(
        f"{split_name} block shape: "
        f"{tuple(blocks.shape)}"
    )

    torch.save(
        blocks,
        output_path
    )

    print(
        f"Saved: {output_path}"
    )

    return blocks


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

print(
    "========== SMOLLM2 SECURITY DATASET =========="
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Vocabulary size: {len(tokenizer):,}"
)

print(
    f"EOS token: {tokenizer.eos_token}"
)

print(
    f"EOS ID: {tokenizer.eos_token_id}"
)

print(
    f"Training block size: {BLOCK_SIZE}"
)


train_blocks = prepare_split(
    source_path=TRAIN_TEXT_PATH,
    output_path=TRAIN_BLOCKS_PATH,
    split_name="TRAIN"
)

validation_blocks = prepare_split(
    source_path=VALIDATION_TEXT_PATH,
    output_path=VALIDATION_BLOCKS_PATH,
    split_name="VALIDATION"
)


print(
    "\n========== SAMPLE BLOCK =========="
)

sample_ids = train_blocks[0].tolist()

print(
    tokenizer.decode(
        sample_ids,
        skip_special_tokens=False
    )
)