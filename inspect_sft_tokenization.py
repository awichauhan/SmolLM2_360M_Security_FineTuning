import json
from pathlib import Path

from transformers import AutoTokenizer


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "instruction"
    / "security_sft_train.jsonl"
)

VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "instruction"
    / "security_sft_validation.jsonl"
)


# =========================================================
# Load tokenizer
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    "========== SFT TOKENIZATION INSPECTION =========="
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
    f"EOS token ID: {tokenizer.eos_token_id}"
)


# =========================================================
# Load JSONL
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


train_examples = load_jsonl(
    TRAIN_DATA_PATH
)

validation_examples = load_jsonl(
    VALIDATION_DATA_PATH
)


print(
    f"\nTrain examples: "
    f"{len(train_examples):,}"
)

print(
    f"Validation examples: "
    f"{len(validation_examples):,}"
)


# =========================================================
# Format one SFT example
# =========================================================
#
# The base model has no magical understanding of
# "instruction" and "response" JSON fields.
#
# We must turn them into actual text.
#
# EOS tells the model:
# "this training example ends here".
# =========================================================

def format_example(example):

    instruction = example[
        "instruction"
    ].strip()

    response = example[
        "response"
    ].strip()

    text = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{response}"
        f"{tokenizer.eos_token}"
    )

    return text


# =========================================================
# Tokenize one example
# =========================================================

def tokenize_example(example):

    formatted_text = format_example(
        example
    )

    input_ids = tokenizer.encode(
        formatted_text,
        add_special_tokens=False
    )

    # STANDARD SFT:
    #
    # labels == input_ids
    #
    # This means loss is calculated across
    # BOTH instruction and response tokens.
    labels = input_ids.copy()

    attention_mask = [
        1
        for _ in input_ids
    ]

    return {
        "text": formatted_text,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# =========================================================
# Percentile helper
# =========================================================

def percentile(
    sorted_values,
    percentile_value
):

    index = int(
        (
            len(sorted_values) - 1
        )
        * percentile_value
    )

    return sorted_values[index]


# =========================================================
# Calculate token-length distribution
# =========================================================

token_lengths = []


for example in train_examples:

    tokenized = tokenize_example(
        example
    )

    token_lengths.append(
        len(
            tokenized["input_ids"]
        )
    )


sorted_lengths = sorted(
    token_lengths
)


print(
    "\n========== TOKEN LENGTH DISTRIBUTION =========="
)

print(
    f"Minimum: "
    f"{min(sorted_lengths):,}"
)

print(
    f"Median: "
    f"{percentile(sorted_lengths, 0.50):,}"
)

print(
    f"75th percentile: "
    f"{percentile(sorted_lengths, 0.75):,}"
)

print(
    f"90th percentile: "
    f"{percentile(sorted_lengths, 0.90):,}"
)

print(
    f"95th percentile: "
    f"{percentile(sorted_lengths, 0.95):,}"
)

print(
    f"99th percentile: "
    f"{percentile(sorted_lengths, 0.99):,}"
)

print(
    f"Maximum: "
    f"{max(sorted_lengths):,}"
)


# =========================================================
# Check candidate sequence lengths
# =========================================================

print(
    "\n========== TRUNCATION ANALYSIS =========="
)


for max_length in [
    128,
    256,
    512,
    1024,
]:

    too_long = sum(
        1
        for length in token_lengths
        if length > max_length
    )

    percentage = (
        too_long
        / len(token_lengths)
    ) * 100

    print(
        f"Over {max_length:4d} tokens: "
        f"{too_long:5,d} "
        f"({percentage:6.2f}%)"
    )


# =========================================================
# Inspect one real example
# =========================================================

sample_example = train_examples[0]

sample = tokenize_example(
    sample_example
)


print(
    "\n========== SAMPLE SFT EXAMPLE =========="
)

print(
    "\nRAW JSON:"
)

print(
    sample_example
)


print(
    "\nFORMATTED TEXT:"
)

print(
    sample["text"]
)


print(
    "\nTOKEN COUNT:"
)

print(
    len(
        sample["input_ids"]
    )
)


print(
    "\nFIRST 40 TOKEN IDs:"
)

print(
    sample["input_ids"][:40]
)


print(
    "\nFIRST 40 TOKEN PIECES:"
)

print(
    tokenizer.convert_ids_to_tokens(
        sample["input_ids"][:40]
    )
)


print(
    "\nINPUT/LABEL CHECK:"
)

print(
    "input_ids == labels:",
    (
        sample["input_ids"]
        ==
        sample["labels"]
    )
)


print(
    "\nATTENTION MASK LENGTH:"
)

print(
    len(
        sample["attention_mask"]
    )
)


# =========================================================
# Inspect longest example
# =========================================================

longest_index = token_lengths.index(
    max(token_lengths)
)

longest_example = train_examples[
    longest_index
]

longest_tokenized = tokenize_example(
    longest_example
)


print(
    "\n========== LONGEST TRAINING EXAMPLE =========="
)

print(
    f"Token count: "
    f"{len(longest_tokenized['input_ids']):,}"
)

print(
    f"Instruction: "
    f"{longest_example['instruction']}"
)

print(
    "\nResponse preview:"
)

print(
    longest_example["response"][:1000]
)