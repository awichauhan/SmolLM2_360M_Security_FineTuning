import math

import torch
from transformers import AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

VALIDATION_BLOCKS_PATH = (
    "data/processed/"
    "smollm2_security_validation_blocks.pt"
)

BATCH_SIZE = 2


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ---------------------------------------------------------
# Load validation data
# ---------------------------------------------------------

validation_blocks = torch.load(
    VALIDATION_BLOCKS_PATH
)

print(
    f"Validation blocks: "
    f"{validation_blocks.shape}"
)


# ---------------------------------------------------------
# Load untouched pretrained model
# ---------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

model.eval()


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

total_loss = 0.0
number_of_batches = 0


@torch.inference_mode()
def evaluate():

    global total_loss
    global number_of_batches

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

            # For causal language modelling,
            # labels are the same sequence.
            # Hugging Face performs the target shift internally.
            labels=batch
        )

        total_loss += outputs.loss.item()

        number_of_batches += 1

        if number_of_batches % 50 == 0:
            print(
                f"Processed "
                f"{number_of_batches} batches..."
            )


evaluate()


# ---------------------------------------------------------
# Final statistics
# ---------------------------------------------------------

average_loss = (
    total_loss / number_of_batches
)

perplexity = math.exp(
    average_loss
)


print(
    "\n========== SECURITY BASELINE =========="
)

print(
    f"Validation loss: "
    f"{average_loss:.4f}"
)

print(
    f"Perplexity: "
    f"{perplexity:.2f}"
)