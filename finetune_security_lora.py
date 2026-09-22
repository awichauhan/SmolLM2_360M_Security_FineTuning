import math
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

TRAIN_BLOCKS_PATH = (
    "data/processed/"
    "smollm2_security_train_blocks.pt"
)

VALIDATION_BLOCKS_PATH = (
    "data/processed/"
    "smollm2_security_validation_blocks.pt"
)

ADAPTER_OUTPUT_PATH = Path(
    "artifacts/security_lora_adapter"
)


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4

MAX_OPTIMIZER_STEPS = 200

REPORT_INTERVAL = 20


# ---------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------

VALIDATION_BATCH_SIZE = 2


# =========================================================
# Reproducibility
# =========================================================

torch.manual_seed(42)


# =========================================================
# Device
# =========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =========================================================
# Load dataset
# =========================================================

train_blocks = torch.load(
    TRAIN_BLOCKS_PATH
)

validation_blocks = torch.load(
    VALIDATION_BLOCKS_PATH
)


print(
    f"Training blocks: "
    f"{train_blocks.shape}"
)

print(
    f"Validation blocks: "
    f"{validation_blocks.shape}"
)


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

train_loader = DataLoader(
    train_blocks,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================================================
# Load pretrained SmolLM2
# =========================================================

print(
    "\nLoading pretrained SmolLM2..."
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

# KV cache is used for autoregressive inference.
# It is unnecessary during training.
model.config.use_cache = False


# =========================================================
# LoRA configuration
# =========================================================

lora_config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    # Low-rank dimension.
    r=8,

    # Scales the LoRA update.
    lora_alpha=16,

    lora_dropout=0.05,

    bias="none",

    # Apply LoRA to the attention projections.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


# =========================================================
# Attach LoRA adapters
# =========================================================

model = get_peft_model(
    model,
    lora_config
)

model = model.to(device)


print(
    "\n========== TRAINABLE PARAMETERS =========="
)

model.print_trainable_parameters()


# =========================================================
# Collect trainable parameters
# =========================================================

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE
)


# =========================================================
# Validation function
# =========================================================

@torch.inference_mode()
def evaluate_security_loss():

    model.eval()

    total_loss = 0.0
    number_of_batches = 0

    for start_index in range(
        0,
        len(validation_blocks),
        VALIDATION_BATCH_SIZE
    ):

        batch = validation_blocks[
            start_index:
            start_index + VALIDATION_BATCH_SIZE
        ].to(device)

        outputs = model(
            input_ids=batch,
            labels=batch
        )

        total_loss += (
            outputs.loss.item()
        )

        number_of_batches += 1

    average_loss = (
        total_loss
        / number_of_batches
    )

    perplexity = math.exp(
        average_loss
    )

    # Switch back into training mode.
    model.train()

    return average_loss, perplexity


# =========================================================
# Training setup
# =========================================================

model.train()

optimizer.zero_grad(
    set_to_none=True
)

optimizer_step = 0
micro_step = 0

interval_loss = 0.0
interval_micro_batches = 0


print(
    "\n========== SECURITY LoRA FINE-TUNING =========="
)

print(
    f"Target optimizer steps: "
    f"{MAX_OPTIMIZER_STEPS}"
)

print(
    f"Gradient accumulation: "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Effective batch size: "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)


# =========================================================
# Training loop
# =========================================================

for batch in train_loader:

    batch = batch.to(device)

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    outputs = model(
        input_ids=batch,
        labels=batch
    )

    original_loss = outputs.loss


    # -----------------------------------------------------
    # Scale loss for gradient accumulation
    # -----------------------------------------------------

    loss = (
        original_loss
        / GRADIENT_ACCUMULATION_STEPS
    )


    # -----------------------------------------------------
    # Backpropagation
    # -----------------------------------------------------

    loss.backward()


    # -----------------------------------------------------
    # Track unscaled loss for readable reporting
    # -----------------------------------------------------

    interval_loss += (
        original_loss.item()
    )

    interval_micro_batches += 1

    micro_step += 1


    # -----------------------------------------------------
    # Update parameters only after enough micro-batches
    # -----------------------------------------------------

    if (
        micro_step
        % GRADIENT_ACCUMULATION_STEPS
        == 0
    ):

        # Prevent unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        optimizer_step += 1


        # -------------------------------------------------
        # Progress report
        # -------------------------------------------------

        if (
            optimizer_step
            % REPORT_INTERVAL
            == 0
        ):

            average_training_loss = (
                interval_loss
                / interval_micro_batches
            )

            print(
                f"Step "
                f"{optimizer_step:03d} | "
                f"Train loss: "
                f"{average_training_loss:.4f}"
            )

            interval_loss = 0.0
            interval_micro_batches = 0


        # -------------------------------------------------
        # Stop at configured number of optimizer updates
        # -------------------------------------------------

        if (
            optimizer_step
            >= MAX_OPTIMIZER_STEPS
        ):
            break


# =========================================================
# Save LoRA adapter
# =========================================================

ADAPTER_OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

model.save_pretrained(
    ADAPTER_OUTPUT_PATH
)

print(
    "\nLoRA adapter saved to:"
)

print(
    ADAPTER_OUTPUT_PATH
)


# =========================================================
# Final security evaluation
# =========================================================

print(
    "\n========== FINAL SECURITY EVALUATION =========="
)

validation_loss, perplexity = (
    evaluate_security_loss()
)


print(
    f"Security validation loss: "
    f"{validation_loss:.4f}"
)

print(
    f"Security perplexity: "
    f"{perplexity:.2f}"
)


# =========================================================
# Compare against untouched base model
# =========================================================

BASELINE_LOSS = 2.9617
BASELINE_PERPLEXITY = 19.33


loss_change = (
    validation_loss
    - BASELINE_LOSS
)

loss_change_percent = (
    loss_change
    / BASELINE_LOSS
) * 100


print(
    "\n========== BEFORE vs AFTER =========="
)

print(
    f"Base loss: "
    f"{BASELINE_LOSS:.4f}"
)

print(
    f"LoRA loss: "
    f"{validation_loss:.4f}"
)

print(
    f"Loss change: "
    f"{loss_change:+.4f} "
    f"({loss_change_percent:+.2f}%)"
)

print()

print(
    f"Base perplexity: "
    f"{BASELINE_PERPLEXITY:.2f}"
)

print(
    f"LoRA perplexity: "
    f"{perplexity:.2f}"
)