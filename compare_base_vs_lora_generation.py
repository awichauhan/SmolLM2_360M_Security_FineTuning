import gc

import torch

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

MAX_NEW_TOKENS = 80


# =========================================================
# Prompts
# =========================================================

PROMPTS = [

    # Prompts already used in our original baseline.
    "A firewall is",
    "Malware is",
    "Authentication is",
    "A software vulnerability is",

    # More domain-specific prompts.
    "Credential dumping is",
    "Privilege escalation is",
    "Supply chain risk is",
    "An intrusion set is",
]


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
# Generation function
# =========================================================

@torch.inference_mode()
def generate_text(
    model,
    prompt
):

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = (
        encoded_input["input_ids"]
        .to(device)
    )

    generated_ids = model.generate(

        input_ids=input_ids,

        max_new_tokens=MAX_NEW_TOKENS,

        # Greedy decoding.
        #
        # We deliberately avoid sampling so that
        # differences come from model parameters,
        # not random token selection.
        do_sample=False,

        pad_token_id=tokenizer.eos_token_id,

        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return generated_text


# =========================================================
# Run all prompts for one model
# =========================================================

def run_generation_suite(
    model,
    model_name
):

    model.eval()

    results = {}

    print(
        f"\n========== {model_name} =========="
    )

    for prompt in PROMPTS:

        print(
            "\n" + "=" * 70
        )

        print(
            f"PROMPT: {repr(prompt)}"
        )

        output = generate_text(
            model=model,
            prompt=prompt
        )

        results[prompt] = output

        print("\nOUTPUT:\n")

        print(output)

    return results


# =========================================================
# Load BASE model
# =========================================================

print(
    "\nLoading BASE SmolLM2..."
)

base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)

base_model = base_model.to(
    device
)


base_results = run_generation_suite(
    model=base_model,
    model_name="BASE MODEL"
)


# =========================================================
# Free base model memory
# =========================================================

del base_model

gc.collect()

if device.type == "mps":
    torch.mps.empty_cache()


# =========================================================
# Load base model again
# =========================================================

print(
    "\nLoading SmolLM2 + SECURITY LoRA..."
)

lora_base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)


# =========================================================
# Attach trained security adapter
# =========================================================

lora_model = PeftModel.from_pretrained(
    lora_base_model,
    ADAPTER_PATH
)

lora_model = lora_model.to(
    device
)


lora_results = run_generation_suite(
    model=lora_model,
    model_name="SECURITY LoRA MODEL"
)


# =========================================================
# Side-by-side comparison
# =========================================================

print(
    "\n\n========== FINAL COMPARISON =========="
)


for prompt in PROMPTS:

    print(
        "\n" + "=" * 80
    )

    print(
        f"PROMPT: {repr(prompt)}"
    )

    print(
        "\n----- BASE -----\n"
    )

    print(
        base_results[prompt]
    )

    print(
        "\n----- SECURITY LoRA -----\n"
    )

    print(
        lora_results[prompt]
    )