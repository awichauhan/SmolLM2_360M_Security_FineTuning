import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

MAX_NEW_TOKENS = 60


# ---------------------------------------------------------
# 1. Select device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ---------------------------------------------------------
# 2. Load tokenizer and pretrained model
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

model.eval()


# ---------------------------------------------------------
# 3. Baseline prompts
# ---------------------------------------------------------

PROMPTS = [
    # General-domain prompts
    "The capital of France is",
    "Machine learning is",

    # Security-domain prompts
    "A firewall is",
    "Malware is",
    "Authentication is",
    "A software vulnerability is",
]


# ---------------------------------------------------------
# 4. Generate text
# ---------------------------------------------------------

@torch.inference_mode()
def generate_text(prompt):

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = encoded_input["input_ids"].to(device)

    print(
        f"\nPrompt token count: "
        f"{input_ids.shape[1]}"
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,

        # Greedy generation for our first controlled baseline.
        do_sample=False,

        # Use the pretrained EOS token for padding if needed.
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return generated_text


# ---------------------------------------------------------
# 5. Run baseline experiment
# ---------------------------------------------------------

print("\n========== BASE MODEL GENERATION ==========")

for prompt in PROMPTS:

    print("\n" + "=" * 70)

    print(f"PROMPT: {repr(prompt)}")

    print("=" * 70)

    generated_text = generate_text(prompt)

    print("\nOUTPUT:\n")

    print(generated_text)