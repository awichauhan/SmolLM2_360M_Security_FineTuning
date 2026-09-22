import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


# ---------------------------------------------------------
# 1. Select device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ---------------------------------------------------------
# 2. Load tokenizer
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("\nTokenizer loaded.")


# ---------------------------------------------------------
# 3. Load pretrained causal language model
# ---------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

print("Model loaded.")


# ---------------------------------------------------------
# 4. Put model into inference mode
# ---------------------------------------------------------

model.eval()


# ---------------------------------------------------------
# 5. Count model parameters
# ---------------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n========== MODEL INFORMATION ==========")

print(f"Model: {MODEL_NAME}")
print(f"Device: {device}")

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


# ---------------------------------------------------------
# 6. Inspect architecture configuration
# ---------------------------------------------------------

print("\n========== CONFIGURATION ==========")

print(f"Vocabulary size: {model.config.vocab_size}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Attention heads: {model.config.num_attention_heads}")

if hasattr(model.config, "num_key_value_heads"):
    print(
        f"KV heads: "
        f"{model.config.num_key_value_heads}"
    )

print(
    f"Maximum context length: "
    f"{model.config.max_position_embeddings}"
)


# ---------------------------------------------------------
# 7. Inspect tokenizer
# ---------------------------------------------------------

print("\n========== TOKENIZER ==========")

print(
    f"Tokenizer vocabulary size: "
    f"{len(tokenizer)}"
)

print(
    f"EOS token: "
    f"{tokenizer.eos_token}"
)

print(
    f"EOS token ID: "
    f"{tokenizer.eos_token_id}"
)


# ---------------------------------------------------------
# 8. Simple tokenization test
# ---------------------------------------------------------

text = "Authentication protects access to computer systems."

token_ids = tokenizer.encode(text)

print("\n========== TOKENIZATION TEST ==========")

print(f"Text: {text}")
print(f"Token IDs: {token_ids}")
print(f"Token count: {len(token_ids)}")

print(
    "Tokens:",
    tokenizer.convert_ids_to_tokens(token_ids)
)