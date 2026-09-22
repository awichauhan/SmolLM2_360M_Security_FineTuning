from pathlib import Path
import hashlib
import html
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parent


TRAIN_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_train.txt"
)

VALIDATION_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_validation.txt"
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "instruction"
)


TRAIN_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "security_sft_train.jsonl"
)

VALIDATION_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "security_sft_validation.jsonl"
)

# =========================================================
# Instruction templates
# =========================================================
#
# We are NOT generating new security knowledge here.
#
# The response will still come directly from our
# MITRE/NIST source text.
#
# We only vary the way the instruction is phrased.
# =========================================================

INSTRUCTION_TEMPLATES = [
    "What is {concept}?",
    "Explain {concept}.",
    "Define {concept} in cybersecurity.",
    "Describe {concept} in a cybersecurity context.",
]


# =========================================================
# Text cleaning
# =========================================================

def clean_text(text):

    # Convert HTML entities:
    #
    # &amp; -> &
    # &lt;  -> <
    text = html.unescape(text)

    # Remove HTML tags such as:
    #
    # <em>D</em>
    # <strong>...</strong>
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # Convert Markdown links:
    #
    # [credential dumping](url)
    # ->
    # credential dumping
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# Parse our existing security corpus
# =========================================================
#
# Our processed corpus contains records such as:
#
# Term: authentication
# Definition: ...
#
# and:
#
# Name: ...
# Description: ...
#
#
# We use a small STATE MACHINE rather than splitting
# blindly on blank lines.
# =========================================================

def parse_records(file_path):

    text = file_path.read_text(
        encoding="utf-8"
    )

    records = []

    current_concept = None
    current_response_lines = []

    collecting_response = False


    # -----------------------------------------------------
    # Save the record currently being constructed.
    # -----------------------------------------------------

    def flush_record():

        nonlocal current_concept
        nonlocal current_response_lines
        nonlocal collecting_response

        if current_concept is None:
            return

        response = " ".join(
            current_response_lines
        )

        concept = clean_text(
            current_concept
        )

        response = clean_text(
            response
        )


        # ---------------------------------------------
        # Basic quality filtering
        # ---------------------------------------------

        if not concept:
            pass

        elif not response:
            pass

        # We observed examples such as:
        #
        # <em>D</em>
        #
        # in generation.
        #
        # For our SOC-oriented instruction dataset,
        # isolated single-letter glossary symbols
        # are not useful examples.
        elif (
            len(concept) == 1
            and concept.isalpha()
        ):
            pass

        # Very tiny responses are generally not
        # useful supervised examples.
        elif len(response) < 20:
            pass

        else:

            records.append(
                {
                    "concept": concept,
                    "response": response,
                }
            )


        current_concept = None
        current_response_lines = []
        collecting_response = False


    # -----------------------------------------------------
    # Read corpus line by line.
    # -----------------------------------------------------

    for raw_line in text.splitlines():

        line = raw_line.strip()


        # ---------------------------------------------
        # Detect:
        #
        # Term: ...
        # Name: ...
        # ---------------------------------------------

        concept_match = re.match(
            r"^(Term|Name):\s*(.+)$",
            line,
            flags=re.IGNORECASE
        )

        if concept_match:

            # New record begins.
            # Save the previous one first.
            flush_record()

            current_concept = (
                concept_match.group(2)
            )

            continue


        # ---------------------------------------------
        # Detect:
        #
        # Definition: ...
        # Description: ...
        # ---------------------------------------------

        response_match = re.match(
            r"^(Definition|Description):\s*(.*)$",
            line,
            flags=re.IGNORECASE
        )

        if (
            response_match
            and current_concept is not None
        ):

            collecting_response = True

            first_response_text = (
                response_match.group(2)
            )

            if first_response_text:

                current_response_lines.append(
                    first_response_text
                )

            continue


        # ---------------------------------------------
        # Continue multi-line definition/description.
        # ---------------------------------------------

        if (
            current_concept is not None
            and collecting_response
            and line
        ):

            current_response_lines.append(
                line
            )


    # Save final record.
    flush_record()

    return records


# =========================================================
# Deduplicate records
# =========================================================

def deduplicate_records(records):

    unique_records = []

    seen = set()


    for record in records:

        # Normalize for duplicate detection.
        key = (
            record["concept"].lower(),
            record["response"].lower(),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_records.append(
            record
        )


    return unique_records


# =========================================================
# Deterministically choose an instruction template
# =========================================================
#
# We don't use Python's built-in hash() because its value
# may vary between interpreter sessions.
#
# SHA-256 gives us stable/reproducible assignment.
# =========================================================

def build_instruction(concept):

    digest = hashlib.sha256(
        concept.encode("utf-8")
    ).digest()

    template_index = (
        digest[0]
        % len(INSTRUCTION_TEMPLATES)
    )

    template = (
        INSTRUCTION_TEMPLATES[
            template_index
        ]
    )

    return template.format(
        concept=concept
    )


# =========================================================
# Convert records into SFT examples
# =========================================================

def build_examples(records):

    examples = []


    for record in records:

        instruction = build_instruction(
            record["concept"]
        )

        examples.append(
            {
                "instruction": instruction,
                "response": record["response"],
            }
        )


    return examples


# =========================================================
# Remove train → validation leakage
# =========================================================

def remove_validation_overlap(
    train_records,
    validation_records
):

    train_keys = {
        (
            record["concept"].lower(),
            record["response"].lower(),
        )
        for record in train_records
    }


    clean_validation_records = []


    for record in validation_records:

        key = (
            record["concept"].lower(),
            record["response"].lower(),
        )

        if key not in train_keys:

            clean_validation_records.append(
                record
            )


    removed_count = (
        len(validation_records)
        - len(clean_validation_records)
    )


    return (
        clean_validation_records,
        removed_count
    )


# =========================================================
# Save JSONL
# =========================================================

def save_jsonl(
    examples,
    output_path
):

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for example in examples:

            json.dump(
                example,
                file,
                ensure_ascii=False
            )

            file.write("\n")


# =========================================================
# Main
# =========================================================

print(
    "========== SECURITY SFT DATASET PREPARATION =========="
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Parse source files
# ---------------------------------------------------------

train_records = parse_records(
    TRAIN_SOURCE_PATH
)

validation_records = parse_records(
    VALIDATION_SOURCE_PATH
)


print(
    f"\nParsed train records: "
    f"{len(train_records):,}"
)

print(
    f"Parsed validation records: "
    f"{len(validation_records):,}"
)


# ---------------------------------------------------------
# Deduplicate within each split
# ---------------------------------------------------------

train_records = deduplicate_records(
    train_records
)

validation_records = deduplicate_records(
    validation_records
)


print(
    f"\nUnique train records: "
    f"{len(train_records):,}"
)

print(
    f"Unique validation records: "
    f"{len(validation_records):,}"
)


# ---------------------------------------------------------
# Prevent exact examples appearing in both splits
# ---------------------------------------------------------

(
    validation_records,
    removed_overlap_count
) = remove_validation_overlap(
    train_records,
    validation_records
)


print(
    f"Validation overlaps removed: "
    f"{removed_overlap_count:,}"
)


# ---------------------------------------------------------
# Build instruction-response examples
# ---------------------------------------------------------

train_examples = build_examples(
    train_records
)

validation_examples = build_examples(
    validation_records
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

save_jsonl(
    train_examples,
    TRAIN_OUTPUT_PATH
)

save_jsonl(
    validation_examples,
    VALIDATION_OUTPUT_PATH
)


print(
    f"\nFinal train examples: "
    f"{len(train_examples):,}"
)

print(
    f"Final validation examples: "
    f"{len(validation_examples):,}"
)


print(
    f"\nSaved:"
)

print(
    TRAIN_OUTPUT_PATH
)

print(
    VALIDATION_OUTPUT_PATH
)


# =========================================================
# Inspect samples
# =========================================================

print(
    "\n========== SAMPLE TRAINING EXAMPLES =========="
)


for index, example in enumerate(
    train_examples[:5],
    start=1
):

    print(
        f"\n--- Example {index} ---"
    )

    print(
        f"Instruction: "
        f"{example['instruction']}"
    )

    print(
        f"Response: "
        f"{example['response'][:500]}"
    )