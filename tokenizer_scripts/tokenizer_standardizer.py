"""
python tokenizer_scripts/tokenizer_standardizer.py standardize-obpe \
  --subwords path/to/subwords \
  --output_prefix output_prefix \
  --output_dir ./path/to/output_directory
"""

import os
import re
import argparse

# Compile regex patterns once
WHITESPACE_PATTERN = re.compile(r"\s+")
SEPARATOR_WHITESPACE_PATTERN = re.compile(r"▁\s+")

def clean_bpe(line: str) -> str:
    """
    Standardize BPE tokenization.
    Keeps underscores (▁) as word boundaries and ensures single spacing.
    Removes whitespace right after separator ▁.
    """
    line = line.strip()
    line = WHITESPACE_PATTERN.sub(" ", line)
    line = SEPARATOR_WHITESPACE_PATTERN.sub("▁", line)
    return line

def clean_unigram(line: str) -> str:
    """
    Standardize Unigram tokenization.
    Removes redundant whitespace and preserves SentencePiece underscores.
    Removes whitespace right after separator ▁.
    """
    line = line.strip()
    line = WHITESPACE_PATTERN.sub(" ", line)
    line = SEPARATOR_WHITESPACE_PATTERN.sub("▁", line)
    return line

def clean_obpe(line: str) -> str:
    """
    Standardize OBPE tokenization.
    Normalize whitespace.
    Keep </w> markers alignment.
    """
    line = line.strip()
    line = WHITESPACE_PATTERN.sub(" ", line)
    return line

def standardize_file(input_path: str, output_path: str, tokenizer_type: str):
    print(f"🔧 Standardizing {tokenizer_type.upper()} file: {input_path}")

    # Select cleaning function
    if tokenizer_type == "bpe":
        cleaner = clean_bpe
    elif tokenizer_type == "unigram":
        cleaner = clean_unigram
    elif tokenizer_type == "obpe":
        cleaner = clean_obpe
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = [cleaner(line) for line in fin if line.strip()]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n")

    print(f"Standardized file saved at: {output_path}")
    print(f"   → Total sentences: {len(lines)}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Standardize subword tokenization outputs (BPE, Unigram, OBPE)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # BPE / Unigram / OBPE subcommands
    for tok in ["bpe", "unigram", "obpe"]:
        sub = subparsers.add_parser(f"standardize-{tok}", help=f"Standardize {tok.upper()} subwords")
        sub.add_argument("--subwords", required=True, help="Path to tokenized subwords file")
        sub.add_argument("--output_prefix", required=True, help="Output filename prefix")
        sub.add_argument("--output_dir", required=True, help="Output directory for standardized files")

    args = parser.parse_args()

    tokenizer_type = args.command.replace("standardize-", "")
    input_file = args.subwords
    output_path = os.path.join(args.output_dir, f"{args.output_prefix}.standardized.txt")

    standardize_file(input_file, output_path, tokenizer_type)

if __name__ == "__main__":
    main()