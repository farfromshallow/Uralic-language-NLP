#!/usr/bin/env python3
"""
Input  : whitespace-tokenized sentences (one sentence per line)
Output : OBPE-subworded text with </w> marking word boundaries

python tokenizer_scripts/tokenizer_obpe.py \
    --codes /Users/Ingrid/OBPE/models/fi+et_sme_obpe/merges.txt \
    --input ./downstream_task/joeynmt/data/sme_train.sme ./downstream_task/joeynmt/data/sme_dev.sme ./downstream_task/joeynmt/data/sme_test.sme \
    --output ./downstream_task/joeynmt/data/processed/train.sme.obpe ./downstream_task/joeynmt/data/processed/dev.sme.obpe ./downstream_task/joeynmt/data/processed/test.sme.obpe
"""

import argparse
from collections import defaultdict


def load_codes(codes_path):
    """
    Load OBPE merge rules.
    Each line: "a b" (two aymbols)
    """
    merges = []
    with open(codes_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            merges.append((a, b))
    return merges


def get_pairs(symbols):
    """Return set of adjacent symbol pairs."""
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


def apply_bpe_to_word(word, merges):
    """
    Apply OBPE merges to a single word.
    OBPE operates on characters.
    """
    symbols = list(word)

    if len(symbols) == 1:
        return symbols

    merge_dict = {pair: i for i, pair in enumerate(merges)}

    while True:
        pairs = get_pairs(symbols)
        candidates = [(merge_dict[p], p) for p in pairs if p in merge_dict]
        if not candidates:
            break

        _, best_pair = min(candidates)
        new_symbols = []
        i = 0

        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1

        symbols = new_symbols

    return symbols


def apply_obpe_sentence(sentence, merges):
    """
    Apply OBPE to one sentence.
    Each word is segmented independently.
    """
    output_tokens = []
    words = sentence.strip().split()

    for word in words:
        subwords = apply_bpe_to_word(word, merges)
        for sw in subwords:
            output_tokens.append(sw)
        # OBPE word boundary marker
        output_tokens[-1] = output_tokens[-1] + "</w>"

    return " ".join(output_tokens)


def main():
    parser = argparse.ArgumentParser(description="Apply trained OBPE tokenizer")
    parser.add_argument("--codes", required=True, help="OBPE merge rules")
    parser.add_argument("--input", nargs='+', required=True, help="Input text files")
    parser.add_argument("--output", nargs='+', required=True, help="Output OBPE files")
    args = parser.parse_args()

    merges = load_codes(args.codes)

    for input_file, output_file in zip(args.input, args.output):
        with open(input_file, encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    fout.write("\n")
                    continue
                obpe_line = apply_obpe_sentence(line, merges)
                fout.write(obpe_line + "\n")

    print(f"OBPE-applied text written to {args.output}")


if __name__ == "__main__":
    main()