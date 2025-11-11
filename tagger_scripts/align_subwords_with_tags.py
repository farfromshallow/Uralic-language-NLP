'''
python tagger_scripts/align_subwords_with_tags.py \
  --subwords path/to/subwords \
  --tags path/to/tags \
  --output path/to/output \
  --tokenizer_type bpe/unigram/obpe

'''
import argparse
import sys
import unicodedata
import string
import re
from typing import Optional


def is_unicode_punct(token: str) -> bool:
    """Return True if token consists only of true punctuation characters (category 'P'), excluding symbols and ASCII hyphen-minus."""
    if not token:
        return False
    for ch in token:
        cat = unicodedata.category(ch)
        if not cat.startswith('P'):
            return False
        if ch == '-':  # Exclude ASCII hyphen-minus
            return False
    return True


def extract_punct_surface(token: str, tokenizer_type: str) -> Optional[str]:
    """Return the punctuation surface form for validation based on tokenizer markers."""
    if not token:
        return None

    # Skip tokens that are symbols (category 'S') or contain any symbol characters
    if any(unicodedata.category(ch).startswith('S') for ch in token):
        return None

    if tokenizer_type in {"bpe", "unigram"}:
        if token.startswith('▁'):
            raw = token[1:]
            if raw and is_unicode_punct(raw):
                return raw
        elif is_unicode_punct(token):
            return token
    elif tokenizer_type == "obpe":
        if token.endswith("</w>"):
            raw = token[:-4]
            if raw and is_unicode_punct(raw):
                return raw
        elif is_unicode_punct(token):
            return token

    return None


def align_subwords_with_tags(subword_file, tag_file, output_file, tokenizer_type):
    with open(subword_file, 'r', encoding='utf-8') as f_sub:
        subword_lines = [line.strip() for line in f_sub if line.strip()]
    with open(tag_file, 'r', encoding='utf-8') as f_tag:
        tag_lines = [line.strip() for line in f_tag if line.strip()]

    if len(subword_lines) != len(tag_lines):
        print(f"Warning: Sentence count mismatch: {len(subword_lines)} subword vs {len(tag_lines)} tags", file=sys.stderr)

    validation_errors = []

    with open(output_file, 'w', encoding='utf-8') as fout:
        for idx, (sub_sent, tag_sent) in enumerate(zip(subword_lines, tag_lines)):
            sub_tokens = sub_sent.split()
            word_tags = tag_sent.split()
            aligned = []
            word_groups = []
            current_group = []

            # Normalize tokens by stripping spaces and standardizing separators
            def normalize_token(tok):
                return tok.strip()

            sub_tokens = [normalize_token(tok) for tok in sub_tokens]

            # Systematically group subwords based on separator pattern.
            if tokenizer_type in ["bpe", "unigram"]:
                # Group subwords into words. Rules (BPE/Unigram):
                # - A token that starts with the SentencePiece word-start marker '▁'(U+2581) begins a new word group.
                # - Punctuation handling:
                #   * If punctuation occurs at the end of a sentence, it is its own group.
                #   * If punctuation is followed by a token starting with '▁', it is its own group.
                #   * Otherwise, punctuation is part of the current word group.
                word_groups = []
                current_group = []
                for i, tok in enumerate(sub_tokens):
                    if tok.startswith('▁'):
                        # start a new group
                        if current_group:
                            word_groups.append(current_group)
                        current_group = [tok]
                    elif is_unicode_punct(tok):
                        # Check if token is at sentence end or followed by a token starting with '▁'
                        is_last_token = (i == len(sub_tokens) - 1)
                        next_starts_with_underscore = (not is_last_token and sub_tokens[i+1].startswith('▁'))
                        if is_last_token or next_starts_with_underscore:
                            # punctuation as own group
                            if current_group:
                                word_groups.append(current_group)
                                current_group = []
                            word_groups.append([tok])
                        else:
                            # punctuation as part of current group
                            if not current_group:
                                current_group = [tok]
                            else:
                                current_group.append(tok)
                    else:
                        # continuation of current word
                        if not current_group:
                            # no current group (edge case), start one
                            current_group = [tok]
                        else:
                            current_group.append(tok)

                if current_group:
                    word_groups.append(current_group)
            elif tokenizer_type == "obpe":
                # For OBPE: tokens ending with '</w>' close a word. We perform streaming alignment:
                # accumulate subwords until a closing marker appears, then immediately assign the
                # next gold tag to that group and append aligned subword-tag pairs.
                aligned = []
                tag_idx = 0
                current_group = []

                for tok in sub_tokens:
                    current_group.append(tok)
                    if tok.endswith("</w>"):
                        # assign tag for this completed word group
                        if tag_idx < len(word_tags):
                            tag = word_tags[tag_idx]
                        else:
                            tag = "O"
                            print(f"Warning: tags exhausted at sentence {idx+1} while processing OBPE groups", file=sys.stderr)
                        for sub in current_group:
                            aligned.append((sub, tag))
                        tag_idx += 1
                        current_group = []

                # If any trailing subwords remain (no closing </w>) treat them as a final group
                if current_group:
                    if tag_idx < len(word_tags):
                        tag = word_tags[tag_idx]
                    else:
                        tag = "O"
                        print(f"Warning: tags exhausted at sentence {idx+1} for trailing OBPE group", file=sys.stderr)
                    for sub in current_group:
                        aligned.append((sub, tag))
                    tag_idx += 1

                # If there are leftover tags (more tags than groups), log a warning
                if tag_idx < len(word_tags):
                    print(f"Warning: more tags ({len(word_tags)}) than OBPE groups ({tag_idx}) in sentence {idx+1}", file=sys.stderr)

                # Skip the later strict index-based mapping for OBPE since we already built `aligned`.
                # In order to keep downstream code consistent, set word_groups to an empty list so
                # the later block that maps groups by index is bypassed. (The code below will use `aligned`.)
                word_groups = []
            else:
                print(f"Error: Unknown tokenizer type {tokenizer_type}", file=sys.stderr)
                sys.exit(1)

            # Strict alignment: one tag per word group, no reassignment or reuse of tags.
            if not aligned:
                if len(word_groups) != len(word_tags):
                    print(f"\n[DEBUG Sentence {idx+1}]", file=sys.stderr)
                    print("Subwords:", sub_tokens, file=sys.stderr)
                    print("Word groups:", file=sys.stderr)
                    for g in word_groups:
                        print("  ", g, file=sys.stderr)
                    print("Tags:", word_tags, file=sys.stderr)
                aligned = []
                # Assign tags strictly by index; if more groups than tags, assign 'O' to extra groups;
                # if more tags than groups, ignore extra tags.
                for i, group in enumerate(word_groups):
                    if i < len(word_tags):
                        tag = word_tags[i]
                    else:
                        tag = "O"  # Default tag for extra groups
                    for sub in group:
                        aligned.append((sub, tag))

            # Validate punctuation alignment
            sentence_failures = []
            for sub, tag in aligned:
                punct_surface = extract_punct_surface(sub, tokenizer_type)
                if punct_surface and tag != "PUNCT":
                    sentence_failures.append(f"token '{punct_surface}' tagged as '{tag}'")

            if sentence_failures:
                validation_errors.append((idx + 1, sentence_failures))

            # Write sentence in CoNLL format
            for sub, tag in aligned:
                fout.write(f"{sub}\t{tag}\n")
            fout.write("\n")

    if validation_errors:
        print("⚠️  Alignment validation detected mismatches:", file=sys.stderr)
        for sent_idx, failures in validation_errors:
            for failure in failures:
                print(f"  sentence {sent_idx}: {failure}", file=sys.stderr)

    print(f"✅ Aligned subwords written to {output_file}")

    return validation_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align subword tokens with POS tags")
    parser.add_argument("--subwords", required=True, help="Path to subword file")
    parser.add_argument("--tags", required=True, help="Path to tag file")
    parser.add_argument("--output", required=True, help="Output CoNLL file")
    parser.add_argument("--tokenizer_type", required=True, choices=["bpe", "unigram", "obpe"],
                        help="Specify tokenizer type (bpe/unigram/obpe)")
    args = parser.parse_args()

    align_subwords_with_tags(args.subwords, args.tags, args.output, args.tokenizer_type)