import sentencepiece as spm # import sentencepiece library
import os
import argparse

# Function to train and encode using Unigram model
def train_and_encode_unigram(input_file, output_file, model_prefix, vocab_size=5000):
    # Train Unigram model with safe configuration
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type=unigram --character_coverage=1.0 "
        f"--unk_piece=<unk> --hard_vocab_limit=false"
    )

    # Load model
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Encode sentences
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            pieces = sp.encode(line.strip(), out_type=str)
            f_out.write(" ".join(pieces) + "\n")

    print(f"Unigram tokenized file saved at {output_file}")

# Function to train and encode using BPE model
def train_and_encode_bpe(input_file, output_file, model_prefix, vocab_size=5000):
    # Train BPE model with safe configuration
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 "
        f"--unk_piece=<unk> --hard_vocab_limit=false"
    )

    # Load model
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Encode sentences
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            pieces = sp.encode(line.strip(), out_type=str)
            f_out.write(" ".join(pieces) + "\n")

    print(f"BPE tokenized file saved at {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and apply tokenizer (BPE or Unigram).')
    parser.add_argument('--tokenizer', choices=['bpe', 'unigram'], required=True, help='Tokenizer type to use')
    parser.add_argument('--input', nargs='+', required=True, help='Input text file(s) to tokenize')
    parser.add_argument('--output', nargs='+', required=True, help='Corresponding output file(s)')
    parser.add_argument('--model_prefix', required=True, help='Prefix for the trained model')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size for the tokenizer')

    args = parser.parse_args()

    if len(args.input) != len(args.output):
        raise ValueError("Number of input files and output files must match.")

    for in_file, out_file in zip(args.input, args.output):
        if args.tokenizer == 'unigram':
            train_and_encode_unigram(in_file, out_file, args.model_prefix, args.vocab_size)
        elif args.tokenizer == 'bpe':
            train_and_encode_bpe(in_file, out_file, args.model_prefix, args.vocab_size)