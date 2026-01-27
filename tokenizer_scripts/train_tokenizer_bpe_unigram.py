'''
python tokenizer_scripts/train_tokenizer_bpe_unigram.py \
    --tokenizer unigram \
    --train_file ./pilot_data/ud_data/text/et_train_v5.txt \
    --eval_files ./pilot_data/ud_data/text/et_train_v5.txt ./pilot_data/ud_data/text/et_dev_v5.txt ./pilot_data/ud_data/text/et_test_v5.txt \
    --output_files ./pilot_data/ud_data/subword/et_train.bpe ./pilot_data/ud_data/subword/et_dev.bpe ./pilot_data/ud_data/subword/et_test.bpe \
    --model_prefix et_unigram_model \
    --vocab_size 5000
'''

import sentencepiece as spm
import os
import argparse

# def train_tokenizer(input_file, model_prefix, tokenizer_type, vocab_size=5000):
#     """train the tokenizer and store the model"""
#     if tokenizer_type == 'unigram':
#         model_type = 'unigram'
#     elif tokenizer_type == 'bpe':
#         model_type = 'bpe'
#     else:
#         raise ValueError(f"Invalid tokenizer: {tokenizer_type}")
    
#     # training step
#     spm.SentencePieceTrainer.Train(
#         f"--input={input_file} --model_prefix={model_prefix} "
#         f"--vocab_size={vocab_size} --model_type={model_type} --character_coverage=1.0 "
#         f"--unk_piece=<unk> --hard_vocab_limit=false"
#     )
    
#     print(f"{tokenizer_type.upper()} training completed: {model_prefix}.model")
#     return f"{model_prefix}.model"

def train_tokenizer(input_file, model_prefix, tokenizer_type, vocab_size=5000):
    """train the tokenizer and store outputs in tokenizer-specific subdirectories"""
    
    # Define the directory structure based on tokenizer type
    # Paths: ./models/[tokenizer]/ and ./vocab/[tokenizer]/
    base_model_dir = os.path.join(".", "models", tokenizer_type)
    base_vocab_dir = os.path.join(".", "vocab", tokenizer_type)
    
    # Ensure directories exist
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_vocab_dir, exist_ok=True)

    # SentencePiece uses the prefix to name the .model and .vocab files
    # We initially save both to the model directory
    model_path_prefix = os.path.join(base_model_dir, model_prefix)
    
    if tokenizer_type not in ['unigram', 'bpe']:
        raise ValueError(f"Invalid tokenizer: {tokenizer_type}")
# 1. Training step
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_path_prefix} "
        f"--vocab_size={vocab_size} --model_type={tokenizer_type} --character_coverage=1.0 "
        f"--unk_piece=<unk> --hard_vocab_limit=false"
    )
    
    model_file = f"{model_path_prefix}.model"
    temp_vocab_file = f"{model_path_prefix}.vocab"
    final_vocab_path = os.path.join(base_vocab_dir, f"{model_prefix}.vocab")

    # 2. Move the vocabulary file to the designated ./vocab/[tokenizer]/ folder
    if os.path.exists(temp_vocab_file):
        os.rename(temp_vocab_file, final_vocab_path)
    
    print(f"--- {tokenizer_type.upper()} Training Results ---")
    print(f"Model saved to: {model_file}")
    print(f"Vocab saved to: {final_vocab_path}")
    
    return model_file

def encode_with_tokenizer(model_file, input_file, output_file):
    """tokenization step"""
# Ensure the output directory for tokenized text exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
            
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    # encode sentences
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        sentence_count = 0
        for line in f_in:
            line = line.strip()
            if line:  # skip line
                pieces = sp.encode(line, out_type=str)
                f_out.write(" ".join(pieces) + "\n")
                sentence_count += 1
    
    print(f"Tokenized {sentence_count} sentences, and saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train tokenizer and apply to dev/train sets')
    parser.add_argument('--tokenizer', choices=['bpe', 'unigram'], required=True, 
                       help='Tokenizer type')
    parser.add_argument('--train_file', required=True, 
                       help='training file')
    parser.add_argument('--eval_files', nargs='+', required=True, 
                       help='files to tokenize')
    parser.add_argument('--output_files', nargs='+', required=True, 
                       help='tokenized files')
    parser.add_argument('--model_prefix', required=True, 
                       help='model prefix')
    parser.add_argument('--vocab_size', type=int, default=5000, 
                       help='vocal size (keep consistency among tokenizers)')

    args = parser.parse_args()

    if len(args.eval_files) != len(args.output_files):
        raise ValueError("Amount of sentences should be aligned")

    # 1. applying training data
    model_file = train_tokenizer(
        args.train_file, 
        args.model_prefix, 
        args.tokenizer, 
        args.vocab_size
    )
    
    # 2. tokenize
    for eval_file, output_file in zip(args.eval_files, args.output_files):
        encode_with_tokenizer(model_file, eval_file, output_file)
    
    print(f"\nAll done!")
    print(f"Model saved as: {model_file}")
    print(f"Output file: {', '.join(args.output_files)}")