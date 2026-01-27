'''
python tokenizer_scripts/tokenizer.py \
    --model_file models/unigram/joint_unigram_model.model \
    --input_files \
        downstream_task/joeynmt/data/extracted_train.fi \
        downstream_task/joeynmt/data/et-tatoeba.et \
        downstream_task/joeynmt/data/sme_train.sme \
        downstream_task/joeynmt/data/sme_dev.sme \
        downstream_task/joeynmt/data/sme_test.sme \
    --output_files \
        downstream_task/joeynmt/data/processed/train.fi-tatoeba.unigram \
        downstream_task/joeynmt/data/processed/train.et-tatoeba.unigram \
        downstream_task/joeynmt/data/processed/train.sme.unigram \
        downstream_task/joeynmt/data/processed/dev.sme.unigram \
        downstream_task/joeynmt/data/processed/test.sme.unigram
'''

import sentencepiece as spm
import os
import argparse

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
            if line:  # skip empty lines
                # out_type=str returns a list of tokens ["Hel", "sinki"]
                pieces = sp.encode(line, out_type=str)
                f_out.write(" ".join(pieces) + "\n")
                sentence_count += 1
    
    print(f"Tokenized {sentence_count} sentences -> {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply existing tokenizer to files')
    parser.add_argument('--model_file', required=True, 
                       help='Path to the existing .model file')
    parser.add_argument('--input_files', nargs='+', required=True, 
                       help='Raw text files to tokenize')
    parser.add_argument('--output_files', nargs='+', required=True, 
                       help='Output paths for tokenized files')

    args = parser.parse_args()

    if len(args.input_files) != len(args.output_files):
        raise ValueError("Number of input files must match number of output files")

    # Check if model exists
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Model file not found: {args.model_file}")

    # Run inference only
    print(f"Loading model: {args.model_file}")
    for input_file, output_file in zip(args.input_files, args.output_files):
        encode_with_tokenizer(args.model_file, input_file, output_file)

    print(f"\nAll done! Output files: {', '.join(args.output_files)}")