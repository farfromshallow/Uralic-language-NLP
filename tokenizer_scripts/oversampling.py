'''
python tokenizer_scripts/oversampling.py \
    --input downstream_task/joeynmt/data/processed/train.sme.obpe \
    --output downstream_task/joeynmt/data/processed/train.sme_oversampled.obpe
'''

import argparse
import os

def oversample_file(input_path, output_path, multiplier):
    """
    Oversamples a given file by duplicating its content 'multiplier' times.
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for _ in range(multiplier):
            outfile.writelines(lines)
    
    print(f"Oversampled '{input_path}' by {multiplier} times to '{output_path}'.")
    print(f"Original lines: {len(lines)}, New lines: {len(lines) * multiplier}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oversample a text file by duplicating its content.")
    parser.add_argument("--input", nargs='+', required=True, help="Path to the input files.")
    parser.add_argument("--output", nargs='+', required=True, help="Path to the output oversampled files.")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Number of times to duplicate the file content (default: 5).")
    
    args = parser.parse_args()

    for in_path, out_path in zip(args.input, args.output):
        if not os.path.exists(in_path):
            print(f"Input file not found: {in_path}")
            continue
        oversample_file(in_path, out_path, args.multiplier)