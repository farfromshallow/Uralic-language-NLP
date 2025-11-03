#!/usr/bin/env python3
"""
batch extract all splits of specific genres (train, dev, test) of Finnish-TDT
"""
import subprocess
import os
import sys

def run_extraction(dataset, input_file, genres, output_txt, output_conllu):
    cmd = [
        sys.executable,
        'tokenizer_scripts/extract_genre.py',
        '--dataset', dataset,
        '--input', input_file,
        '--genres'] + genres + [
        '--output_txt', output_txt,
        '--output_conllu', output_conllu
    ]
    print(f"\nExtract: {os.path.basename(input_file)}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0

def main():
    base_dir = "/Users/Ingrid/Uralic-language-NLP"
    output_dir = os.path.join(base_dir, "ud_data", "extracted_genres")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Batch extract all splits of specific genres (train, dev, test)")
    print("=" * 60)
    print("\nFinnish genres: b (blog), w (wikipedia)\n")
    
    # Estonian splits: kom, reddit
    # estonian_base = os.path.join(base_dir, "ud_data", "UD_Estonian-EWT")
    # estonian_genres = ['kom', 'reddit']
    
    # print("\n" + "=" * 60)
    # print("Processing Estonian data...")
    # print("=" * 60)
    
    # for split in ['train', 'dev', 'test']:
    #     input_file = os.path.join(estonian_base, f"et_ewt-ud-{split}.conllu")
    #     if os.path.exists(input_file):
    #         print(f"\nProcessing {split} split...")
    #         output_txt = os.path.join(output_dir, f"et_kom_reddit-{split}.txt")
    #         output_conllu = os.path.join(output_dir, f"et_kom_reddit-{split}.conllu")
            
    #         success = run_extraction(
    #             'estonian',
    #             input_file,
    #             estonian_genres,
    #             output_txt,
    #             output_conllu
    #         )
    #         if success:
    #             print(f"{split} split extracted")
    #         else:
    #             print(f"{split} split extraction failed")
    #     else:
    #         print(f"File does not exist: {os.path.basename(input_file)}")
    
    # Finnish splits: b, w
    finnish_base = os.path.join(base_dir, "ud_data", "UD_Finnish-TDT")
    finnish_genres = ['b', 'w']
    
    print("\n" + "=" * 60)
    print("Processing Finnish data...")
    print("=" * 60)
    
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(finnish_base, f"fi_tdt-ud-{split}.conllu")
        if os.path.exists(input_file):
            print(f"\nProcessing {split} split...")
            output_txt = os.path.join(output_dir, f"fi_blog_wiki-{split}.txt")
            output_conllu = os.path.join(output_dir, f"fi_blog_wiki-{split}.conllu")
            
            success = run_extraction(
                'finnish',
                input_file,
                finnish_genres,
                output_txt,
                output_conllu
            )
            if success:
                print(f"{split} split extracted")
            else:
                print(f"{split} split extraction failed")
        else:
            print(f"File does not exist: {os.path.basename(input_file)}")
    
    print("\n" + "=" * 60)
    print("Batch extraction completed!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    
    # List all generated files (Finnish only)
    prefix = 'fi_blog_wiki'
    for split in ['train', 'dev', 'test']:
        txt_file = os.path.join(output_dir, f"{prefix}-{split}.txt")
        conllu_file = os.path.join(output_dir, f"{prefix}-{split}.conllu")
        if os.path.exists(txt_file):
            size = os.path.getsize(txt_file) / 1024
            lines = sum(1 for _ in open(txt_file, 'r', encoding='utf-8'))
            print(f"  - {prefix}-{split}.txt: {lines} lines, {size:.1f} KB")
        if os.path.exists(conllu_file):
            size = os.path.getsize(conllu_file) / 1024
            print(f"  - {prefix}-{split}.conllu: {size:.1f} KB")

if __name__ == '__main__':
    main()

