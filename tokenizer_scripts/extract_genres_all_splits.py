
"""
batch extract all splits of specific genres (train, dev, test) of Finnish-TDT and Russian-Taiga
"""
import os
import sys
import re

def sentence_matches(sent_lines, dataset, genres):
    meta = {}
    for ln in sent_lines:
        if not ln.startswith('#'):
            continue
        m = re.match(r'#\s*([^=]+?)\s*=\s*(.+)', ln)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip().lower()
            meta[key] = val

    sid = meta.get('sent_id') or meta.get('send_id') or meta.get('sentid') or ''
    genre_meta = meta.get('genre', '')

    if dataset == 'finnish':
        sid = sid.strip()
        for g in genres:
            if sid.startswith(g.lower()):
                return True
        for v in meta.values():
            for g in genres:
                if v.startswith(g.lower()):
                    return True
        return False
    else:  # russian
        for g in genres:
            if genre_meta == g.lower():
                return True
        for v in meta.values():
            for g in genres:
                if g.lower() in v.split():
                    return True
        return False

def extract_genre_file(in_path, out_path, dataset, genres):
    matched = 0
    total = 0
    try:
        with open(in_path, 'r', encoding='utf-8') as inf, open(out_path, 'w', encoding='utf-8') as outf:
            buf = []
            for line in inf:
                line = line.rstrip('\n')
                if line == '':
                    if buf:
                        total += 1
                        if sentence_matches(buf, dataset, genres):
                            outf.write('\n'.join(buf) + '\n\n')
                            matched += 1
                        buf = []
                    continue
                buf.append(line)
            if buf:
                total += 1
                if sentence_matches(buf, dataset, genres):
                    outf.write('\n'.join(buf) + '\n\n')
                    matched += 1
    except FileNotFoundError:
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return False
    except Exception as e:
        print("Error during extraction:", e, file=sys.stderr)
        return False

    print(f"Processed {total} sentences, wrote {matched} matching sentences to {out_path}")
    return True

def run_extraction(dataset, input_file, genres, output_conllu):
    print(f"\nExtract: {os.path.basename(input_file)}")
    print(f"Dataset: {dataset}  Genres: {genres}")
    print(f"Output: {output_conllu}")
    print("-" * 60)
    return extract_genre_file(input_file, output_conllu, dataset, genres)

def main():
    base_dir = "/Users/Ingrid/Uralic-language-NLP/pilot_data"
    output_dir = os.path.join(base_dir, "ud_data", "extracted_genres")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Batch extract all splits of specific genres (train, dev, test)")
    print("=" * 60)
    
    # Finnish splits: b, w, wn, u, t
    finnish_base = os.path.join(base_dir, "ud_data", "UD_Finnish-TDT")
    finnish_genres = ['b', 'w', 'wn', 'u', 't']
    
    print("\n" + "=" * 60)
    print("Processing Finnish data...")
    print("=" * 60)
    
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(finnish_base, f"fi_tdt-ud-{split}.conllu")
        if os.path.exists(input_file):
            print(f"\nProcessing {split} split...")
            output_conllu = os.path.join(output_dir, f"fi_extracted_genres-{split}.conllu")

            success = run_extraction(
                'finnish',
                input_file,
                finnish_genres,
                output_conllu
            )
            print(f"{split} split {'extracted' if success else 'extraction failed'}")
        else:
            print(f"File does not exist: {os.path.basename(input_file)}")
    
    # List Finnish generated files
    prefix = 'fi_extracted_genres'
    for split in ['train', 'dev', 'test']:
        conllu_file = os.path.join(output_dir, f"{prefix}-{split}.conllu")
        if os.path.exists(conllu_file):
            size = os.path.getsize(conllu_file) / 1024
            print(f"  - {prefix}-{split}.conllu: {size:.1f} KB")

    # Russian splits by genre: news, social, wiki
    russian_base = os.path.join(base_dir, "ud_data", "UD_Russian-Taiga")
    russian_genres = ['news', 'social', 'wiki']

    print("\n" + "=" * 60)
    print("Processing Russian data...")
    print("=" * 60)

    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(russian_base, f"ru_taiga-ud-{split}.conllu")
        if os.path.exists(input_file):
            print(f"\nProcessing {split} split...")
            output_conllu = os.path.join(output_dir, f"ru_extracted_genres-{split}.conllu")

            success = run_extraction(
                'russian',
                input_file,
                russian_genres,
                output_conllu
            )
            print(f"{split} split {'extracted' if success else 'extraction failed'}")
        else:
            print(f"File does not exist: {os.path.basename(input_file)}")
    
    # List Russian generated files
    prefix = 'ru_extracted_genres'
    for split in ['train', 'dev', 'test']:
        conllu_file = os.path.join(output_dir, f"{prefix}-{split}.conllu")
        if os.path.exists(conllu_file):
            size = os.path.getsize(conllu_file) / 1024
            print(f"  - {prefix}-{split}.conllu: {size:.1f} KB")
    
    print("\n" + "=" * 60)
    print("Batch extraction completed!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")

if __name__ == '__main__':
    main()