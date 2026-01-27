import json
import os
import glob

# Word boundary of sentencepiece tokenizers: starting with  ▁(U+2581)
SPIECE_MARKER = '▁' #(U+2581) '\u2581'
# Word boundary of OBPE ends with <w>
OBPE_MARKER = '</w>'

def load_lines(filepath):
    """loading"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def group_subwords_by_markers(subword_list, strategy):
    """
    text: täiesti nõus .
    Example Input (OBPE): ['täiesti</w>', 'nõ', 'us</w>', '.</w>']
    Example Input (BPE): ['▁täiesti', '▁nõus', '▁.']
    Example Output: [['täiesti</w>'], ['nõ', 'us</w>'], ['.</w>']]
    """
    groups = []
    current_group = []

    if strategy == 'OBPE':
        # --- OBPE word boundary pattern: End-of-word marker </w> ---
        for sub in subword_list:
            current_group.append(sub)
            if sub.endswith(OBPE_MARKER):
                groups.append(current_group)
                current_group = []
        # if some subwords have been left
        if current_group:
            groups.append(current_group)

    else:
        # --- SentencePiece (BPE/Unigram): Start-of-word marker   ---
        for sub in subword_list:
            if sub.startswith(SPIECE_MARKER):
                if current_group:
                    groups.append(current_group)
                current_group = [sub]
            else:
                # including exceptions
                if not current_group: 
                    current_group = [sub]
                else:
                    current_group.append(sub)
        
        # last subword group
        if current_group:
            groups.append(current_group)

    return groups

def detect_strategy(first_line_subwords):
    if OBPE_MARKER in first_line_subwords:
        return 'OBPE'
    return 'SPIECE'

def process_files(subword_path, txt_path, tag_path, output_path):
    try:
            txt_lines = load_lines(txt_path)
            tag_lines = load_lines(tag_path)
            sub_lines = load_lines(subword_path)
    except Exception as e:
            print(f"    [ERROR] Cannot open files: {e}")
            return
    
    # 1. check line count
    if not (len(txt_lines) == len(tag_lines) == len(sub_lines)):
        print(f"Error: Line count mismatch! Skipping.")
        return

    # 2. check Tokenizer (only run through the first line)
    strategy = detect_strategy(sub_lines[0])
    print(f"  -> Detected Strategy: {strategy}")

    full_dataset = []
    success_count = 0
    error_count = 0

    # 3. process by line
    for line_idx, (txt_l, tag_l, sub_l) in enumerate(zip(txt_lines, tag_lines, sub_lines)):
        
        words = txt_l.split()
        tags = tag_l.split()
        subwords = sub_l.split()
        
        # group subwords by markers
        subword_groups = group_subwords_by_markers(subwords, strategy)
        
        # 4. Alignment Sanity Check
        # subword count should be equal to original word count
        if len(subword_groups) != len(words):
            if error_count < 5: 
                # print first 5 errors
                print(f"  [Warning] Line {line_idx} Alignment Failed:")
                print(f"    Original Words ({len(words)}): {words}")
                print(f"    Subword Groups ({len(subword_groups)}): {subword_groups}")
            error_count += 1
            continue

        # 5. First-Token Tagging
        for i, (word, gold_tag, group) in enumerate(zip(words, tags, subword_groups)):
            
            # label token with real tag and label the rest with <PAD>
            train_labels = [gold_tag] + ["<PAD>"] * (len(group) - 1)
            
            full_dataset.append({
                "id": len(full_dataset),
                "orig_word": word,
                "orig_tag": gold_tag,
                "subwords": group,          #  ▁talo 或 talo</w>
                "train_labels": train_labels
            })
        
        success_count += 1

    if full_dataset:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_dataset, f, ensure_ascii=False, indent=2)
            print(f"    [SUCCESS] JSON Generated: {os.path.basename(output_path)}")
            print(f"    [FAIL] No valid data found. Check warnings above.")
    
    print(f"--> Saved to {output_path}")
    print(f"    Success: {success_count} sentences")
    print(f"    Failed:  {error_count} sentences (mismatch)\n")

# def resolve_filename(subword_filename, file_type):
#     """
#     智能文件名解析器
#     subword_filename: "finnish_train.bpe"
#     file_type: ".txt" 或 ".tags"
#     返回: 预期的目标文件名 (如 "fi_train_v4.txt")
#     """
#     # 1. 去掉后缀，拿到 core (如 finnish_train)
#     # 尝试按第一个点分割 (fi.train.bpe -> fi) 或下划线?
#     # 你的文件似乎混用了点和下划线，我们用更通用的方法：
    
#     # 假设 subword 文件名格式是 [前缀].[切分].[算法] (fi.train.bpe)
#     # 或者 [前缀]_[切分].[算法] (fi_train.bpe)
    
#     base = subword_filename.rsplit('.', 1)[0] # 去掉 .bpe/.unigram
    
#     # 2. 处理前缀映射 (finnish -> fi)
#     for bad_prefix, good_prefix in NAME_MAPPING.items():
#         if base.startswith(bad_prefix):
#             base = base.replace(bad_prefix, good_prefix, 1)
#             break
            
#     # 3. 处理 et-et 这种特殊情况 (双重前缀)
#     base = base.replace("et-et", "et")
            
#     # 4. 这里的 base 现在应该是 "fi.train" 或 "fi_train"
#     # UD 数据集通常用下划线连接，把点号换成下划线
#     base = base.replace('.', '_')
    
#     # 5. 拼接 v4 后缀
#     # 最终变成: fi_train_v4.txt
#     return f"{base}_v4{file_type}"

    
def main():
    data_dir = "./pilot_data/ud_data"  # <--- Modify the data directory HERE!!!
    # subfolders
    subwords_dir = os.path.join(data_dir, "subwords")
    text_dir = os.path.join(data_dir, "text")
    tags_dir = os.path.join(data_dir, "tags")    
    # output subfolder 
    output_dir = os.path.join(data_dir, "aligned_json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Verify directories exist
    print(f"--- Checking Directories ---")
    for d in [subwords_dir, text_dir, tags_dir]:
        if os.path.exists(d):
            print(f"[OK] Found directory: {d}")
        else:
            print(f"[FAIL] Directory NOT found: {d}")
            print("       Please check your folder naming/structure.")
            return

    # iterate subwords folder 
        # List all files in subwords
    files = os.listdir(subwords_dir)
    print(f"\n--- Scanning 'subwords' folder ({len(files)} files found) ---")
    if not os.path.exists(subwords_dir):
        print(f"Error: Subwords directory not found at {subwords_dir}")
        return

    for filename in files:
        # example: "fi_train.bpe"
        parts = filename.rsplit('.', 1) 
        
        # 2 parts ["et_dev", "bpe"]
        if len(parts) != 2:
            print(f"[SKIP] Unknown format: {filename}")
            continue
            
        prefix = parts[0]      # "et_dev"
        tokenizer_type = parts[1] # "bpe"
        
        # file name of text and tag files
        txt_filename = f"{prefix}_v4.txt"   # "et_dev_v4.txt"
        tag_filename = f"{prefix}_v4.tags"  # "et_dev_v4.tags"
        
        # output file name 
        output_filename = f"{prefix}_{tokenizer_type}_aligned.json"

            
        # construct full paths
        subword_path = os.path.join(subwords_dir, filename)
        txt_path = os.path.join(text_dir, txt_filename)
        tag_path = os.path.join(tags_dir, tag_filename)
        output_path = os.path.join(output_dir, output_filename)
            
        # Check if reference files exist before processing
        if os.path.exists(txt_path) and os.path.exists(tag_path):
            process_files(subword_path, txt_path, tag_path, output_path)
        else:
            print(f"Skipping {filename}:")
            if not os.path.exists(txt_path): print(f"  Missing Text: {txt_path}")
            if not os.path.exists(tag_path): print(f"  Missing Tags: {tag_path}")
            continue
    print(f"\n--- DONE ---")

if __name__ == "__main__":
    main()