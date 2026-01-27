import json
import os
import unicodedata

# --- 配置 ---
SPIECE_MARKER = '\u2581'  #('▁'): U+2581
OBPE_MARKER = '</w>'

# 映射表：如果 subword 文件名用了全称（如 finnish），这里转为简写（fi）
# 如果你的 subword 文件名已经是 et_dev.bpe 这种简写，这个映射表可能用不到，但留着以防万一
NAME_MAPPING = {
    "finnish": "fi",
    "estonian": "et",
    "hungarian": "hu",
    "et-et": "et",
}


def load_lines(filepath):
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def normalize_text(text):
    """
    [关键] 统一字符编码标准 (NFKC)
    解决 ä (U+00E4) 和 ä (a + U+0308) 不相等的问题
    """
    if text is None: return ""
    return unicodedata.normalize('NFKC', text)

def clean_subword(sub, strategy):
    """移除标记符，并进行规范化"""
    if strategy == 'OBPE':
        core = sub.replace(OBPE_MARKER, "")
    else:
        core = sub.replace(SPIECE_MARKER, "")
    
    # 简单的 UNK 处理 (如果有 <unk> 这种标记，视为空字符)
    if core.lower() in ['<unk>', '[unk]']:
        return None 
    
    return normalize_text(core)

def debug_char_codes(text):
    return [ord(c) for c in text]

def greedy_align_subwords(words, subwords, strategy):
    groups = []
    sub_idx = 0
    total_subs = len(subwords)
    
    for word_idx, raw_target_word in enumerate(words):
        current_group = []
        reconstructed_text = ""
        
        target_word_norm = normalize_text(raw_target_word)
        
        while sub_idx < total_subs:
            sub = subwords[sub_idx]
            cleaned_char = clean_subword(sub, strategy)
            
            # 尝试拼接
            if cleaned_char is None:
                # 如果是 UNK，假设它匹配了剩余部分 (简化逻辑)
                reconstructed_text = target_word_norm
            else:
                reconstructed_text += cleaned_char
            
            current_group.append(sub)
            sub_idx += 1
            
            # 2. 核心比对
            if reconstructed_text == target_word_norm:
                break
            
            # 3. 溢出检查
            if len(reconstructed_text) > len(target_word_norm):
                break 
        
        # 检查本轮对齐结果
        if reconstructed_text != target_word_norm:
            return None, {
                "orig": raw_target_word,
                "orig_norm": target_word_norm,
                "built": reconstructed_text,
                "orig_codes": debug_char_codes(target_word_norm),
                "built_codes": debug_char_codes(reconstructed_text)
            }
        
        groups.append(current_group)

    if sub_idx < total_subs:
        return None, {"error": "Leftover subwords", "leftover": subwords[sub_idx:]}

    return groups, None

def process_single_pair(sub_path, txt_path, tag_path, out_path):
    try:
        sub_lines = load_lines(sub_path)
        txt_lines = load_lines(txt_path)
        tag_lines = load_lines(tag_path)
    except Exception as e:
        print(f"    [ERROR] Read Failed: {e}")
        return

    # 基础行数检查
    if not (len(sub_lines) == len(txt_lines) == len(tag_lines)):
        print(f"    [FAIL] Line Mismatch! Sub:{len(sub_lines)} Txt:{len(txt_lines)} Tag:{len(tag_lines)}")
        return

    # 策略检测
    strategy = 'OBPE' if OBPE_MARKER in sub_lines[0] else 'SPIECE'
    dataset = []
    skipped_count = 0
    
    for i, (sub_l, txt_l, tag_l) in enumerate(zip(sub_lines, txt_lines, tag_lines)):
        words = txt_l.split()
        tags = tag_l.split()
        subs = sub_l.split()
        
        # 基本长度过滤
        if len(words) != len(tags):
            skipped_count += 1
            continue
        
        # Greedy Alignment
        groups, error_info = greedy_align_subwords(words, subs, strategy)
        
        if error_info:
            # 只打印前 1 个错误，避免刷屏
            # if skipped_count < 1:
            #     print(f"    [WARN] Line {i} Mismatch: {error_info.get('orig', 'Unknown')}")
            if skipped_count < 5:
                print(f"    [WARN] Mismatch at Sentence Index {i}:") # 这里的 i 就是句子的索引
                print(f"           Target Word: '{error_info.get('orig_norm', 'Unknown')}'")
                print(f"           Built Word:  '{error_info.get('built', 'Unknown')}'")
            skipped_count += 1
            continue
        
        # 构建数据项
        for w, t, g in zip(words, tags, groups):
            labels = [t] + ["<PAD>"] * (len(g) - 1)
            dataset.append({
                "id": len(dataset),
                "orig_word": w,
                "orig_tag": t,
                "subwords": g,
                "train_labels": labels
            })

    # 保存结果
    if dataset:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"    [SUCCESS] Generated: {os.path.basename(out_path)} ({len(dataset)} words)")
        if skipped_count > 0:
            print(f"              (Skipped {skipped_count} sentences due to mismatch)")
    else:
        print(f"    [FAIL] No valid data found.")

# def main():
#     # --- 路径配置 ---
#     base_dir = "./pilot_data/ud_data"
    
#     # 根据你的要求修改目录名
#     sub_dir = os.path.join(base_dir, "subword") # 之前的 subwords 改为了 subword
#     text_dir = os.path.join(base_dir, "text")
#     tags_dir = os.path.join(base_dir, "tags")
#     output_dir = os.path.join(base_dir, "aligned_json")

#     # 检查输入目录是否存在
#     if not os.path.exists(sub_dir):
#         print(f"[CRITICAL] Subwords directory not found: {sub_dir}")
#         print("           (Did you mean 'subwords' plural?)")
#         return

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 获取 subword 文件
#     files = [f for f in os.listdir(sub_dir) if not f.startswith('.')]
#     print(f"--- Scanning '{os.path.basename(sub_dir)}' folder ({len(files)} files) ---")

#     for filename in files:
#         # filename example: "et_dev.bpe"
#         # 1. 解析文件名
#         try:
#             parts = filename.rsplit('.', 1)
#             if len(parts) != 2:
#                 print(f"[SKIP] Invalid format: {filename}")
#                 continue
            
#             base_prefix = parts[0]     # "et_dev"
#             tokenizer_type = parts[1]  # "bpe"
#             text_lookup_key = base_prefix  # 默认键值
#             # obpe specific rule: l2_l1_split -> l2_split
#             if tokenizer_type == 'obpe' :
#                 token_parts = base_prefix.split('_')
#                 if len(token_parts) >= 3:
#                     l2 = token_parts[0] # sme, kpv, hu
#                     split = token_parts[-1] # train, dev, test
#                     text_lookup_key = f"{l2}_{split}" # -> sme_train
            
#             # 应用前缀映射 (防止 finnish vs fi 的问题)
#             for bad, good in NAME_MAPPING.items():
#                 if base_prefix.startswith(bad):
#                     base_prefix = base_prefix.replace(bad, good, 1)
#                     break

#             # 2. 构造目标文件名 (根据你的新规则)
#             # Pattern: lan_split_v5.txt / .tags
#             txt_filename = f"{text_lookup_key}_v5.txt" 
#             tag_filename = f"{text_lookup_key}_v5.tags"
            
#             # 3. 构造输出文件名
#             # Suffix: _aligned_v3.json
#             output_filename = f"{base_prefix}_{tokenizer_type}_aligned_v3.json"

#             # 4. 完整路径
#             subword_path = os.path.join(sub_dir, filename)
#             txt_path = os.path.join(text_dir, txt_filename)
#             tag_path = os.path.join(tags_dir, tag_filename)
#             output_path = os.path.join(output_dir, output_filename)

#             # 5. 执行处理
#             if os.path.exists(txt_path) and os.path.exists(tag_path):
#                 print(f"\nProcessing: {filename}")
#                 process_single_pair(subword_path, txt_path, tag_path, output_path)
#             else:
#                 print(f"Skipping {filename}:")
#                 if not os.path.exists(txt_path): print(f"  Missing Text: {txt_filename}")
#                 if not os.path.exists(tag_path): print(f"  Missing Tags: {tag_filename}")
#                 continue

#         except Exception as e:
#             print(f"[ERROR] Failed to process {filename}: {e}")

#     print(f"\n--- DONE ---")


def main():
    base_dir = "./pilot_data/ud_data"

    sub_dir = os.path.join(base_dir, "subword")
    text_dir = os.path.join(base_dir, "text")
    tags_dir = os.path.join(base_dir, "tags")
    output_dir = os.path.join(base_dir, "aligned_json")

    for d in [sub_dir, text_dir, tags_dir]:
        if not os.path.exists(d):
            print(f"[CRITICAL] Missing directory: {d}")
            return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --------- EXPLICIT CONFIG ---------
    languages = ["sme", "hu"]
    splits = ["train", "dev", "test"]   # adjust if needed
    # -----------------------------------

    print("--- Running explicit SME/HU OBPE alignments ---")

    for lang in languages:
        for split in splits:
            subword_filename = f"{lang}_et_{split}.obpe"
            text_filename = f"{lang}_{split}_v5.txt"
            tag_filename = f"{lang}_{split}_v5.tags"
            output_filename = f"{lang}_et_{split}_obpe_aligned_v3.json"

            subword_path = os.path.join(sub_dir, subword_filename)
            txt_path = os.path.join(text_dir, text_filename)
            tag_path = os.path.join(tags_dir, tag_filename)
            out_path = os.path.join(output_dir, output_filename)

            print(f"\nProcessing: {subword_filename}")

            missing = False
            if not os.path.exists(subword_path):
                print(f"  [SKIP] Missing subword: {subword_filename}")
                missing = True
            if not os.path.exists(txt_path):
                print(f"  [SKIP] Missing text: {text_filename}")
                missing = True
            if not os.path.exists(tag_path):
                print(f"  [SKIP] Missing tags: {tag_filename}")
                missing = True

            if missing:
                continue

            process_single_pair(subword_path, txt_path, tag_path, out_path)

    print("\n--- DONE ---")

if __name__ == "__main__":
    main()