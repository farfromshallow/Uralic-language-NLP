import sentencepiece as spm
import os

# ================= 配置区域 =================

# 1. 定义你的 Tokenizer 模型路径
MODELS = {
    "bpe":     "./Uralic-language-NLP/models/bpe",      # 你的 UD BPE 模型
    "unigram": "./Uralic-language-NLP/models/unigram",  # 你的 UD Unigram 模型
    "obpe":    "./Uralic-language-NLP/models/obpe",     # 你的 UD OBPE 模型
    "en":      "./Uralic-language-NLP/models/en_bpe"       # 英语目标端模型 (假设都用同一个)
}

# 2. 定义需要处理的文件列表
# 格式: ("语言代码", "原始文件路径", "输出目录")
FILES_TO_PROCESS = [
    # --- training (High Resource) ---
    ("src", "data/raw/train.fi",  "data/processed"),
    ("trg", "data/raw/train.en",  "data/processed"), 
    
    ("src", "data/raw/train.et",  "data/processed"),
    ("trg", "data/raw/train.et.en", "data/processed"), 

    # --- Training set (Low Resource - North Sami) ---
    ("src", "downstream_task/joeynmt/data/sme_train.sme.txt", "data/processed"),
    ("trg", "downstream_task/joeynmt/data/sme_train.en.txt", "data/processed"),

    # --- dev & test (Sami only) ---
    ("src", "downstream_task/joeynmt/data/sme_dev.sme.txt",   "data/processed"),
    ("trg", "downstream_task/joeynmt/data/sme_dev.en.txt", "data/processed"),

    ("src", "downstream_task/joeynmt/data/sme_test.sme.txt",  "data/processed"),
    ("trg", "downstream_task/joeynmt/data/sme_test.en.txt", "data/processed"),
]

# ===========================================

def encode_file(sp_model, input_file, output_file):
    print(f"   Processing: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            # segment and write, link using space
            # out_type=str: SentencePiece output format ['Hel', 'sinki']
            tokens = sp_model.encode(line, out_type=str)
            
            # link tokens with space
            fout.write(" ".join(tokens) + "\n")

def main():
    # 我们要跑三轮：BPE, Unigram, OBPE
    # 注意：英语目标端 (trg) 通常只需要跑一次 BPE 即可，
    # 但为了文件后缀对应方便，我们可以生成三份，或者只生成一份 .bpe
    
    experiment_types = ["bpe", "unigram", "obpe"]

    for exp_type in experiment_types:
        print(f"\n🚀 开始处理实验组: {exp_type.upper()} ...")
        
        # load model (UD trained)
        src_model_path = MODELS[exp_type]
        trg_model_path = MODELS["en"] # 英语通常不变
        
        if not os.path.exists(src_model_path):
            print(f"❌ 找不到模型文件: {src_model_path}，跳过。")
            continue

        sp_src = spm.SentencePieceProcessor(model_file=src_model_path)
        sp_trg = spm.SentencePieceProcessor(model_file=trg_model_path)

        for role, raw_path, out_dir in FILES_TO_PROCESS:
            if not os.path.exists(raw_path):
                print(f"⚠️  警告: 找不到原始文件 {raw_path}")
                continue
            
            # 构造输出文件名
            # 例如: train.fi -> train.fi.obpe
            filename = os.path.basename(raw_path)
            out_path = os.path.join(out_dir, f"{filename}.{exp_type}")
            
            # 根据角色选择模型
            if role == "src":
                # 源语言 (Fi/Et/Sme) -> 使用实验模型 (BPE/Unigram/OBPE)
                encode_file(sp_src, raw_path, out_path)
            else:
                # 目标语言 (En) -> 使用英语模型
                # 注意：为了脚本通用，通常英语我们只生成一份 .bpe
                # 但如果你想保持文件名一致，也可以生成 .en.obpe (内容其实是英语BPE)
                # 这里为了清晰，我们假设英语后缀总是跟随实验类型
                encode_file(sp_trg, raw_path, out_path)

if __name__ == "__main__":
    main()