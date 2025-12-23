'''
python tokenizer_bpe_unigram_correct.py \
    --tokenizer bpe \
    --train_file ./pilot_data/ud_data/text/finnish_train.txt \
    --eval_files ./pilot_data/ud_data/text/finnish_dev.txt ./pilot_data/ud_data/text/finnish_test.txt \
    --output_files ./pilot_data/ud_data/tokenized/finnish_dev.bpe ./pilot_data/ud_data/tokenized/finnish_test.bpe \
    --model_prefix finnish_bpe_model \
    --vocab_size 5000
'''

import sentencepiece as spm
import os
import argparse

def train_tokenizer(input_file, model_prefix, tokenizer_type, vocab_size=5000):
    """train the tokenizer and store the model"""
    if tokenizer_type == 'unigram':
        model_type = 'unigram'
    elif tokenizer_type == 'bpe':
        model_type = 'bpe'
    else:
        raise ValueError(f"不支持的 tokenizer 类型: {tokenizer_type}")
    
    # training step
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type={model_type} --character_coverage=1.0 "
        f"--unk_piece=<unk> --hard_vocab_limit=false"
    )
    
    print(f"{tokenizer_type.upper()} 模型训练完成: {model_prefix}.model")
    return f"{model_prefix}.model"

def encode_with_tokenizer(model_file, input_file, output_file):
    """tokenization step"""
    
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    # 编码句子
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        sentence_count = 0
        for line in f_in:
            line = line.strip()
            if line:  # 跳过空行
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
                       help='eval files')
    parser.add_argument('--output_files', nargs='+', required=True, 
                       help='output files')
    parser.add_argument('--model_prefix', required=True, 
                       help='model prefix')
    parser.add_argument('--vocab_size', type=int, default=5000, 
                       help='vocal size (keep consistency among tokenizers)')

    args = parser.parse_args()

    if len(args.eval_files) != len(args.output_files):
        raise ValueError("Amount of sentences should be aligned")

    # 1. 使用训练集训练 tokenizer
    model_file = train_tokenizer(
        args.train_file, 
        args.model_prefix, 
        args.tokenizer, 
        args.vocab_size
    )
    
    # 2. 对每个评估文件进行编码
    for eval_file, output_file in zip(args.eval_files, args.output_files):
        encode_with_tokenizer(model_file, eval_file, output_file)
    
    print(f"\nAll done!")
    print(f"Model saved as: {model_file}")
    print(f"Output file: {', '.join(args.output_files)}")