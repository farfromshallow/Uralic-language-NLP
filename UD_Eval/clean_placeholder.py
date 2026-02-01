import os

def clean_conllu_format(input_path):
    # 自动生成新文件名，例如 sme_test.parsed -> sme_test.cleaned.conllu
    output_path = input_path.replace(".parsed", "") + ".cleaned.conllu"
    
    print(f"正在清洗文件: {input_path}")
    print(f"输出目标文件: {output_path}")
    
    line_count = 0
    fix_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        content = f_in.readlines()
        
        for i, line in enumerate(content):
            line_count += 1
            # 保持注释行和空行不变
            if line.startswith("#") or line.strip() == "":
                f_out.write(line)
                continue
            
            cols = line.split('\t')
            if len(cols) == 10:
                deps_val = cols[8] # 第 9 列
                # 容错核心逻辑：如果不是下划线且没有冒号，强制修复
                if deps_val != "_" and ":" not in deps_val:
                    cols[8] = "_"
                    fix_count += 1
                
                f_out.write('\t'.join(cols))
            else:
                f_out.write(line)
        
        # 强制修复：确保文件以空行结尾
        if content and content[-1].strip() != "":
            f_out.write("\n")
            print("已自动在文件末尾添加缺失的空行。")

    print(f"处理完成！共检查 {line_count} 行，修复了 {fix_count} 处非标准 DEPS 占位符。")

if __name__ == "__main__":
    # 请在这里填入你报错的文件路径
    target = "pilot_data/ud_data/UD_North_Sami-Giella/sme_giella-ud-test.conllu"
    clean_conllu_format(target)