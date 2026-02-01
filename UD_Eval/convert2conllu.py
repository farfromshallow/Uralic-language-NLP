import os

DEFAULT_INPUT_DIR = "./pilot_data/ud_data/subword" 
DEFAULT_OUTPUT_DIR = "./pilot_data/ud_data/pre-tok_conllu"


def process_file(filename):
    input_path = os.path.join(DEFAULT_INPUT_DIR, filename)
    
    if not os.path.exists(input_path):
        print(f" {DEFAULT_INPUT_DIR} cannot find '{filename}'")
        return

    output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename + ".conllu")
    
    model_type = 'obpe' if 'obpe' in filename.lower() else 'bpe'
    
    if not os.path.exists(DEFAULT_OUTPUT_DIR):
        os.makedirs(DEFAULT_OUTPUT_DIR)


    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        print(f"identify {filename} as {model_type.upper()} model")
        
        for line in f_in:
            subwords = line.strip().split()
            if not subwords:
                continue
                
            tokens = []
            current_token = ""
            for sub in subwords:
                if model_type == 'bpe': 
                    if sub.startswith("▁"):
                        if current_token: tokens.append(current_token)
                        current_token = sub.replace("▁", "")
                    else:
                        current_token += sub
                elif model_type == 'obpe': 
                    if sub.endswith("</w>"):
                        current_token += sub.replace("</w>", "")
                        tokens.append(current_token)
                        current_token = ""
                    else:
                        current_token += sub
            
            if current_token: tokens.append(current_token)

            for i, token in enumerate(tokens, 1):
                f_out.write(f"{i}\t{token}\t_\t_\t_\t_\t0\t_\t_\t_\n")
            f_out.write("\n")

    print(f"Save converted conllu to: {output_path}")

if __name__ == "__main__":
    # 运行时询问文件名
    target_file = input("Subword input: ").strip()
    process_file(target_file)