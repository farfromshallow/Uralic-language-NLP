import requests
import os

INPUT_DIR = "./pilot_data/ud_data/pre-tok_conllu"
MODEL_NAME = "komi_zyrian-lattice-ud-2.12"
# hu: hungarian-szeged-ud-2.12
# sme: north_sami-giella-ud-2.12
# kpv: komi_zyrian-lattice-ud-2.12
def parse_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(input_path):
        print(f"Cannot find: {input_path}")
        return

    url = "http://lindat.mff.cuni.cz/services/udpipe/api/process"

    print(f"Uploading and parsing: {filename} ...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        conllu_data = f.read()

    params = {
        "model": MODEL_NAME,
        "tagger": "1",     
        "parser": "1",
        "input": "conllu"      
    }
    
    try:
        files = {"data": conllu_data}
        response = requests.post(url, data=params, files=files)
        response.raise_for_status() 
        
        result_json = response.json()
        
        output_path = input_path + ".parsed"
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(result_json['result'])
        
        print(f"Save output to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    target_file = input("conllu input: ").strip()
    parse_file(target_file)