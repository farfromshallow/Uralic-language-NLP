import requests
import os
import sys
from ufal.udpipe import Model, Pipeline, ProcessingError # local model only
# MODEL_NAME = "komi_zyrian-lattice-ud-2.12" # API inference
# hu: hungarian-szeged-ud-2.12
# sme: north_sami-giella-ud-2.12
# kpv: komi_zyrian-lattice-ud-2.12

# ==========================================

INPUT_DIR = "./pilot_data/ud_data/pre-tok_conllu"

# Local
MODE = "LOCAL"                        
MODEL_TARGET = "UD_Eval/kpv_custom.udpipe"    

# API inference
# MODE = "API"                       
# MODEL_TARGET = "north_sami-giella-ud-2.12"  
# hu: hungarian-szeged-ud-2.12
# sme: north_sami-giella-ud-2.12
# kpv: komi_zyrian-lattice-ud-2.12

def run_local_parsing(filename, model_path):
    try:
        from ufal.udpipe import Model, Pipeline, ProcessingError
    except ImportError:
        print("❌ Cannot find ufal.udpipe library。pip install ufal.udpipe first")
        return

    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(model_path):
        print(f"❌ Cannot find local model file: {model_path}")
        return

    print(f"Loading model: {model_path} ...")
    model = Model.load(model_path)
    if not model:
        print("Model loading failed")
        return

    # Pipeline: model, tokenizer=none, tagger, parser, output=conllu
    pipeline = Pipeline(model, "conllu", "tagger", "parser", "conllu")
    
    print(f"Parsing: {filename} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        conllu_data = f.read()

    error = ProcessingError()
    processed_text = pipeline.process(conllu_data, error)

    if error.occurred():
        print(f"Error: {error.message}")
    else:
        output_path = input_path + ".parsed"
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(processed_text)
        print(f"Save parsed conllu to: {output_path}")


def run_api_parsing(filename, model_name):
    input_path = os.path.join(INPUT_DIR, filename)
    url = "http://lindat.mff.cuni.cz/services/udpipe/api/process"

    print(f"Uploading and parsing: {filename} (Model: {model_name}) ...")

    with open(input_path, 'r', encoding='utf-8') as f:
        conllu_data = f.read()

    params = {
        "model": model_name,
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

        print(f"Save parsed conllu to: {output_path}")

    except Exception as e:
        print(f"API Error: {e}")


def main():
    target_file = input(f"Target file ({INPUT_DIR}): ").strip()
    full_path = os.path.join(INPUT_DIR, target_file)

    if not os.path.exists(full_path):
        print(f"File does not exist: {full_path}")
        return

    if MODE == "LOCAL":
        run_local_parsing(target_file, MODEL_TARGET)
    elif MODE == "API":
        run_api_parsing(target_file, MODEL_TARGET)
    else:
        print("Config error")

if __name__ == "__main__":
    main()