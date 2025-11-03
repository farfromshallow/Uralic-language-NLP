from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("/Users/Ingrid/OBPE/tokenizers/fi_et_obpe_tokenizer_ud.json")

def tokenize_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # Tokenize
            encoding = tokenizer.encode(line)
            tokens = encoding.tokens
            
            # Write space-separated tokens
            f_out.write(' '.join(tokens) + '\n')

# Apply to all datasets
tokenize_file('./pilot_data/ud_data/extracted_genres/fi_blog_wiki-train.txt', './pilot_data/ud_data/subwords/ud_fi_train_obpe.subwords')
tokenize_file('./pilot_data/ud_data/extracted_genres/fi_blog_wiki-dev.txt', './pilot_data/ud_data/subwords/ud_fi_dev_obpe.subwords')
tokenize_file('./pilot_data/ud_data/UD_Estonian-EWT/et_ewt-ud-test.txt', './pilot_data/ud_data/subwords/ud_et_test_obpe.subwords')