import os

def extract_text_and_pos_from_conllu(conllu_file, text_output_file, pos_output_file, language='finnish'):
    
    with open(conllu_file, 'r', encoding='utf-8') as f_in, \
         open(text_output_file, 'w', encoding='utf-8') as f_text, \
         open(pos_output_file, 'w', encoding='utf-8') as f_pos:
        
        current_pos_tags = []
        current_tokens = []
        sentence_count = 0
        
        for line in f_in:
            line = line.strip()
            
            # 【rely on tokens only
            if line.startswith('#'):
                continue
            
            # Empty line = sentence boundary
            if not line:
                if current_tokens and current_pos_tags:
                    # delete regular expression process
                    output_text = ' '.join(current_tokens)
                    
                    f_text.write(output_text + '\n')
                    f_pos.write(' '.join(current_pos_tags) + '\n')
                    
                    sentence_count += 1
                    
                    current_pos_tags = []
                    current_tokens = []
                continue
            
            # Parse token line
            parts = line.split('\t')
            # info in CoNLL-U: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
            if len(parts) >= 10: # 标准 CoNLL-U 至少 10 列
                token_id = parts[0].strip()
                token_form = parts[1].strip() # 这是单词
                pos_tag = parts[3].strip()    # 这是 UPOS 标签
                
                # Multi-word token ID (range IDs)
                
                if '-' in token_id or '.' in token_id:
                    continue
                
                try:
                    current_pos_tags.append(pos_tag)
                    current_tokens.append(token_form)
                except ValueError:
                    continue
        
        # Handle last sentence
        if current_tokens and current_pos_tags:
            output_text = ' '.join(current_tokens)
            f_text.write(output_text + '\n')
            f_pos.write(' '.join(current_pos_tags) + '\n')
            sentence_count += 1
            
        print(f"Extracted {sentence_count} sentences from {conllu_file}")
        print(f"  -> Saved to: {os.path.basename(text_output_file)} & {os.path.basename(pos_output_file)}\n")

if __name__ == "__main__":
    

    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/extracted_genres/fi_extracted_genres-train.conllu',
    #     text_output_file='./pilot_data/ud_data/text/fi_train_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/fi_train_v5.tags'
    # )


    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/extracted_genres/fi_extracted_genres-dev.conllu',
    #     text_output_file='./pilot_data/ud_data/text/fi_dev_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/fi_dev_v5.tags'
    # )

    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/extracted_genres/fi_extracted_genres-test.conllu',
    #     text_output_file='./pilot_data/ud_data/text/fi_test_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/fi_test_v5.tags'
    # )

    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/extracted_genres/ru_extracted_genres-train.conllu',
    #     text_output_file='./pilot_data/ud_data/text/ru_train_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/ru_train_v5.tags'
    # )


    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/extracted_genres/ru_extracted_genres-dev.conllu',
    #     text_output_file='./pilot_data/ud_data/text/ru_dev_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/ru_dev_v5.tags'
    # )

    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/extracted_genres/ru_extracted_genres-test.conllu',
    #     text_output_file='./pilot_data/ud_data/text/ru_test_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/ru_test_v5.tags'
    # )

    extract_text_and_pos_from_conllu(
        conllu_file='./pilot_data/ud_data/UD_Estonian-EWT/et_ewt-ud-test.conllu',
        text_output_file='./pilot_data/ud_data/text/et_test_v5.txt',
        pos_output_file='./pilot_data/ud_data/tags/et_test_v5.tags'
    )

    extract_text_and_pos_from_conllu(
        conllu_file='./pilot_data/ud_data/UD_Estonian-EWT/et_ewt-ud-dev.conllu',
        text_output_file='./pilot_data/ud_data/text/et_dev_v5.txt',
        pos_output_file='./pilot_data/ud_data/tags/et_dev_v5.tags'
    )

    extract_text_and_pos_from_conllu(
        conllu_file='./pilot_data/ud_data/UD_Estonian-EWT/et_ewt-ud-train.conllu',
        text_output_file='./pilot_data/ud_data/text/et_train_v5.txt',
        pos_output_file='./pilot_data/ud_data/tags/et_train_v5.tags'
    )

    extract_text_and_pos_from_conllu(
        conllu_file='./pilot_data/ud_data/UD_Hungarian-Szeged/hu_szeged-ud-dev.conllu',
        text_output_file='./pilot_data/ud_data/text/hu_dev_v5.txt',
        pos_output_file='./pilot_data/ud_data/tags/hu_dev_v5.tags'
    )
    extract_text_and_pos_from_conllu(
        conllu_file='./pilot_data/ud_data/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu',
        text_output_file='./pilot_data/ud_data/text/hu_test_v5.txt',
        pos_output_file='./pilot_data/ud_data/tags/hu_test_v5.tags'
    )
    extract_text_and_pos_from_conllu(
        conllu_file='./pilot_data/ud_data/UD_Hungarian-Szeged/hu_szeged-ud-train.conllu',
        text_output_file='./pilot_data/ud_data/text/hu_train_v5.txt',
        pos_output_file='./pilot_data/ud_data/tags/hu_train_v5.tags'
    )
    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/UD_North_Sami-Giella/sme_giella-ud-full.conllu',
    #     text_output_file='./pilot_data/ud_data/text/sme_full_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/sme_full_v5.tags'
    # )
    # extract_text_and_pos_from_conllu(
    #     conllu_file='./pilot_data/ud_data/UD_Kimi_Zyrian-Lattice/kpv_lattice-ud-full.conllu',
    #     text_output_file='./pilot_data/ud_data/text/kpv_full_v5.txt',
    #     pos_output_file='./pilot_data/ud_data/tags/kpv_full_v5.tags'
    # )