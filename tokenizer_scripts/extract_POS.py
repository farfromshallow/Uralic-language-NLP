def extract_tags_from_conllu(conllu_file, output_file):
    """Extract POS tags from CoNLL-U file."""
    import os
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(conllu_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        current_tags = []
        
        for line in f_in:
            line = line.strip()
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Empty line = sentence boundary
            if not line:
                if current_tags:
                    f_out.write(' '.join(current_tags) + '\n')
                    current_tags = []
                continue
            
            # Parse token line
            parts = line.split('\t')
            if len(parts) >= 4:
                token_id = parts[0].strip()
                
                # Only process regular tokens (token_id is a positive integer)
                # Skip multi-word tokens (format: "1-2") and empty nodes (format: ".1" or "1.1")
                try:
                    # Check if token_id is a valid integer
                    int(token_id)
                    # If it's a valid integer, it's a regular token
                    pos_tag = parts[3].strip()  # UPOS column
                    if pos_tag:  # Only add if POS tag is not empty
                        current_tags.append(pos_tag)
                except ValueError:
                    # token_id is not an integer, skip (multi-word token or empty node)
                    continue
        
        # Handle last sentence if file doesn't end with empty line
        if current_tags:
            f_out.write(' '.join(current_tags) + '\n')

# Extract tags for all datasets
#extract_tags_from_conllu('./pilot_data/ud_data/extracted_genres/fi_blog_wiki-train.conllu', './pilot_data/ud_data/tags/finnish_train.tags')
#extract_tags_from_conllu('./pilot_data/ud_data/extracted_genres/fi_blog_wiki-dev.conllu', './pilot_data/ud_data/tags/finnish_dev.tags')
extract_tags_from_conllu('./pilot_data/ud_data/UD_Estonian-EWT/et_ewt-ud-test.conllu', './pilot_data/ud_data/tags/estonian_test_1.tags')