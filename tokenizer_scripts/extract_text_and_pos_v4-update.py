import os
import re

def precise_punctuation_separation(text, language='finnish'):
    """
    Add white space to punctuation precisely, preserving certain patterns
    but separating different punctuation marks
    """
    # Step 1: Protect patterns that should not be separated
    if language == 'finnish':
        # Protect Finnish number+colon+case ending patterns (e.g., 4:stä)
        text = re.sub(r'\b(\d+):([a-zA-ZäöåÄÖÅ]+)\b', r'FINNISH_NUM_COLON_\1_COLON_\2', text)

    ##########################################################################################
    if language == 'estonian':
        # Protect Estonian words with apostrophes (e.g., Google'isse, Homasho't)
        text = re.sub(r"([A-Za-zäöüõÄÖÜÕ]+)(['\u2019])([a-zA-ZäöüõÄÖÜÕ]+)",
                      r"APOSTROPHE_PROTECTED_\1_APOSTROPHE_\3", text)
    
    # Compound words (e.g., cha-cha)
    text = re.sub(r'(\w+)-(\w+)', r'COMPOUND_\1_HYPHEN_\2', text)
    
    # Numbers (e.g., 100-200, -30, 1,000, 30:n)
    text = re.sub(r'\b(-?\d+)-(\d+)\b', r'NUMBER_\1_HYPHEN_\2', text)
    text = re.sub(r'\b(-?\d+),(\d+)\b', r'NUMBER_\1_COMMA_\2', text)
    text = re.sub(r'\b(-?\d+):(\d+)\b', r'NUMBER_\1_COLON_\2', text)
    
    # Ellipsis (e.g., ..., …)
    text = re.sub(r'\.{2,}|\u2026', 'ELLIPSIS_PROTECTED', text)
    
    # Step 2: Separate punctuation that should be independent
    # Handle combinations of different punctuation types
    # Close punctuation followed by sentence-ending punctuation
    text = re.sub(r'([)\]"\'`\u201d\u2019\u00bb])([.,!?;:])', r'\1 \2', text)
    # Sentence-ending punctuation followed by open punctuation
    text = re.sub(r'([.,!?;:])([(\["\'`\u201c\u2018\u00ab])', r'\1 \2', text)
    
    # Sentence-level punctuation (add space before, not after to avoid double spacing)
    text = re.sub(r'([.!?;:])(?=\s|$)', r' \1', text)

    # Handle different types of quotation marks
    text = re.sub(r'(["\'\u201c\u201d\u2018\u2019\u00ab\u00bb])', r' \1 ', text)
    
    # Brackets
    text = re.sub(r'([()])', r' \1 ', text)
    
    # Comma (except in numbers - already protected)
    text = re.sub(r',(?=\s|$)', r' , ', text)
    
    # Dash (except in compound words and numbers)
    text = re.sub(r'(?<=\s)-(?=\s|$)', r' - ', text)
    #################################################################################
    
    # Step 3: Restore protected patterns
    if language == 'finnish':
        text = re.sub(r'FINNISH_NUM_COLON_(\d+)_COLON_([a-zA-ZäöåÄÖÅ]+)', r'\1:\2', text)

    if language == 'estonian':
        text = re.sub(r"APOSTROPHE_PROTECTED_([A-Za-z]+)_APOSTROPHE_([a-z]+)", r"\1'\2", text)

    text = re.sub(r'COMPOUND_(\w+)_HYPHEN_(\w+)', r'\1-\2', text)
    text = re.sub(r'NUMBER_(-?\d+)_HYPHEN_(\d+)', r'\1-\2', text)
    text = re.sub(r'NUMBER_(-?\d+)_COMMA_(\d+)', r'\1,\2', text)
    text = re.sub(r'ELLIPSIS_PROTECTED', r'...', text)
    text = re.sub(r'NUMBER_(-?\d+)_COLON_(\d+)', r'\1:\2', text)
    
    # Step 4: Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_and_pos_from_conllu(conllu_file, text_output_file, pos_output_file, language='finnish'):
    """Extract both plain text sentences and POS tags from CoNLL-U file."""
    
    # Create output directories if they don't exist
    for output_file in [text_output_file, pos_output_file]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
    
    with open(conllu_file, 'r', encoding='utf-8') as f_in, \
         open(text_output_file, 'w', encoding='utf-8') as f_text, \
         open(pos_output_file, 'w', encoding='utf-8') as f_pos:
        
        current_sentence_text = None
        current_pos_tags = []
        current_tokens = []
        sentence_count = 0
        
        for line in f_in:
            line = line.strip()
            
            # Extract text from comment lines
            if line.startswith('# text = '):
                current_sentence_text = line.replace('# text = ', '').strip()
                continue
            
            # Skip other comments
            if line.startswith('#'):
                continue
            
            # Empty line = sentence boundary
            if not line:
                if (current_sentence_text or current_tokens) and current_pos_tags:
                    # Determine text to output
                    if current_sentence_text:
                        output_text = current_sentence_text
                    else:
                        output_text = ' '.join(current_tokens)
                    
                    # Apply precise punctuation separation
                    output_text = precise_punctuation_separation(output_text, language)
                    
                    # Write text
                    f_text.write(output_text + '\n')
                    
                    # Write POS tags
                    f_pos.write(' '.join(current_pos_tags) + '\n')
                    
                    sentence_count += 1
                    
                    # Reset for next sentence
                    current_sentence_text = None
                    current_pos_tags = []
                    current_tokens = []
                continue
            
            # Parse token line
            parts = line.split('\t')
            if len(parts) >= 4:
                token_id = parts[0].strip()
                token_form = parts[1].strip()
                pos_tag = parts[3].strip()
                
                try:
                    int(token_id)
                    if pos_tag:
                        current_pos_tags.append(pos_tag)
                    if token_form:
                        current_tokens.append(token_form)
                except ValueError:
                    continue
        
        # Handle last sentence
        if (current_sentence_text or current_tokens) and current_pos_tags:
            if current_sentence_text:
                output_text = current_sentence_text
            else:
                output_text = ' '.join(current_tokens)
            
            output_text = precise_punctuation_separation(output_text, language)
            
            f_text.write(output_text + '\n')
            f_pos.write(' '.join(current_pos_tags) + '\n')
            sentence_count += 1
        
        print(f"Extracted {sentence_count} sentences from {conllu_file}")
        print(f"Text saved to: {text_output_file}")
        print(f"POS tags saved to: {pos_output_file}")

# Test function
def test_punctuation_separation():
    """Test the punctuation separation function"""
    test_cases = [
        'He said, "No."',
        'She replied, "Yes!" and then left.',
        'The list (item 1, item 2) is long.',
        'He paused... and then continued.',
        'This is a compound-word example.',
        'The number is 1,000 - 2,000.',
        'She said, "Hello!" and then smiled.)',
        'The text was "quoted".',
        'Special cases: ), .", etc.',
    ]
    
    print("Testing punctuation separation:")
    for test in test_cases:
        result = precise_punctuation_separation(test)
        print(f"Input:  '{test}'")
        print(f"Output: '{result}'")
        print()

# Usage example
if __name__ == "__main__":
    # Run tests
    test_punctuation_separation()
    
    # Extract data
    # extract_text_and_pos_from_conllu(
    #     './pilot_data/ud_data/extracted_genres/fi_blog_wiki-train.conllu',
    #     './pilot_data/ud_data/text/fi_train_v4.txt',
    #     './pilot_data/ud_data/tags/fi_train_v4.tags',
    #     language='finnish'
    # )
    # extract_text_and_pos_from_conllu(
    #     './pilot_data/ud_data/extracted_genres/fi_blog_wiki-dev.conllu',
    #     './pilot_data/ud_data/text/fi_dev_v4.txt',
    #     './pilot_data/ud_data/tags/fi_train_v4.tags',
    #     language='finnish'
    # )    
    extract_text_and_pos_from_conllu(
        './pilot_data/ud_data/UD_Estonian-EWT/et_ewt-ud-test.conllu',
        './pilot_data/ud_data/text/et_test_v4.txt',
        './pilot_data/ud_data/tags/et_test_v4.tags',
        language='estonian'
    )