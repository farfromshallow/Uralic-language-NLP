import os
import random

def extract_unique_parallel_sentences(file1_path, file2_path, output_file1_path, output_file2_path, num_sentences=2542):
    """
    Extracts a specified number of unique parallel sentences from two files.
    A sentence is considered unique if its content (after stripping whitespace)
    has not been seen before in either file.

    Args:
        file1_path (str): Path to the first input file.
        file2_path (str): Path to the second input file (parallel to file1).
        output_file1_path (str): Path to save the extracted sentences from file1.
        output_file2_path (str): Path to save the extracted sentences from file2.
        num_sentences (int): The target number of unique parallel sentences to extract.
    """
    extracted_sentences1 = []
    extracted_sentences2 = []
    
    # Read all lines first
    all_lines1 = []
    all_lines2 = []

    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2:
            for line1, line2 in zip(f1, f2):
                clean_line1 = line1.strip()
                clean_line2 = line2.strip()

                # Skip empty lines
                if not clean_line1 or not clean_line2:
                    continue
                
                all_lines1.append(clean_line1)
                all_lines2.append(clean_line2)
        
        # Pair up lines and filter for uniqueness within each file
        unique_pairs = []
        seen_in_file1 = set()
        seen_in_file2 = set()
        
        for line1, line2 in zip(all_lines1, all_lines2):
            if line1 not in seen_in_file1 and line2 not in seen_in_file2:
                unique_pairs.append((line1, line2))
                seen_in_file1.add(line1)
                seen_in_file2.add(line2)
        
        # Randomly sample from unique pairs
        if len(unique_pairs) > num_sentences:
            sampled_pairs = random.sample(unique_pairs, num_sentences)
        else:
            sampled_pairs = unique_pairs

        # Ensure output directories exist
        os.makedirs(os.path.dirname(output_file1_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_file2_path), exist_ok=True)

        with open(output_file1_path, 'w', encoding='utf-8') as out_f1, \
             open(output_file2_path, 'w', encoding='utf-8') as out_f2:
            for s1, s2 in sampled_pairs:
                out_f1.write(s1 + '\n')
                out_f2.write(s2 + '\n')

        print(f"Extracted {len(sampled_pairs)} unique parallel sentences.")
        print(f"Output for file 1 saved to: {output_file1_path}")
        print(f"Output for file 2 saved to: {output_file2_path}")

    except FileNotFoundError as e:
        print(f"Error: One of the input files not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define input and output paths
    # These paths are examples, please adjust them to your actual file locations.
    # Assuming your current working directory is 'Uralic-language-NLP'
    
    # Example for Finnish-English parallel corpus
    fi_train_path = "./downstream_task/joeynmt/data/fi-tatoeba.fi"
    en_train_path = "./downstream_task/joeynmt/data/fi-tatoeba.en"

    output_fi_path = "./downstream_task/joeynmt/data/extracted_train.fi"
    output_en_path = "./downstream_task/joeynmt/data/extracted_train.en"

    # Number of sentences to extract
    target_sentences = 2542

    extract_unique_parallel_sentences(fi_train_path, en_train_path,
                                      output_fi_path, output_en_path,
                                      target_sentences)