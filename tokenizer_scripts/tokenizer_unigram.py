import sentencepiece as spm
import os

data_folder = "pilot_data"

# Input files for Estonian and Finnish
et_file = os.path.join(data_folder, "ELRC-EMEA_1k.en-et.et")
fi_file = os.path.join(data_folder, "ELRC-EMEA_1k.en-fi.fi")

# Function to train and encode using Unigram model
def train_and_encode_unigram(input_file, model_prefix, vocab_size=5000):
    # Train Unigram model with safe configuration
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type=unigram --character_coverage=1.0 "
        f"--unk_piece=<unk> --hard_vocab_limit=false"
    )

    # Load model
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Define output file in pilot_data
    output_file = os.path.join(
        data_folder,
        os.path.basename(input_file)
        .replace(".et", ".unigram.et")
        .replace(".fi", ".unigram.fi")
    )

    # Encode sentences
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            pieces = sp.encode(line.strip(), out_type=str)
            f_out.write(" ".join(pieces) + "\n")

    print(f"Unigram tokenized file saved at {output_file}")

# Train and encode for Estonian and Finnish
train_and_encode_unigram(et_file, "unigram_et")
train_and_encode_unigram(fi_file, "unigram_fi")