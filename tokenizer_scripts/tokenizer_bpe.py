import sentencepiece as spm
import os

# Paths to pilot data
data_folder = "pilot_data"
et_file = os.path.join(data_folder, "ELRC-EMEA_1k.en-et.et")
fi_file = os.path.join(data_folder, "ELRC-EMEA_1k.en-fi.fi")


# Function to train and encode BPE
def train_and_encode_bpe(input_file, prefix, vocab_size=5000):
    # Train BPE model
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 --unk_piece=<unk>"
    )

    # Load model
    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")

    # Define output file in the same folder
    output_file = os.path.join(data_folder, os.path.basename(input_file).replace(".et", ".bpe.et").replace(".fi", ".bpe.fi"))

    # Encode sentences
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            pieces = sp.encode(line.strip(), out_type=str)
            f_out.write(" ".join(pieces) + "\n")

    print(f"BPE tokenized file saved at {output_file}")

# Train and encode all files
train_and_encode_bpe(et_file, "bpe_et")
train_and_encode_bpe(fi_file, "bpe_fi")
