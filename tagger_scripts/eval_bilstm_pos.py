'''
python tagger_scripts/eval_bilstm_pos.py \
    --model_path ./models/bilstm_pos/bpe/sme/best_model.pt \
    --test_file ./pilot_data/ud_data/aligned_json/sme_test_bpe_aligned_v3.json \
    --batch_size 32 \
    --seed 42
'''

import torch
import json
import argparse
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
# Import the classes from your training script 
from train_bilstm_pos import BiLSTMPOSTagger, JSONPOSDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_best_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_model.pt")
    parser.add_argument('--test_file', type=str, required=True, help="Path to test .json file")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    set_seed(args.seed)

    
    # 1. Load the Checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path)
    
    # Extract saved components
    word_to_idx = checkpoint['word_to_idx']
    tag_to_idx = checkpoint['tag_to_idx']
    saved_args = checkpoint['args']
    model_state_dict = checkpoint['model_state_dict']

    # 2. Re-create the Model Structure
    model = BiLSTMPOSTagger(
        vocab_size=len(word_to_idx),
        tagset_size=len(tag_to_idx),
        embedding_dim=saved_args['embedding_dim'],
        hidden_dim=saved_args['hidden_dim']
    )
    
    # Load the weights
    model.load_state_dict(model_state_dict)
    model.eval() # Set to evaluation mode (disables dropout)

    # 3. Load Test Data
    print(f"Loading test data: {args.test_file}")
    test_dataset = JSONPOSDataset(args.test_file, word_to_idx, tag_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 4. Run Prediction
    all_preds = []
    all_targets = []

    print("Running evaluation...")
    with torch.no_grad():
        for words, tags, mask in test_loader:
            tag_scores = model(words)
            predicted = tag_scores.argmax(2)

            # Filter out padding (using the mask)
            # mask=1 means real word, mask=0 means padding
            active_mask = (mask == 1) & (tags != tag_to_idx['<PAD>'])
            
            # Extract valid predictions and targets
            preds_clean = predicted[active_mask].cpu().numpy()
            targets_clean = tags[active_mask].cpu().numpy()

            all_preds.extend(preds_clean)
            all_targets.extend(targets_clean)

    # 5. Calculate & Print Metrics
    # Invert tag_to_idx to get actual tag names for the report
    idx_to_tag = {v: k for k, v in tag_to_idx.items() if k != '<PAD>'}
    
    # Get unique labels present in the data to avoid errors if some tags are missing in test set
    labels_in_test = sorted(list(set(all_targets)))
    target_names = [idx_to_tag[i] for i in labels_in_test]

    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    acc = f1_score(all_targets, all_preds, average='micro') # Micro F1 == Accuracy

    print("\n" + "="*30)
    print(f"RESULTS FOR SAVED MODEL")
    print("="*30)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print("-" * 30)
    print("Detailed Classification Report:")
    print(classification_report(all_targets, all_preds, labels=labels_in_test, target_names=target_names, digits=4))

if __name__ == "__main__":
    evaluate_best_model()