'''
python tagger_scripts/train_bilstm_pos.py \
    --train_file ./pilot_data/ud_data/aligned_json/sme_train_bpe_aligned_v3.json \
    --dev_file ./pilot_data/ud_data/aligned_json/sme_dev_bpe_aligned_v3.json \
    --test_file ./pilot_data/ud_data/aligned_json/sme_test_bpe_aligned_v3.json \
    --model_dir ./models/bilstm_pos_bpe/ \
    --epochs 50 \
    --batch_size 32
    --seed 42
'''
# train_bilstm_pos_simple.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
import argparse
import os
import json
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #print(f"Random seed set to: {seed}")


class JSONPOSDataset(Dataset):
    def __init__(self, json_file, word_to_idx, tag_to_idx, max_len=128):
        self.sentences = []
        self.labels = []
        
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        current_subwords = []
        current_tags = []
        
        for item in data:
            # item structure: 
            # {"orig_word": "...", "subwords": ["...", "..."], "train_labels": ["TAG", "<PAD>"]}
            
            # Extend current sentence sequence
            current_subwords.extend(item['subwords'])
            current_tags.extend(item['train_labels'])
            
            # Heuristic Sentence Boundary Detection
            # UD sentences usually end with specific punctuation
            # If sequence gets too long, we also force a break to manage memory
            is_boundary = item['orig_word'] in ['.', '!', '?', '...'] 
            
            if is_boundary or len(current_subwords) >= max_len:
                # Process the sentence if it has content
                if current_subwords:
                    # Truncate if exceeding max_len (rare if split by punctuation)
                    trunc_words = current_subwords[:max_len]
                    trunc_tags = current_tags[:max_len]
                    
                    # Convert to IDs
                    word_ids = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in trunc_words]
                    tag_ids = [tag_to_idx.get(t, tag_to_idx['<PAD>']) for t in trunc_tags]
                    
                    # Padding
                    pad_len = max_len - len(word_ids)
                    padded_words = word_ids + [word_to_idx['<PAD>']] * pad_len
                    padded_tags = tag_ids + [tag_to_idx['<PAD>']] * pad_len
                    mask = [1] * len(word_ids) + [0] * pad_len
                    
                    self.sentences.append((padded_words, mask))
                    self.labels.append(padded_tags)
                
                # Reset
                current_subwords = []
                current_tags = []
        
        # Catch any leftover words as a final sentence
        if current_subwords:
            # ... (Same processing logic as above for leftover)
            trunc_words = current_subwords[:max_len]
            trunc_tags = current_tags[:max_len]
            word_ids = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in trunc_words]
            tag_ids = [tag_to_idx.get(t, tag_to_idx['<PAD>']) for t in trunc_tags]
            pad_len = max_len - len(word_ids)
            padded_words = word_ids + [word_to_idx['<PAD>']] * pad_len
            padded_tags = tag_ids + [tag_to_idx['<PAD>']] * pad_len
            mask = [1] * len(word_ids) + [0] * pad_len
            self.sentences.append((padded_words, mask))
            self.labels.append(padded_tags)

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words, mask = self.sentences[idx]
        return torch.tensor(words), torch.tensor(self.labels[idx]), torch.tensor(mask)

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTMPOSTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        return tag_scores

def build_vocab_from_json(json_files):
    word_freq = defaultdict(int)
    tag_set = set()
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # Add subwords
                for sub in item['subwords']:
                    word_freq[sub] += 1
                # Add tags (excluding <PAD> which we handle manually)
                for tag in item['train_labels']:
                    if tag != '<PAD>':
                        tag_set.add(tag)
    
    # 构建词汇表
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= 1:  # 出现一次就加入
            word_to_idx[word] = len(word_to_idx)
    
    # 构建标签映射
    tag_to_idx = {'<PAD>': 0} # PAD ID must strictly be 0 for NLLLoss
    for idx, tag in enumerate(sorted(list(tag_set))):
        tag_to_idx[tag] = idx + 1 # Shift index to accommodate PAD
    
    return word_to_idx, tag_to_idx

def train_bilstm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    
    os.makedirs(args.model_dir, exist_ok=True)
    set_seed(args.seed)
    
    # 构建词汇表
    print("Vocab...")
    word_to_idx, tag_to_idx = build_vocab_from_json([args.train_file, args.dev_file])
    
    print(f"Vocab Size: {len(word_to_idx)}")
    print(f"Tag Set Size: {len(tag_to_idx)}")
    
    # 创建数据集
    train_dataset = JSONPOSDataset(args.train_file, word_to_idx, tag_to_idx)
    dev_dataset = JSONPOSDataset(args.dev_file, word_to_idx, tag_to_idx)
    test_dataset = JSONPOSDataset(args.test_file, word_to_idx, tag_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Train Sentences: {len(train_dataset)}")
    print(f"Dev Sentences: {len(dev_dataset)}")
    print(f"Test Sentences: {len(test_dataset)}")

    # model setup
    model = BiLSTMPOSTagger(
        vocab_size=len(word_to_idx),
        tagset_size=len(tag_to_idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss(ignore_index=tag_to_idx['<PAD>'])
    
    # training loop
    best_accuracy = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (words, tags, mask) in enumerate(train_loader):
            optimizer.zero_grad()
            tag_scores = model(words)
            
            # Flatten for loss
            loss = criterion(tag_scores.view(-1, len(tag_to_idx)), tags.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # validation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for words, tags, mask in dev_loader:
                tag_scores = model(words)
                predicted = tag_scores.argmax(2)
                
                # Flatten and filter out PAD tokens using mask
                # We must move tensors to CPU for sklearn
                active_mask = (mask == 1) & (tags != tag_to_idx['<PAD>'])
                
                # Extract valid predictions and targets
                preds_clean = predicted[active_mask].cpu().numpy()
                targets_clean = tags[active_mask].cpu().numpy()
                
                all_preds.extend(preds_clean)
                all_targets.extend(targets_clean)
        
        val_acc = accuracy_score(all_targets, all_preds)
        val_macro_f1 = f1_score(all_targets, all_preds, average='macro')
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}, Dev Accuracy: {val_acc:.4f}, Macro F1: {val_macro_f1:.4f}')
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'word_to_idx': word_to_idx,
                'tag_to_idx': tag_to_idx,
                'args': vars(args)
            }, os.path.join(args.model_dir, 'best_model.pt'))
            print(f"Saved best model, acc: {val_acc:.4f}")
    
    print(f"Training Complete. Best Dev Acc: {best_accuracy:.4f}")

if __name__ == '__main__':
    train_bilstm()