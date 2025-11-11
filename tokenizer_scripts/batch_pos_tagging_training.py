"""
Batch training script for all POS tagging models
Trains BiLSTM and Flair on BPE, Unigram, and OBPE tokenizations
Evaluates zero-shot transfer from Finnish to Estonian
"""

import subprocess
import json
import argparse
from pathlib import Path
import time
from datetime import datetime


class BatchTrainer:
    """Batch trainer for all tokenization variants."""
    
    def __init__(self, data_dir, models_dir, results_dir):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.tokenizers = ['bpe', 'unigram', 'obpe']
        self.results = {}
    
    def run_command(self, cmd, description):
        """Run a command and log output."""
        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start_time
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            print(f"\n✓ Completed in {elapsed:.1f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n✗ Failed after {elapsed:.1f} seconds")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False
    
    def train_bilstm_models(self, epochs=30):
        """Train all BiLSTM models."""
        print("\n" + "="*80)
        print("TRAINING BiLSTM MODELS")
        print("="*80)
        
        for tok in self.tokenizers:
            train_file = self.data_dir / f"fi_train_{tok}_aligned.conllu"
            dev_file = self.data_dir / f"fi_dev_{tok}_aligned.conllu"
            model_save = self.models_dir / f"bilstm_{tok}.pt"
            
            if not train_file.exists():
                print(f"\n⚠️  Warning: {train_file} not found, skipping {tok}")
                continue
            
            cmd = [
                'python', 'train_bilstm.py',
                '--mode', 'train',
                '--train_conllu', str(train_file),
                '--dev_conllu', str(dev_file),
                '--model_save', str(model_save),
                '--epochs', str(epochs),
                '--batch_size', '32',
                '--learning_rate', '0.001'
            ]
            
            success = self.run_command(
                cmd,
                f"Training BiLSTM with {tok.upper()} tokenization"
            )
            
            if success:
                self.results[f'bilstm_{tok}'] = {'status': 'trained'}
    
    def train_flair_models(self, epochs=30, embedding='fasttext'):
        """Train all Flair models."""
        print("\n" + "="*80)
        print("TRAINING FLAIR MODELS")
        print("="*80)
        
        for tok in self.tokenizers:
            train_file = self.data_dir / f"fi_train_{tok}_aligned.conllu"
            dev_file = self.data_dir / f"fi_dev_{tok}_aligned.conllu"
            test_file = self.data_dir / f"et_test_{tok}_aligned.conllu"
            model_save_dir = self.models_dir / f"flair_{tok}"
            
            if not train_file.exists():
                print(f"\n⚠️  Warning: {train_file} not found, skipping {tok}")
                continue
            
            cmd = [
                'python', 'train_flair.py',
                '--mode', 'train',
                '--train_conllu', str(train_file),
                '--dev_conllu', str(dev_file),
                '--test_conllu', str(test_file),
                '--model_save_dir', str(model_save_dir),
                '--embedding', embedding,
                '--epochs', str(epochs),
                '--batch_size', '32'
            ]
            
            success = self.run_command(
                cmd,
                f"Training Flair with {tok.upper()} tokenization"
            )
            
            if success:
                self.results[f'flair_{tok}'] = {'status': 'trained'}
    
    def evaluate_bilstm_models(self):
        """Evaluate all BiLSTM models on Estonian test set."""
        print("\n" + "="*80)
        print("EVALUATING BiLSTM MODELS (Zero-Shot)")
        print("="*80)
        
        for tok in self.tokenizers:
            model_path = self.models_dir / f"bilstm_{tok}.pt"
            test_file = self.data_dir / f"et_test_{tok}_aligned.conllu"
            
            if not model_path.exists():
                print(f"\n⚠️  Warning: {model_path} not found, skipping {tok}")
                continue
            
            if not test_file.exists():
                print(f"\n⚠️  Warning: {test_file} not found, skipping {tok}")
                continue
            
            cmd = [
                'python', 'train_bilstm.py',
                '--mode', 'test',
                '--model_path', str(model_path),
                '--test_conllu', str(test_file)
            ]
            
            # Capture output to parse results
            print(f"\n{'='*80}")
            print(f"Evaluating BiLSTM {tok.upper()} on Estonian (zero-shot)")
            print(f"{'='*80}")
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                
                # Parse accuracy and F1 from output
                for line in result.stdout.split('\n'):
                    if 'Accuracy:' in line:
                        acc = float(line.split(':')[1].strip())
                        self.results[f'bilstm_{tok}']['accuracy'] = acc
                    if 'F1 Score:' in line:
                        f1 = float(line.split(':')[1].strip())
                        self.results[f'bilstm_{tok}']['f1'] = f1
                
            except subprocess.CalledProcessError as e:
                print(f"✗ Evaluation failed")
                print(e.stderr)
    
    def evaluate_flair_models(self):
        """Evaluate all Flair models on Estonian test set."""
        print("\n" + "="*80)
        print("EVALUATING FLAIR MODELS (Zero-Shot)")
        print("="*80)
        
        for tok in self.tokenizers:
            model_path = self.models_dir / f"flair_{tok}" / "final-model.pt"
            test_file = Path('flair_data') / f"et_test_{tok}.txt"
            
            if not model_path.exists():
                print(f"\n⚠️  Warning: {model_path} not found, skipping {tok}")
                continue
            
            if not test_file.exists():
                # Try to find the test file in data_dir
                test_file = self.data_dir / f"et_test_{tok}_aligned.conllu"
                if not test_file.exists():
                    print(f"\n⚠️  Warning: Test file not found, skipping {tok}")
                    continue
                
                # Convert to Flair format
                flair_test = Path('flair_data') / f"et_test_{tok}.txt"
                flair_test.parent.mkdir(exist_ok=True)
                self._convert_to_flair_format(test_file, flair_test)
                test_file = flair_test
            
            cmd = [
                'python', 'train_flair.py',
                '--mode', 'test',
                '--model_path', str(model_path),
                '--test_file', str(test_file)
            ]
            
            print(f"\n{'='*80}")
            print(f"Evaluating Flair {tok.upper()} on Estonian (zero-shot)")
            print(f"{'='*80}")
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                
                # Parse results
                for line in result.stdout.split('\n'):
                    if 'Accuracy:' in line:
                        acc = float(line.split(':')[1].strip())
                        self.results[f'flair_{tok}']['accuracy'] = acc
                    if 'F1 Score:' in line:
                        f1 = float(line.split(':')[1].strip())
                        self.results[f'flair_{tok}']['f1'] = f1
                
            except subprocess.CalledProcessError as e:
                print(f"✗ Evaluation failed")
                print(e.stderr)
    
    def _convert_to_flair_format(self, conllu_file, flair_file):
        """Quick conversion to Flair format."""
        with open(conllu_file, 'r') as f_in, open(flair_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if not line:
                    f_out.write('\n')
                    continue
                parts = line.split('\t')
                if len(parts) >= 4 and '-' not in parts[0]:
                    f_out.write(f"{parts[1]} {parts[3]}\n")
    
    def save_results(self):
        """Save all results to JSON."""
        results_file = self.results_dir / 'all_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        
        # Print summary table
        self.print_summary()
    
    def print_summary(self):
        """Print results summary table."""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print()
        
        # Table header
        print(f"{'Model':<20} {'BPE':<15} {'Unigram':<15} {'OBPE':<15}")
        print("-"*80)
        
        # BiLSTM results
        bilstm_results = []
        for tok in self.tokenizers:
            key = f'bilstm_{tok}'
            if key in self.results and 'accuracy' in self.results[key]:
                acc = self.results[key]['accuracy']
                f1 = self.results[key].get('f1', 0)
                bilstm_results.append(f"{acc:.4f}/{f1:.4f}")
            else:
                bilstm_results.append("N/A")
        
        print(f"{'BiLSTM (Acc/F1)':<20} {bilstm_results[0]:<15} {bilstm_results[1]:<15} {bilstm_results[2]:<15}")
        
        # Flair results
        flair_results = []
        for tok in self.tokenizers:
            key = f'flair_{tok}'
            if key in self.results and 'accuracy' in self.results[key]:
                acc = self.results[key]['accuracy']
                f1 = self.results[key].get('f1', 0)
                flair_results.append(f"{acc:.4f}/{f1:.4f}")
            else:
                flair_results.append("N/A")
        
        print(f"{'Flair (Acc/F1)':<20} {flair_results[0]:<15} {flair_results[1]:<15} {flair_results[2]:<15}")
        
        print()
        print("="*80)
        
        # Find best
        best_acc = 0
        best_model = None
        
        for key, result in self.results.items():
            if 'accuracy' in result and result['accuracy'] > best_acc:
                best_acc = result['accuracy']
                best_model = key
        
        if best_model:
            print(f"BEST MODEL: {best_model} (Accuracy: {best_acc:.4f})")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Batch training for all POS tagging models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. Train all models (BiLSTM + Flair, all tokenizers):
   python batch_train.py \\
     --mode all \\
     --data_dir data \\
     --models_dir models \\
     --results_dir results \\
     --epochs 30

2. Train only BiLSTM models:
   python batch_train.py \\
     --mode bilstm \\
     --data_dir data \\
     --models_dir models \\
     --epochs 30

3. Evaluate only (skip training):
   python batch_train.py \\
     --mode eval \\
     --data_dir data \\
     --models_dir models \\
     --results_dir results
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['all', 'bilstm', 'flair', 'eval'],
                       default='all',
                       help='Training mode')
    parser.add_argument('--data_dir', default='data',
                       help='Directory with aligned CoNLL-U files')
    parser.add_argument('--models_dir', default='models',
                       help='Directory to save models')
    parser.add_argument('--results_dir', default='results',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--flair_embedding', default='fasttext',
                       choices=['fasttext', 'flair', 'transformer', 'mixed'],
                       help='Embedding type for Flair')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BatchTrainer(args.data_dir, args.models_dir, args.results_dir)
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("="*80)
    print("BATCH POS TAGGER TRAINING")
    print("="*80)
    print(f"Started: {timestamp}")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Models directory: {args.models_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Epochs: {args.epochs}")
    print("="*80)
    
    # Training phase
    if args.mode in ['all', 'bilstm']:
        trainer.train_bilstm_models(epochs=args.epochs)
    
    if args.mode in ['all', 'flair']:
        trainer.train_flair_models(epochs=args.epochs, embedding=args.flair_embedding)
    
    # Evaluation phase
    if args.mode in ['all', 'eval']:
        trainer.evaluate_bilstm_models()
        trainer.evaluate_flair_models()
    
    # Save results
    trainer.save_results()
    
    # Print timing
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print(f"\n{'='*80}")
    print(f"TOTAL TIME: {hours}h {minutes}m")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()