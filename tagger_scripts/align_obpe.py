"""
OBPE Subword-Tag Alignment Script

Aligns OBPE subword tokens with gold POS tags based on </w> word boundary markers.

Input format:
  - subwords file: "H ä n</w> m e n i</w> k au pp a an</w>"
  - tags file: "PRON VERB NOUN"

Output format (CoNLL-U):
  1    H         _    PRON    _    _    _    _    _    _
  2    ä         _    PRON    _    _    _    _    _    _
  3    n</w>     _    PRON    _    _    _    _    _    _
  4    m         _    VERB    _    _    _    _    _    _
  5    e         _    VERB    _    _    _    _    _    _
  6    n         _    VERB    _    _    _    _    _    _
  7    i</w>     _    VERB    _    _    _    _    _    _
  8    k         _    NOUN    _    _    _    _    _    _
  ...
"""

import argparse
from pathlib import Path


class OBPEAligner:
    """Align OBPE subwords with gold POS tags."""
    
    def __init__(self, eow_marker='</w>'):
        self.eow_marker = eow_marker
    
    def align_sentence(self, subwords, tags):
        """
        Align subwords with tags based on </w> markers.
        
        Important: Each token ending with </w> (including punctuation) 
        gets its own tag from the tags list.
        
        Args:
            subwords: List of OBPE subword tokens
            tags: List of gold POS tags (one per word, including punctuation)
        
        Returns:
            List of (subword, tag) tuples
        
        Example:
            subwords: ['H', 'ä', 'n</w>', 'm', 'e', 'n', 'i</w>', '.</w>']
            tags: ['PRON', 'VERB', 'PUNCT']
            
            Result:
            [('H', 'PRON'), ('ä', 'PRON'), ('n</w>', 'PRON'),
             ('m', 'VERB'), ('e', 'VERB'), ('n', 'VERB'), ('i</w>', 'VERB'),
             ('.</w>', 'PUNCT')]
        """
        aligned = []
        tag_idx = 0
        
        if len(tags) == 0:
            if len(subwords) > 0:  # Only warn if there are actual subwords
                print(f"Warning: No tags provided for subwords: {' '.join(subwords[:10])}...")
            return aligned
        
        for subword in subwords:
            # Check if we have enough tags
            if tag_idx >= len(tags):
                print(f"Error: Ran out of tags at subword '{subword}'")
                print(f"  Remaining subwords: {' '.join(subwords[subwords.index(subword):])}")
                print(f"  Expected {len(tags)} tags but need more")
                # Fill remaining with fallback
                aligned.append((subword, 'X'))
                continue
            
            # Assign current tag to subword
            current_tag = tags[tag_idx]
            aligned.append((subword, current_tag))
            
            # Check if this subword ends a word (contains </w>)
            # This applies to ALL tokens with </w>, including punctuation
            if self.eow_marker in subword:
                tag_idx +=  1  # Move to next tag
        
        # Verify we used all tags
        if tag_idx != len(tags):
            print(f"Warning: Alignment mismatch. Used {tag_idx} tags, expected {len(tags)}")
            print(f"  Subwords: {' '.join(subwords[:20])}...")
            print(f"  Tags: {' '.join(tags)}")
            print(f"  Words (by {self.eow_marker}): {sum(1 for sw in subwords if self.eow_marker in sw)}")
            print(f"  Unused tags: {' '.join(tags[tag_idx:])}")
        
        return aligned
    
    def align_file(self, subwords_file, tags_file, output_file, output_format='conllu'):
        """
        Align entire files.
        
        Args:
            subwords_file: Path to subwords file (one sentence per line)
            tags_file: Path to tags file (one sentence per line)
            output_file: Path to output file
            output_format: 'conllu' or 'simple'
        """
        print(f"Aligning {Path(subwords_file).name}...")
        print(f"  Subwords: {subwords_file}")
        print(f"  Tags: {tags_file}")
        print(f"  Output: {output_file}")
        
        total_sentences = 0
        total_subwords = 0
        total_words = 0
        misalignment_count = 0
        
        with open(subwords_file, 'r', encoding='utf-8') as sf, \
             open(tags_file, 'r', encoding='utf-8') as tf, \
             open(output_file, 'w', encoding='utf-8') as of:
            
            for line_num, (subword_line, tag_line) in enumerate(zip(sf, tf), 1):
                subwords = subword_line.strip().split()
                tags = tag_line.strip().split()
                
                # Handle empty lines
                if not subwords and not tags:
                    of.write('\n')
                    continue
                elif not subwords and tags:
                    print(f"Warning (line {line_num}): Empty subwords but tags present: {' '.join(tags)}")
                    of.write('\n')
                    continue
                elif subwords and not tags:
                    print(f"Warning (line {line_num}): Subwords present but no tags: {' '.join(subwords[:10])}...")
                    # Write subwords without tags as fallback
                    for idx, subword in enumerate(subwords, 1):
                        if output_format == 'conllu':
                            of.write(f"{idx}\t{subword}\t_\tX\t_\t_\t_\t_\t_\t_\n")
                        else:
                            of.write(f"{subword} X\n")
                    of.write('\n')
                    total_sentences += 1
                    total_subwords += len(subwords)
                    continue
                
                # Count words (by </w> markers)
                num_words = sum(1 for sw in subwords if self.eow_marker in sw)
                
                if num_words != len(tags):
                    misalignment_count += 1
                    if misalignment_count <= 10:  # Show first 10 warnings
                        print(f"\n  Warning (line {line_num}): Word count mismatch")
                        print(f"    Words (by {self.eow_marker}): {num_words}")
                        print(f"    Tags: {len(tags)}")
                        print(f"    Subwords: {' '.join(subwords[:20])}...")
                        print(f"    Tags: {' '.join(tags)}")
                        if num_words > len(tags):
                            print(f"    Missing {num_words - len(tags)} tags")
                        else:
                            print(f"    Extra {len(tags) - num_words} tags")
                
                # Align
                aligned = self.align_sentence(subwords, tags)
                
                # Write output
                if output_format == 'conllu':
                    self._write_conllu(of, aligned, line_num)
                else:
                    self._write_simple(of, aligned)
                
                total_sentences += 1
                total_subwords += len(subwords)
                total_words += len(tags)
        
        # Summary
        print(f"\n✓ Alignment complete!")
        print(f"  Sentences: {total_sentences:,}")
        print(f"  Total words: {total_words:,}")
        print(f"  Total subwords: {total_subwords:,}")
        if total_words > 0:
            print(f"  Compression ratio: {total_subwords / total_words:.2f}x")
        if misalignment_count > 0:
            print(f"  ⚠️  Misalignments: {misalignment_count}/{total_sentences} sentences")
        print(f"  Saved to: {output_file}")
    
    def _write_conllu(self, file, aligned, sent_id):
        """Write aligned data in CoNLL-U format."""
        file.write(f"# sent_id = {sent_id}\n")
        
        # Reconstruct original text (removing </w> markers for readability)
        text_tokens = []
        for sw, _ in aligned:
            clean_sw = sw.replace(self.eow_marker, '')
            text_tokens.append(clean_sw)
        text = ''.join(text_tokens)  # OBPE doesn't use spaces within words
        file.write(f"# text = {text}\n")
        
        # Write token lines
        for idx, (subword, tag) in enumerate(aligned, 1):
            # CoNLL-U format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            file.write(f"{idx}\t{subword}\t\t{tag}\n")
        
        file.write('\n')
    
    def _write_simple(self, file, aligned):
        """Write aligned data in simple format (token TAG)."""
        for subword, tag in aligned:
            file.write(f"{subword} {tag}\n")
        file.write('\n')


def validate_alignment(subwords_file, tags_file, eow_marker='</w>'):
    """
    Validate that subwords and tags can be aligned.
    Checks that number of </w> markers matches number of tags.
    """
    print(f"\nValidating alignment...")
    print(f"  Subwords file: {subwords_file}")
    print(f"  Tags file: {tags_file}")
    
    mismatches = []
    total_lines = 0
    empty_lines = 0
    
    with open(subwords_file, 'r', encoding='utf-8') as sf, \
         open(tags_file, 'r', encoding='utf-8') as tf:
        
        for line_num, (subword_line, tag_line) in enumerate(zip(sf, tf), 1):
            subwords = subword_line.strip().split()
            tags = tag_line.strip().split()
            
            # Skip empty lines
            if not subwords and not tags:
                empty_lines += 1
                continue
            
            total_lines += 1
            
            num_words = sum(1 for sw in subwords if eow_marker in sw)
            num_tags = len(tags)
            
            if num_words != num_tags:
                mismatches.append({
                    'line': line_num,
                    'words': num_words,
                    'tags': num_tags,
                    'subwords': subwords[:15],
                    'tag_list': tags
                })
    
    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} misalignments out of {total_lines} sentences")
        print(f"  Empty lines: {empty_lines}")
        print(f"\nFirst 10 misalignments:")
        for i, mismatch in enumerate(mismatches[:10], 1):
            print(f"\n  {i}. Line {mismatch['line']}:")
            print(f"     Words (by {eow_marker}): {mismatch['words']}")
            print(f"     Tags: {mismatch['tags']}")
            print(f"     Subwords: {' '.join(mismatch['subwords'])}{'...' if len(mismatch['subwords']) >= 15 else ''}")
            print(f"     Tags: {' '.join(mismatch['tag_list'])}")
            if mismatch['words'] > mismatch['tags']:
                print(f"     → Missing {mismatch['words'] - mismatch['tags']} tags")
            else:
                print(f"     → Extra {mismatch['tags'] - mismatch['words']} tags")
        
        if len(mismatches) > 10:
            print(f"\n  ... and {len(mismatches) - 10} more")
        
        return False
    else:
        print(f"✓ All {total_lines} sentences can be aligned correctly!")
        print(f"  Empty lines: {empty_lines}")
        return True


def inspect_alignment(aligned_file, num_sentences=5):
    """
    Inspect aligned file to verify correctness.
    Shows first few sentences with word boundaries highlighted.
    """
    print(f"\n{'='*80}")
    print(f"INSPECTING ALIGNED FILE: {aligned_file}")
    print(f"{'='*80}\n")
    
    with open(aligned_file, 'r', encoding='utf-8') as f:
        sent_count = 0
        current_sent = []
        
        for line in f:
            line = line.strip()
            
            if line.startswith('#'):
                if current_sent and sent_count < num_sentences:
                    print(line)
                continue
            
            if not line:
                if current_sent and sent_count < num_sentences:
                    # Group by word (using </w> markers)
                    words = []
                    current_word = []
                    
                    for subword, tag in current_sent:
                        current_word.append(f"{subword}({tag})")
                        if '</w>' in subword:
                            # This </w> marks end of word (including punctuation)
                            words.append('+'.join(current_word))
                            current_word = []
                    
                    if current_word:  # Leftover subwords (shouldn't happen)
                        words.append('+'.join(current_word))
                    
                    print(f"\nWord groupings (each ends with </w>):")
                    for i, word in enumerate(words, 1):
                        # Highlight punctuation words
                        if any(p in word for p in ['.</w>', ',</w>', '!</w>', '?</w>', ':</w>', ';</w>', '-</w>', '(</w>', ')</w>']):
                            print(f"  Word {i}: {word} ← PUNCTUATION")
                        else:
                            print(f"  Word {i}: {word}")
                    
                    print(f"\n{'-'*80}\n")
                    sent_count += 1
                
                current_sent = []
                
                if sent_count >= num_sentences:
                    break
                continue
            
            # Parse token line
            parts = line.split('\t')
            if len(parts) >= 4:
                subword = parts[1]
                tag = parts[3]
                current_sent.append((subword, tag))


def main():
    parser = argparse.ArgumentParser(
        description='Align OBPE subwords with gold POS tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Align OBPE subwords with tags:
   python align_obpe.py \\
     --subwords outputs/fi_train_obpe.subwords \\
     --tags outputs/fi_train.tags \\
     --output data/fi_train_obpe_aligned.conllu

2. Align with custom end-of-word marker:
   python align_obpe.py \\
     --subwords outputs/fi_train_obpe.subwords \\
     --tags outputs/fi_train.tags \\
     --output data/fi_train_obpe_aligned.conllu \\
     --eow_marker "</w>"

3. Validate alignment before creating output:
   python align_obpe.py \\
     --subwords outputs/fi_train_obpe.subwords \\
     --tags outputs/fi_train.tags \\
     --output data/fi_train_obpe_aligned.conllu \\
     --validate

4. Batch align multiple files:
   python align_obpe.py \\
     --batch \\
     --subwords_dir outputs \\
     --tags_dir outputs \\
     --output_dir data \\
     --pattern obpe

Example subwords format (OBPE with </w> markers):
  H ä n</w> m e n i</w> k au pp a an</w>

Example tags format (one per word):
  PRON VERB NOUN

Output format (CoNLL-U):
  # sent_id = 1
  # text = Hänmenikauuppaan
  1    H         _    PRON    _    _    _    _    _    _
  2    ä         _    PRON    _    _    _    _    _    _
  3    n</w>     _    PRON    _    _    _    _    _    _
  4    m         _    VERB    _    _    _    _    _    _
  5    e         _    VERB    _    _    _    _    _    _
  6    n         _    VERB    _    _    _    _    _    _
  7    i</w>     _    VERB    _    _    _    _    _    _
  8    k         _    NOUN    _    _    _    _    _    _
  ...
        """
    )
    
    # Single file mode
    parser.add_argument('--subwords', help='Path to subwords file')
    parser.add_argument('--tags', help='Path to tags file')
    parser.add_argument('--output', help='Path to output aligned file')
    
    # Batch mode
    parser.add_argument('--batch', action='store_true', help='Batch mode')
    parser.add_argument('--subwords_dir', help='Directory with subwords files')
    parser.add_argument('--tags_dir', help='Directory with tags files')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--pattern', default='obpe', help='Pattern to match files (default: obpe)')
    
    # Options
    parser.add_argument('--eow_marker', default='</w>', help='End-of-word marker (default: </w>)')
    parser.add_argument('--format', choices=['conllu', 'simple'], default='conllu',
                       help='Output format (default: conllu)')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate alignment before creating output')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect output file after creation')
    parser.add_argument('--inspect_sentences', type=int, default=5,
                       help='Number of sentences to inspect (default: 5)')
    parser.add_argument('--force', action='store_true',
                       help='Force alignment even if validation fails')
    
    args = parser.parse_args()
    
    # Initialize aligner
    aligner = OBPEAligner(eow_marker=args.eow_marker)
    
    if args.batch:
        # Batch mode
        if not all([args.subwords_dir, args.tags_dir, args.output_dir]):
            parser.error("--subwords_dir, --tags_dir, and --output_dir required for batch mode")
        
        subwords_dir = Path(args.subwords_dir)
        tags_dir = Path(args.tags_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all matching files
        pattern = f"*{args.pattern}*.subwords"
        subwords_files = list(subwords_dir.glob(pattern))
        
        if not subwords_files:
            print(f"No files matching pattern '{pattern}' found in {subwords_dir}")
            return
        
        print(f"Found {len(subwords_files)} files matching '{pattern}'")
        print()
        
        for subwords_file in subwords_files:
            # Derive tags filename
            base_name = subwords_file.stem.replace(f'_{args.pattern}', '').replace('.subwords', '')
            tags_file = tags_dir / f"{base_name}.tags"
            output_file = output_dir / f"{base_name}_{args.pattern}_aligned.conllu"
            
            if not tags_file.exists():
                print(f"Warning: Tags file not found: {tags_file}")
                continue
            
            print(f"\nProcessing: {subwords_file.name}")
            
            # Validate if requested
            if args.validate:
                valid = validate_alignment(subwords_file, tags_file, args.eow_marker)
                if not valid and not args.force:
                    print(f"  ⚠️  Skipping due to validation errors (use --force to override)")
                    continue
                elif not valid and args.force:
                    print(f"  ⚠️  Validation failed but continuing due to --force")
            
            # Align
            aligner.align_file(subwords_file, tags_file, output_file, args.format)
            
            # Inspect if requested
            if args.inspect:
                inspect_alignment(output_file, args.inspect_sentences)
    
    else:
        # Single file mode
        if not all([args.subwords, args.tags, args.output]):
            parser.error("--subwords, --tags, and --output required")
        
        # Validate if requested
        if args.validate:
            valid = validate_alignment(args.subwords, args.tags, args.eow_marker)
            if not valid and not args.force:
                print("\n⚠️  Validation found issues. Proceed anyway? (y/n)")
                response = input().strip().lower()
                if response != 'y':
                    print("Aborted.")
                    return
            elif not valid and args.force:
                print("\n⚠️  Validation failed but continuing due to --force")
        
        # Align
        aligner.align_file(args.subwords, args.tags, args.output, args.format)
        
        # Inspect if requested
        if args.inspect:
            inspect_alignment(args.output, args.inspect_sentences)


if __name__ == '__main__':
    main()