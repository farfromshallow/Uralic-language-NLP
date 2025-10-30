# Tokenization Comparison: Finnish and Estonian

## Overview
This project explores the effect of different tokenization methods for two target languages:
- **High-resource:** Finnish  
- **Low-resource:** Estonian  

The goal is to analyze how different tokenizers influence sentence segmentation and representation for languages with varying data availability and morphological complexity.

---

## Tokenizers Compared
- **BPE (Byte Pair Encoding)**
- **OBPE (Overlap BPE)**:  <https://github.com/Vaidehi99/OBPE>
- **Unigram**

Each tokenizer will be applied to the same dataset to ensure fair comparison of token-level segmentation behavior and downstream translation performance.

---

## Pilot Study Dataset
The pilot experiments use the **ELRC-EMEA** corpus — a bilingual dataset derived from the European Medicines Agency’s PDF documents:

> European Medicines Agency (EMEA)  
> <https://www.ema.europa.eu>  
> February 2020 (EN–MT)

Dataset reference:  
**ELRC-2709-EMEA** (Public dataset distributed via [ELRC-SHARE](https://www.elrc-share.eu))

For each language, **1,000 randomly extracted sentence pairs** were selected.  
Only sentences **longer than 50 tokens** were included, as longer sequences tend to exhibit more complex grammatical structures.

---

## Repository Structure

```
├── pilot_data/
│   ├── elrc_medicine_domain
│     ├── ELRC-EMEA_1k.en-fi.en      # English–Finnish bilingual subset (1k pairs)
│     ├── ELRC-EMEA_1k.en-fi.fi
│     ├── ELRC-EMEA_1k.en-et.en      # English–Estonian bilingual subset (1k pairs)
│     ├── ELRC-EMEA_1k.en-et.et
│   ├── wmt24pp_news_domain
│     ├── wmt24pp.en-fi.en      # English–Finnish WMT24++ (998 rows)
│     ├── wmt24pp.en-fi.fi
│     ├── wmt24pp.en-et.en      # English–Estonian WMT24++ (998 rows)
│     ├── wmt24pp.en-et.et
│   ├── tokenized_data
│     ├── ELRC-EMEA_1k.bpe.et      # ELRC-EMEA datasets tokenization output
│     ├── ELRC-EMEA_1k.unigram.et
│     ├── ELRC-EMEA_1k.obpe.et
│     ├── ELRC-EMEA_1k.bpe.fi
│     ├── ELRC-EMEA_1k.unigram.fi
│     ├── ELRC-EMEA_1k.obpe.fi
│     ├── wmt24pp.bpe.et      # WMT24++ tokenization output
│     ├── wmt24pp.unigram.et
│     ├── wmt24pp.obpe.et
│     ├── wmt24pp.bpe.fi
│     ├── wmt24pp.unigram.fi
│     ├── wmt24pp.obpe.fi
├── downstrem_tasks_data/
│   ├── task1_keeper
│     ├── dataset_keeper
│   ├── task2_keeper
├── tokenizer_scripts/
│   ├── tokenizer_bpe_unigram.py
├── README.md
```
---

### Unigram and BPE Tokenization Script

A flexible tokenizer script `tokenizer_bpe_unigram.py` can train and apply both **BPE** and **Unigram** tokenizers. Users can specify input files, output files, model prefix, vocabulary size, and tokenizer type directly via command-line arguments. This allows seamless experimentation without modifying the script for different datasets.

**Usage Examples:**

```bash
# Unigram tokenization
python tokenizer_scripts/tokenizer_bpe_unigram.py \
  --tokenizer unigram \
  --input pilot_data/wmt24pp.en-et.et \
  --output pilot_data/wmt24pp.unigram.en-et \
  --model_prefix unigram_model \
  --vocab_size 5000

# BPE tokenization
python tokenizer_scripts/tokenizer_bpe_unigram.py \
  --tokenizer bpe \
  --input pilot_data/wmt24pp.en-fi.fi \
  --output pilot_data/wmt24pp.bpe.en-fi \
  --model_prefix bpe_model \
  --vocab_size 5000
```

## Next Steps
1. Train and apply each tokenizer to the pilot dataset.  
2. Compare vocabulary sizes, subword distributions, and token-per-sentence ratios.  
3. Evaluate the impact on translation quality metrics (e.g., BLEU, COMET).

---

## License

This repository contains publicly available datasets:

- **ELRC-2709-EMEA**: Bilingual corpus from the European Medicines Agency (EN–FI / EN–ET).  
- **WMT24++ (en-et / en-fi)**: Publicly available parallel datasets from the WMT24 News Translation task.

Ensure that you comply with the respective data-sharing and citation guidelines when using, redistributing, or publishing results based on these datasets.  

- For **ELRC-EMEA**, see [ELRC-SHARE](https://www.elrc-share.eu).  
- For **WMT24++**, refer to the [WMT24 dataset license and terms]([https://www.statmt.org/wmt24/translation-task.html](https://www2.statmt.org/wmt24/translation-task.html#_acknowledgements)).

No proprietary data from this repository may be redistributed beyond the usage permissions granted by the original dataset licenses.

---


