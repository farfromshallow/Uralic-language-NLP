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
│     ├── wmt24pp.en-fi.en      # English–Finnish WMT24 ++ (1k pairs)
│     ├── wmt24pp.en-fi.fi
│     ├── wmt24pp.en-et.en      # English–Estonian WMT24 ++ (1k pairs)
│     ├── wmt24pp.en-et.et
├── downstrem_tasks_data/
│   ├── task1_keeper
│     ├── dataset_keeper
│   ├── task2_keeper
├── tokenizer_scripts/
│   ├── tokenize_bpe.py
│   ├── tokenize_obpe.py
│   ├── tokenize_unigram.py
├── README.md
```
---

## Next Steps
1. Train and apply each tokenizer to the pilot dataset.  
2. Compare vocabulary sizes, subword distributions, and token-per-sentence ratios.  
3. Evaluate the impact on translation quality metrics (e.g., BLEU, COMET).

---

## License
This repository uses publicly available ELRC data (ELRC-2709-EMEA).  
Ensure compliance with ELRC’s data-sharing and citation guidelines when using or redistributing the data.

---


