# Tokenization Comparison: Finnish and Estonian

## Overview
This project explores the effect of different tokenization methods for two target languages:
- **High-resource:** Finnish  
- **Low-resource:** Estonian  

The goal is to analyze how different tokenizers influence sentence segmentation and representation for languages with varying data availability and morphological complexity.

---

## Tokenizers Compared
- **BPE (Byte Pair Encoding)**:  <https://github.com/github/rust-gems>
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
