---
license: other
library_name: pytorch
tags:
- tabular-classification
- tabpfn
- taco
- pot
---

# TabPFN-TACO

Built with PriorLabs-TabPFN.

This repository contains two classifier checkpoints:

- `TabPFN-TACO-classifier.ckpt`: TACO classifier checkpoint with learned compression.
- `TabPFN-POT-classifier.ckpt`: POT classifier checkpoint without learned compression.

Both checkpoints were trained from scratch. They do not redistribute or use
TabPFN or TabICL pretrained weights.


## Citation

```bibtex
@inproceedings{zabergja2026endtoend,
  title = {End-to-End Compression for Tabular Foundation Models},
  author = {Zab{\"e}rgja, Guri and Kamel, Rafiq and Kadra, Arlind and Frey, Christian M. M. and Grabocka, Josif},
  booktitle = {International Conference on Machine Learning},
  year = {2026},
}
```
