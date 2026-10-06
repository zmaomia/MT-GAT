# MT-GAT: Multi-task Graph Attention Network for Forest Soil Mapping

[![Preprint](https://img.shields.io/badge/Preprint-SSRN%207287455-blue)](https://ssrn.com/abstract=7287455)

PyTorch Geometric implementation of the paper:

> **Multi-task Graph Attention Network for Mapping Forest Soil Organic Layer Thickness and Soil Organic Matter from Multi-source Remote Sensing Data**
> Zhu Mao, Omid Abdi, Ville Laamanen, Veli-Pekka Kivinen, Jori Uusitalo
> *Preprint available at SSRN:* [https://ssrn.com/abstract=7287455](https://ssrn.com/abstract=7287455)

## Overview

MT-GAT is a multi-task learning graph attention network that jointly predicts **soil organic layer thickness** and **soil organic matter content** in boreal forests from multi-source remote sensing data.

## Repository structure

This repository is built on top of [PyTorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric). The MT-GAT-specific code is:

- `train_MT_GAT.py` – model definition, training and evaluation  <!-- TODO: correct if the model is defined elsewhere -->

All other folders (`torch_geometric/`, `benchmark/`, `examples/`, ...) come from the PyG library.


## Training MT-GAT

To train the MT-GAT model, run:


```bash
python train_MT_GAT.py


## Citation

If you use this code in your research, please cite:

```bibtex
@article{mao7287455multi,
  title   = {Multi-task Graph Attention Network for Mapping Forest Soil Organic Layer Thickness and Soil Organic Matter from Multi-source Remote Sensing Data},
  author  = {Mao, Zhu and Abdi, Omid and Laamanen, Ville and Kivinen, Veli-Pekka and Uusitalo, Jori},
  journal = {Available at SSRN 7287455},
  year    = {2026},
  url     = {https://ssrn.com/abstract=7287455}
}
```

