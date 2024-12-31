<div align="center">
<h1> 🐍GraphSSM (Graph State Space Models)</h1>
<h3>State Space Models on Temporal Graphs: A First-Principles Study</h3>

Jintang Li<sup>1</sup>, Ruofan Wu<sup>2</sup>, Boqun Ma<sup>3</sup>, Xinzhou Jin<sup>1</sup>, Liang Chen<sup>1</sup>, Zibin Zheng<sup>1</sup>

<sup>1</sup>Sun Yat-sen University, <sup>2</sup>Coupang, <sup>3</sup>Shanghai Jiao Tong University
 

[![arXiv](https://img.shields.io/badge/arXiv-2406.00943-b31b1b.svg)](https://arxiv.org/abs/2406.00943)

</div>

## Environments
> [!NOTE]
> Higher versions should be also compatible.

```
conda create -n GraphSSM python=3.10
conda activate GraphSSM

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install -r requirements.txt
```

## Reproduction

```
bash scripts/s4.sh
bash scripts/s5.sh
bash scripts/s6.sh
```

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation

```bib
@inproceedings{graphssm,
  author       = {Jintang Li and
                  Ruofan Wu and
                  Xinzhou Jin and
                  Boqun Ma and
                  Liang Chen and
                  Zibin Zheng},
  title        = {State Space Models on Temporal Graphs: {A} First-Principles Study},
  booktitle    = {NeurIPS},
  year         = {2024}
}
```

