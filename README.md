# GNN-TF

Implementation accompanying **Graph Neural Networks with Transformer Fusion of Brain Connectivity Dynamics and Tabular Data for Forecasting Future Tobacco Use**, by Runzhi Zhou and Xi Luo.

## Installation

Use Python 3.9.7. Dependencies: PyTorch 2.0.0, PyTorch Geometric 2.3.0, NumPy 1.23.5, SciPy 1.10.0, Matplotlib 3.7.1, and NetworkX 2.8.4.

```bash
python -m pip install -r requirements.txt
```

For BERT, ViT, and GPT-2, install Transformers 4.28.1 and its dependencies:

```bash
python -m pip install -r requirements-transfer.txt
```

For CUDA, use PyTorch 2.0.0 with CUDA 11.8. `--device auto` selects CUDA when available; `--device cpu` and `--device cuda` select it explicitly.

## Training

Run five outer folds and five inner folds:

```bash
python train.py --evaluation nested-cv --outer-folds 5 --cv_size 5 --epochs 1000 --patience 50
```

The training defaults are `--nhid 128 --batch_size 32 --lr 0.0001 --seed 123 --ifadamw Adam --droplast False`. Each outer fold averages the probabilities of its inner models. Outputs include fold ROC AUC, mean and standard deviation across outer folds, and out-of-fold predictions.

Select a pretrained encoder:

```bash
python train.py --model bert --evaluation nested-cv
python train.py --model vit --evaluation nested-cv
python train.py --model gpt2 --evaluation nested-cv
```

Pretrained weights are downloaded on first use: `textattack/bert-base-uncased-yelp-polarity`, `google/vit-base-patch16-224`, and `microsoft/DialogRPT-updown`, respectively.

For a short run:

```bash
python train.py --evaluation nested-cv --outer-folds 2 --cv_size 2 --epochs 2 --num-subjects 24 --num-nodes 12 --nhid 32
```

`--evaluation holdout` (the default) uses one stratified training/validation/test split. Use `--output-dir` to choose a new directory for another run; existing run files are preserved. Run `python train.py --help` for all options.

## Data

Without `--data`, the program generates synthetic ROI signals and covariates. NCANDA data are not distributed. To train on your own data:

```bash
python train.py --data path/to/data.npz --evaluation nested-cv
```

The NPZ file contains `adjacency` `[subjects, windows, nodes, nodes]`, `node_features` `[nodes, features]`, `tabular` `[subjects, covariates]`, binary `labels` `[subjects]`, and `system_ids` `[nodes]` for plotting. Adjacency matrices must be symmetric with zero diagonal and positive weighted degrees after adding self-loops.

## Visualization

```bash
python visualize.py --run-dir outputs/scratch
```

This plots token attention and the top 25 undirected connections using edge-only PyG GNNExplainer. The defaults are 32 subjects, 500 steps, and learning rate 0.001. Directional masks are averaged before aggregation across subjects and windows.

Connection plots use the first two node-feature columns as x/y coordinates.

For a shorter run, add `--subjects 2 --explainer-steps 5`. Use `--run-dir outputs/bert`, `outputs/vit`, or `outputs/gpt2` for the corresponding encoder.

## Outputs

Each model writes to `outputs/<model>/`:

```text
outputs/scratch/
├── data.npz
├── model.pt
├── metrics.json
├── predictions.csv
└── visualizations/
    ├── token_attention.png
    ├── top_undirected_edges.csv
    └── top_undirected_edges.png
```

For nested CV, `model.pt` stores a model from the first outer fold for visualization; the CV metrics use all outer folds. Frozen pretrained weights are loaded from their source checkpoint.

## Citation and license

Please cite the accompanying paper. Citation metadata are provided in [CITATION.cff](CITATION.cff). The code is distributed under [GPL-3.0](LICENSE).
