# xyz-omics

Compression of large spatial-transcriptomics point clouds (BARseq) into a small
**representative measure** by minimizing a varifold distance.

A dataset of `N` points with gene-expression features (up to ~30M points) is
approximated by `M ≪ N` points `(X̂, P̂)`, optimized so that the varifold distance
`‖µ − µ̂‖` is minimal. Kernels are evaluated with [KeOps](https://www.kernel-operations.io/)
on GPU.

## Layout

```
src/config.py        all experiment parameters (single place to edit)
src/io/              .npz/.vtk loading, VTK/VTP export, plots
src/subsampling/     initialization: kmeans | random
src/losses/          varifold loss, isotropic & anisotropic Gaussian kernels
src/optim/           LBFGS.py, Adam.py (full-batch + mini-batch)
src/preprocessing/   spatial tiling (blocks, strips, overlaps, halos)
scripts/             entry points
```

## Usage

Set the run parameters in `src/config.py` (`RUN_ID`, `Input`, `M`, kernel type,
optimizer, …), then:

```bash
PYTHONPATH=. python scripts/barseq_varifold.py            # single global run
PYTHONPATH=. python scripts/barseq_varifold_tiled.py      # tiled (divide & conquer)
PYTHONPATH=. python scripts/barseq_varifold_minibatch.py  # Algorithm 3, mini-batch Adam
```

Inputs are read from `data/BARSeq/`, outputs (`.vtp`, loss curves, `results_*.pt`)
are written to `data/BARSeq/output/` and can be inspected in ParaView.

## Requirements

Python 3.11+, `torch`, `pykeops`, `numpy`, `scikit-learn`, `pyvista`, `matplotlib`.
A CUDA GPU is strongly recommended (falls back to CPU).
