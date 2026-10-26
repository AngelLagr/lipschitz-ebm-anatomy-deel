# Lipschitz EBM Anatomy

Study of the stabilization of Energy-Based Model (EBM) training via a
Lipschitz constraint on the energy network.

Codebase: [ebm-anatomy](https://github.com/point0bar1/ebm-anatomy) (Nijkamp, Hill,
Han, Zhu, Wu — *On the Anatomy of MCMC-Based Maximum Likelihood Learning of
Energy-Based Models*, AAAI 2020, [arXiv:1903.12370](https://arxiv.org/abs/1903.12370)).
Lipschitz constraints are implemented using the
[deel-torchlip](https://github.com/deel-ai/deel-torchlip) library, and we also
tested exactly orthogonal convolutions via
[orthogonium](https://github.com/deel-ai/orthogonium).

This repository is code of my final-year internship report
*Stabilizing Energy-Based Models via the Lipschitz Constraint* (Angel Lagrange,
PFE ANITI / DEEL team, IRT Saint Exupéry, supervisors Victor Boutin and Franck
Mamalet, 2026), whose full report is in this repository (only in a french version). This README is only a code-oriented summary of it.

---

## 1. Code

| File | Idea tested | Status |
|---|---|---|
| `train_data.py` | Baseline training with or without the Lipschitz constraint, with or without controlled resizing of detail resolution (`DIM_RESIZE`, `ORIG_SZ`): the image is downsampled then upscaled back using nearest-neighbor interpolation, which limits the high-frequency content the network needs to learn to represent | Functional |
| `train_data_dual_head.py` | Corresponds to the **DualHead** architecture from the PFE report (Appendix C): a shared 1-Lipschitz backbone, an HKR head, and an unconstrained EBM head, combined via sigmoid weighting `α(x) = σ(E_HKR(x))` based on estimated proximity to the distribution | According to the report, this significantly underperforms in practice (FID ≈ 80, versus 46 for the baseline on CIFAR-10). |
| `train_data_coarsetofine.py` | Cascaded multi-resolution training (coarse → fine), each stage initializing the next; relates to the Zhao et al. line of work cited in section 2.3.3 of the report as motivation for multi-scale learning via downsampling — note that in this code we only reduce the detail resolution, not the image resolution | To be explored further; the theories and models behave as expected, but no convincing results in terms of FID. |
| `train_data_dual_models_warmup` | Training with Lipschitz pre-initialization using 2 models (section 6.4): one model first learns while being Lipschitz-constrained in order to bring the buffer closer to the target distribution, then this buffer is reused to train an unconstrained network on top of it (the buffer is not reset with noise again, but with noise already refined by the first model) | Functional. Very good performance shown in the report; however, the energy is no longer proportional to a log-probability (the EBM properties are lost) since there are 2 models, adding complexity |
| `train_data_freqscheduled.py` | Persistent buffer initialization noise filtered by frequency band (FFT), with a curriculum going from purely low-frequency noise to a uniform low/mid/high-frequency mix over the course of training | Functional; the report (Appendix E) empirically confirms the stability gain brought by multi-frequency initialization noise compared to uniform high-frequency noise, and notes that a progressive introduction of frequencies gives the best results but complicates the pipeline |
| `train_data_maco.py` | Phase-based variant (MACO) instead of running Langevin directly on the image | **Not functional**, flagged as such in the code |

`nets.py` also contains the **Semi-Lipschitz** architecture from the report
(section 4.5): only the first layers (`SpectralConv2d`/`SpectralLinear` +
`GroupSort2`) are constrained to be 1-Lipschitz; very good results are
obtained when starting from FFT-filtered noise. It also contains a `Halflip`
network whose entire feature extractor is Lipschitz-constrained — results with
this model are rather poor. It also includes fully Lipschitz networks such as
`Wide_ResNet_LLip`.

Each training variant has an associated evaluation script
(`eval_checkpoint.py`, `eval_checkpoint_coarsetofine.py`,
`eval_checkpoint_dualhead.py`) that reloads a checkpoint, regenerates samples
via long-run MCMC, and recomputes the diagnostics.

---

## 2. Usage

### Training

In the chosen training script (`train_data*.py`), define at the top of the
file:

```python
EXP_DIR = './out_data/Expe/experiment_name/'
CONFIG_FILE = './config_locker/config_name.json'
```

then run the script. An `EXP_DIR` folder is created with the subfolders
`checkpoints/`, `shortrun/`, `longrun/`, `plots/`, `code/` (a copy of the code
and config used, for reproducibility).

### Configuration files (`config_locker/`)

Each JSON file fully defines an experiment: dataset (`mnist`, `cifar10`,
`flowers`), architecture (`net_type`, a key of `net_bank` in `train_data.py`),
initial `Lip_const`, optimizer, Langevin parameters (`epsilon`,
`num_shortrun_steps`, `shortrun_init`), and logging frequencies.

### Evaluation

`eval_checkpoint*.py` automatically retrieves the config file saved in
`EXP_DIR/code/`, reloads a given checkpoint, regenerates samples via long-run
MCMC, and recomputes the FID.
