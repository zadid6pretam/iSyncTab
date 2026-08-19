# iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony (ECCV 2026)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-Multimodal%20Image--Tabular%20Learning-orange)
![Model](https://img.shields.io/badge/Model-iSyncTab-blueviolet)
![Method](https://img.shields.io/badge/Method-Neural%20Synchrony--guided%20Paired%20Feature%20Sequencing-informational)
![Algorithm](https://img.shields.io/badge/Algorithm-NS--PFS-purple)
![Architecture](https://img.shields.io/badge/Architecture-NS--PFS%20%2B%20OMT%20-critical)
![Backbone](https://img.shields.io/badge/Backbone-Linformer-teal)
![Formulation](https://img.shields.io/badge/Formulation-Column%20Permutation%20Problem-9cf)
![Domain](https://img.shields.io/badge/Domain-Image%20%2B%20Tabular%20Data-9cf)
![Extension](https://img.shields.io/badge/Extension-Audio%20%2B%20Video-9cf)
[![Conference](https://img.shields.io/badge/Conference-ECCV%202026-blue)](https://eccv.ecva.net/)
![Status](https://img.shields.io/badge/Status-Accepted-brightgreen)
[![Paper](https://img.shields.io/badge/Paper-Published-success)](https://doi.org/10.1007/978-3-032-37035-8)

<p align="center">
  <img src="iSyncTab_Architecture.png" alt="iSyncTab Architecture" width="1000">
</p>

iSyncTab is a neural synchrony-guided feature sequencing framework for **multimodal image-tabular learning**. It introduces **Neural Synchrony-guided Paired Feature Sequencing (NS-PFS)** to derive a coherent cross-modal feature order from image and tabular representations, framing feature sequencing through the lens of the **Column Permutation Problem (CPP)**. Rather than treating fused multimodal features as an arbitrarily ordered representation, iSyncTab clusters modality-specific features and aligns image and tabular feature clusters using a synchrony matrix that combines **energy coherence and centroid similarity**, followed by **Hungarian matching** to obtain paired cross-modal clusters. NS-PFS then constructs a synchronized feature sequence that promotes structural coherence and reduces feature dispersion across modalities. The ordered representation is processed by an **Order-aware Memory-augmented Transformer (OMT)** with a Linformer backbone, learnable memory tokens, and an auxiliary sequencing-consistency loss that encourages the model to preserve the learned feature order during prediction. This design makes iSyncTab suitable for heterogeneous image-tabular prediction tasks, including medical imaging and visual classification with structured metadata. Across diverse multimodal benchmarks, iSyncTab demonstrates strong classification performance, improved training stability, data efficiency, and a favorable accuracy-computational cost trade-off compared with tabular-only, image-only, and recent multimodal learning baselines. The generality of the proposed mechanism was further evaluated on **audio-video multimodal data**, demonstrating its applicability beyond image-tabular learning.

## Overview

**iSyncTab** is a multimodal architecture for problems where each example has:

- **Tabular metadata** (numeric + categorical + optional text-like fields), and  
- **Image data** (e.g., medical images, natural images).
- iSyncTab itself is **not specific** to HAM10000/Pokemon/DVM/Deep Lesion/CheXpert/Pet Finder: any dataset with tabular + image inputs can be used by providing a matching PyTorch `Dataset` / `DataLoader`.

The key idea is to treat **both tabular features and image features as tokens**, then use **Neural Synchrony-guided Paired Feature Sequencing (NS-PFS)** to learn a synchronized global permutation across modalities before feeding the ordered token sequence into the **Order-aware Memory-augmented Transformer (OMT)** with a **Linformer** backbone.

## Citation

Al Zadid Sultan Bin Habib, Md Younus Ahamed, Prashnna Kumar Gyawali, Gianfranco Doretto, and Donald A. Adjeroh. **“iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony.”** In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2026. https://doi.org/10.1007/978-3-032-37035-8

BibTeX:
```bibtex
@inproceedings{habib2026isynctab,
  title     = {iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna Kumar and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year      = {2026},
  doi       = {10.1007/978-3-032-37035-8}
}
```
- Paper: https://link.springer.com/chapter/10.1007/978-3-032-37035-8 

## Files and Repository Structure

### Python package: `isynctab/`

This folder contains the core iSyncTab implementations for both image-tabular and audio-video multimodal learning:

- **`__init__.py`** - Package initializer and high-level API exports for `iSyncTab`, `iSyncTab_AV`, and `iSyncTabAV`.

- **`iSyncTab.py`** - Main iSyncTab implementation for **image-tabular multimodal learning**, including:
  - `set_seed` for reproducibility.
  - ResNet-based image token encoding.
  - Tabular token encoding for numerical, categorical, and text-style features.
  - Custom PyTorch-based GPU KMeans clustering.
  - **Neural Synchrony-guided Paired Feature Sequencing (NS-PFS)**.
  - Hungarian matching for cross-modal feature-cluster pairing.
  - Metric-aware feature sequencing and global token permutation.
  - **Order-aware Memory-augmented Transformer (OMT)** with a Linformer backbone.
  - Learnable memory tokens.
  - Classification and feature-sequencing consistency objectives.
  - `iSyncTab` as the high-level image-tabular multimodal model.

- **`iSyncTab_AV.py`** - General audio-video extension of iSyncTab, including:
  - Configurable audio and video token dimensions and sequence lengths.
  - Projection of heterogeneous audio and video representations into a shared embedding space.
  - Custom PyTorch-based GPU KMeans clustering.
  - NS-PFS-based audio-video feature sequencing.
  - Hungarian matching for synchronized cross-modal cluster pairing.
  - OMT-based multimodal fusion.
  - Classification and feature-sequencing consistency objectives.
  - `iSyncTab_AV` as the general audio-video model.
  - `iSyncTabAV` as an alternative API alias.

The audio-video implementation is **dataset-independent** and can be used with precomputed audio and video token representations from different feature extractors and datasets.

---

### Demo Notebook

- **`iSyncTab_Demo_PIP_Install.ipynb`**  
  Provides an end-to-end demonstration of the publicly installable iSyncTab package. The notebook includes:

  - Installation using:

    ```bash
    pip install isynctab
    ```

  - Import and package verification.
  - Initialization of the image-tabular `iSyncTab` model.
  - A complete **HAM10000 demonstration** using image and tabular data.
  - Optuna-based hyperparameter tuning.
  - NS-PFS feature sequencing and OMT-based training.
  - Model evaluation with displayed outputs.
  - A **generalized image-tabular example** using replaceable/dummy datasets to demonstrate how users can adapt iSyncTab to their own paired image-tabular data.
  - Example code for loading pretrained **model weights**.
  - Example code for restoring a complete **training checkpoint** and resuming or evaluating a trained model.

This notebook is intended to serve as the primary quick-start and package-usage reference for users installing iSyncTab from PyPI.

---

### Experiment Notebooks: `Experiments/`

iSyncTab was evaluated on **six image-tabular multimodal datasets** in the ECCV 2026 study. Some of the representative experiment notebooks with their displayed outputs are provided in the `Experiments/` directory to support reproducibility and further analysis.

- **`iSyncTab_DeepLesion_Subset.ipynb`**  
  Contains the iSyncTab experiment on the DeepLesion subset, including image-tabular preprocessing, model configuration, training, and evaluation with displayed results.

- **`iSyncTab_CheXPert_Subset.ipynb`**  
  Contains the iSyncTab experiment on the CheXpert subset, including multimodal preprocessing, model training, evaluation, and displayed experimental results.

- **`iSyncTab_HAM_Diagnostics.ipynb`**  
  Contains extended analysis on the HAM10000 dataset, including diagnostic experiments, ablations, robustness analyses, and additional evaluations with displayed outputs.

- **`iSyncTab_AV_RAVDESS_Sensitivity.ipynb`**  
  Evaluates the generality of the proposed feature-sequencing mechanism beyond image-tabular learning using the **RAVDESS audio-video dataset**. The notebook contains NS-PFS sensitivity analysis across different sequencing metrics and cluster configurations with displayed results.

The original iSyncTab framework focuses on **image-tabular multimodal learning**, while the audio-video experiment demonstrates that the proposed NS-PFS mechanism can also be applied to other heterogeneous multimodal feature streams.

---

### Pretrained Model and Checkpoint

The trained iSyncTab model weights and complete checkpoint for the **HAM10000** experiment will be released separately through the **Hugging Face Model Hub**.

The GitHub repository focuses on source code, package implementation, experiment notebooks, and reproducibility resources, while larger trained model artifacts are hosted separately.

The demo notebook includes examples showing how to:

```python
# Load model weights
model.load_state_dict(torch.load("isynctab_ham10000_weights.pt"))

# Load a complete checkpoint
checkpoint = torch.load("isynctab_ham10000_checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

### Main Dependencies

The repository uses the following main dependencies:

```
numpy>=1.24
pandas>=2.0
torch>=2.2
torchvision>=0.17
scipy>=1.11
Pillow>=10.0
linformer>=0.2
optuna>=3.6
matplotlib>=3.7

torchaudio>=2.2
scikit-learn>=1.3
opencv-python>=4.8
tqdm>=4.66
```

### Other Top-Level Files

- **`requirements.txt`** - Python dependencies required to run the iSyncTab package, experiment notebooks, image-tabular experiments, and audio-video experiments.
- **`iSyncTab_Architecture.png`** - High-level architecture diagram of the iSyncTab framework, illustrating NS-PFS-based cross-modal feature sequencing and OMT-based multimodal learning.
- **`iSyncTab_Demo_PIP_Install.ipynb`** - Main PyPI installation and usage demonstration, including HAM10000 Optuna tuning, a generalized replaceable-dataset example, and examples for loading model weights and checkpoints.
- **`Experiments/`** - Reproducibility and analysis notebooks for DeepLesion, CheXpert, HAM10000 diagnostics/ablations, and RAVDESS audio-video sensitivity analysis.
- **`LICENSE`** - MIT license for the iSyncTab source-code repository.
- **`README.md`** - Project overview, installation instructions, methodology, package usage, experimental resources, repository structure, links, and citation information.
- **`.gitignore`** - Git ignore rules for Python cache files, Jupyter temporary files, local datasets, checkpoints, model weights, experiment outputs, and other generated artifacts.
- **`pyproject.toml`** - Modern Python build-system configuration and package metadata used for installation and PyPI distribution.
- **`setup.cfg`** - Setuptools package configuration containing package metadata, dependencies, classifiers, project links, and package-discovery settings.

### Repository Layout

```
iSyncTab/
│
├── isynctab/
│   ├── __init__.py
│   ├── iSyncTab.py
│   └── iSyncTab_AV.py
│
├── Experiments/
│   ├── iSyncTab_DeepLesion_Subset.ipynb
│   ├── iSyncTab_CheXPert_Subset.ipynb
│   ├── iSyncTab_HAM_Diagnostics.ipynb
│   └── iSyncTab_AV_RAVDESS_Sensitivity.ipynb
│
├── iSyncTab_Demo_PIP_Install.ipynb
├── iSyncTab_Architecture.png
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── setup.cfg
└── .gitignore
```

### Tested Environment

- Python 3.10.13
- numpy 1.24.0+
- pandas 2.0.0+
- scipy 1.11.0+
- torch 2.2.0+
- torchvision 0.17.0+
- torchaudio 2.2.0+
- linformer 0.2.0+
- scikit-learn 1.3.0+
- opencv-python 4.8.0+
- tqdm 4.66.0+
- optuna 3.6.0+
- Pillow 10.0.0+
- matplotlib 3.7.0+
- jupyterlab 4.0.0+

## Installation

You can install **iStructTab** in several ways depending on your workflow.

---

### Option 1: Clone the Repository (Recommended for Development)

```bash
git clone https://github.com/zadid6pretam/iStructTab.git
cd iStructTab
pip install -r requirements.txt
pip install -e .
```

### Option 2: Install Directly from GitHub (No Cloning Needed)

```bash
pip install "git+https://github.com/zadid6pretam/iStructTab.git"
```

### Option 3: Use a Virtual Environment

```bash
python -m venv istructtab-env
source istructtab-env/bin/activate  # On Windows: istructtab-env\Scripts\activate

git clone https://github.com/zadid6pretam/iStructTab.git
cd iStructTab
pip install -r requirements.txt
pip install -e .
```

### Option 4: Local Install Without Editable Mode

```bash
git clone https://github.com/zadid6pretam/iStructTab.git
cd iStructTab
pip install -r requirements.txt
pip install .
```

### Option 5: Install from PyPI

```bash
pip install istructtab
```
## Example Usage

Below is a minimal example showing how to train **iStructTab** on a dummy multimodal classification dataset with tabular features and image-like inputs.

```python
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from istructtab import iStructTab, set_seed

# Reproducibility
set_seed(42)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Dummy multimodal classification data
# -------------------------------------------------------
num_samples = 300
num_tab_features = 40
num_classes = 3
image_size = 64

X_tab, y = make_classification(
    n_samples=num_samples,
    n_features=num_tab_features,
    n_informative=15,
    n_redundant=10,
    n_classes=num_classes,
    random_state=42,
)

X_tab = X_tab.astype(np.float32)
y = y.astype(np.int64)

# Dummy image inputs: (N, C, H, W)
X_img = np.random.rand(num_samples, 3, image_size, image_size).astype(np.float32)

# Train/test split
X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
    X_tab,
    X_img,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Standardize tabular features
scaler = StandardScaler()
X_tab_train = scaler.fit_transform(X_tab_train).astype(np.float32)
X_tab_test = scaler.transform(X_tab_test).astype(np.float32)

# Convert to tensors
X_tab_train = torch.tensor(X_tab_train, dtype=torch.float32).to(device)
X_tab_test = torch.tensor(X_tab_test, dtype=torch.float32).to(device)

X_img_train = torch.tensor(X_img_train, dtype=torch.float32).to(device)
X_img_test = torch.tensor(X_img_test, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# -------------------------------------------------------
# Initialize iStructTab
# -------------------------------------------------------
model = iStructTab(
    num_tab_features=num_tab_features,
    num_classes=num_classes,
    d_model=128,
    tab_depth=2,
    tab_heads=4,
    oemt_k=64,
    oemt_M=10,
    oemt_heads=4,
    oemt_layers=2,
    linformer_k=32,
    lambda_fs=0.1,
    pretrained_resnet=False,
    img_in_channels=3,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# -------------------------------------------------------
# Train
# -------------------------------------------------------
model.train()

epochs = 5
batch_size = 32

for epoch in range(epochs):
    permutation = torch.randperm(X_tab_train.size(0), device=device)
    total_loss = 0.0

    for start in range(0, X_tab_train.size(0), batch_size):
        idx = permutation[start:start + batch_size]

        batch_tab = X_tab_train[idx]
        batch_img = X_img_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()

        out = model(batch_tab, batch_img, y=batch_y)
        loss = out["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_tab.size(0)

    avg_loss = total_loss / X_tab_train.size(0)
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
model.eval()

with torch.no_grad():
    out = model(X_tab_test, X_img_test)
    logits = out["logits"]
    preds = logits.argmax(dim=1)

    accuracy = (preds == y_test).float().mean().item()

print(f"Test accuracy: {accuracy:.4f}")
print("GEDS feature sequence shape:", out["sequence"].shape)
print("GEDS scores shape:", out["geds_scores"].shape)
```

- The returned output dictionary contains the model predictions, GEDS sequencing information, feature-sequencing scores, and optional training losses. For supervised classification, iStructTab returns:

```python
{
    "logits": ...,       # Tensor of shape (B, num_classes)
    "sequence": ...,     # Tensor of shape (m,), learned GEDS feature order
    "geds_scores": ...,  # Tensor of shape (m,), GEDS feature scores
    "fs_scores": ...,    # Tensor of shape (B, m), OEMT-predicted feature-sequencing scores
    "beta": ...,         # Tensor of shape (B, m), target sequencing vector

    # Returned only when labels y are provided
    "loss": ...,         # Total loss = CE loss + lambda_fs * feature-sequencing loss
    "loss_ce": ...,      # Cross-entropy classification loss
    "loss_fs": ...       # Feature-sequencing regularization loss
}
```

- For binary classification, set:

```python
num_classes = 2

model = iStructTab(
    num_tab_features=num_tab_features,
    num_classes=num_classes,
    d_model=128,
    tab_depth=2,
    tab_heads=4,
    oemt_k=64,
    oemt_M=10,
    oemt_heads=4,
    oemt_layers=2,
    linformer_k=32,
    lambda_fs=0.1,
    pretrained_resnet=False,
    img_in_channels=3,
)
```

- During evaluation, predictions can be obtained from `out["logits"]`, and standard classification metrics such as accuracy, precision, recall, and F1-score can be computed using `scikit-learn`.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.eval()

with torch.no_grad():
    out = model(X_tab_test, X_img_test)
    preds = out["logits"].argmax(dim=1).cpu().numpy()
    y_true = y_test.cpu().numpy()

metrics = {
    "accuracy": accuracy_score(y_true, preds),
    "macro_precision": precision_score(y_true, preds, average="macro", zero_division=0),
    "macro_recall": recall_score(y_true, preds, average="macro", zero_division=0),
    "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
}

print(metrics)
```

## For fuller experiments, Optuna tuning, and diagnostic analysis, see:

- **`HAM_iStructTab.ipynb`**

This notebook contains the full HAM10000 experiment from the iStructTab workflow, including multimodal image-tabular preprocessing, Optuna-based hyperparameter tuning, GEDS feature sequencing, OEMT training, evaluation metrics, robustness checks, calibration analysis, and diagnostic visualizations.

- **`iStructTab_PIP_Install_Check.ipynb`**

This notebook provides a minimal installation check for the PyPI/GitHub-installed `istructtab` package, including import verification, package/version checks, initialization of core iStructTab components, and a small toy workflow to confirm that the installed package runs correctly.


## Related Work and Project Context

iStructTab is part of my PhD research on structured tabular and multimodal deep learning, with a focus on feature ordering/sequencing, and representation learning for heterogeneous data. The project extends my broader research direction on order-aware tabular modeling by studying how image and tabular representations can be fused through a structured feature sequence rather than treated as an unordered concatenated vector.

In this work, iStructTab formulates multimodal image-tabular fusion as a feature sequencing problem inspired by the Column Permutation Problem (CPP). It introduces Graph-Enhanced Descriptor Sequencing (GEDS) to construct a data-driven feature order and uses an Order-Aware Efficient Transformer with Memory Augmentation (OEMT) to preserve and exploit that order during prediction. This connects directly to my dissertation research themes on feature ordering, structure-aware representation learning, and efficient deep learning for tabular and multimodal data.

### GOTabPFN (ICML 2026)

Our recent ICML 2026 Regular main conference paper on feature ordering and compression for tabular foundation models for high-dimensional low-sample-size tabular data:
- **GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data**

- GitHub: https://github.com/zadid6pretam/GOTabPFN
- - **Find it on ICML portal:** https://icml.cc/virtual/2026/poster/62523
- **Project Webpage:** https://www.zadidhabib.com/gotabpfn.html
- **OpenReview:** https://openreview.net/forum?id=fpqfV3lCIB
- **Hugging Face Space:** [ZeroGPU Live Demo](https://zadid6pretam-GOTabPFN.hf.space) *(recommended; faster GPU-backed testing)* | [CPU Backup Demo](https://zadid6pretam-GOTabPFN-CPU.hf.space) *(use if ZeroGPU is unavailable)* | [ZeroGPU Space Repository](https://huggingface.co/spaces/zadid6pretam/GOTabPFN) | [CPU Backup Space Repository](https://huggingface.co/spaces/zadid6pretam/GOTabPFN_CPU)

```bibtex
@inproceedings{habib2026gotabpfn,
  title     = {GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna Kumar and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

### iSyncTab (ECCV 2026)

Our neural synchrony-based cross-modal feature sequencing framework for multimodal learning with image and tabular data. iSyncTab addresses the image–tabular integration problem by aligning and sequencing cross-modal feature groups before structured multimodal representation learning.

- **iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony**  
- Accepted at the European Conference on Computer Vision (ECCV 2026)
- GitHub: https://github.com/zadid6pretam/iSyncTab (will be made public soon)
- Project Page: https://www.zadidhabib.com/isynctab.html (will be made public soon)

```bibtex
@inproceedings{habib2026isynctab,
  title     = {iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year      = {2026}
}
```
- If you are interested in cross-modal feature sequencing, neural synchrony-guided image–tabular integration, and order-aware multimodal representation learning, please refer to the iSyncTab repository, project page, and paper.


### BSTabDiff (ICLR 2026 DeLTa Workshop)

Our generative modeling framework for high-dimensional low-sample-size tabular data:
- **BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation**

- GitHub: https://github.com/zadid6pretam/BSTabDiff

- OpenReview: https://openreview.net/forum?id=RKNDy0KhGT


```bibtex
@inproceedings{habib2026bstabdiff,
  title     = {BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna Kumar and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {ICLR 2026 2nd Workshop on Deep Generative Models in Machine Learning: Theory, Principle and Efficacy (DeLTa)},
  year      = {2026}
}
```
- If you are interested in high-dimensional tabular synthesis, block-subunit generation, and diffusion/flow priors for HDLSS tabular data, please also refer to the BSTabDiff repository and paper.


### iStructTab (ICPR 2026)

Our structured feature sequencing framework for multimodal learning with image and tabular data. This work involves feature sequencing or ordering for multimodal image-tabular representation learning.

- **iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data**  

- GitHub: https://github.com/zadid6pretam/iStructTab
- Paper: https://link.springer.com/chapter/10.1007/978-3-032-31404-8_43 
- arXiv: https://arxiv.org/abs/2608.04348

```bibtex
@inproceedings{habib2026istructtab,
  title     = {iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition},
  year      = {2026},
  address   = {Lyon, France}
}
```
- If you are interested in structured feature sequencing, multimodal fusion of image and tabular data (the integration problem), and feature order-aware tabular representation learning, please also refer to the iStructTab repository and paper.

## DynaTab (AAAI 2026 NeurAI Workshop)

One of our older works on learned feature ordering for high-dimensional tabular data:

- **DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data**

- GitHub: https://github.com/zadid6pretam/DynaTab
- Paper Link: https://proceedings.mlr.press/v308/habib26a.html

Bibtex:
```bash
@InProceedings{dynatab,
  title = 	 {{DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data}},
  author =       {Habib, Al Zadid Sultan Bin and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = 	 {{Proceedings of the First Workshop on NeuroAI Multimodal Intelligence @ AAAI 2026}},
  pages     = {27--57},
  year      = {2026},
  volume    = {308},
  series    = {{Proceedings of Machine Learning Research}},
  publisher = {PMLR},
  url = 	 {https://proceedings.mlr.press/v308/habib26a.html}
}
```
- If you are interested in learned feature ordering, neural rewiring for high-dimensional tabular data, and sequential backbone design for HDLSS settings, please also refer to the benchmark study in DynaTab repository and paper.


### TabSeq (ICPR 2024)

Our earlier work on sequential modeling for tabular data:

- **TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering**  

-  GitHub: https://github.com/zadid6pretam/TabSeq  

-  Springer ICPR 2024 proceedings: https://link.springer.com/chapter/10.1007/978-3-031-78128-5_27

-  arXiv: https://arxiv.org/abs/2410.13203

```bibtex
@inproceedings{habib2024tabseq,
  title={TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering},
  author={Habib, Al Zadid Sultan Bin and Wang, Kesheng and Hartley, Mary-Anne and Doretto, Gianfranco and A. Adjeroh, Donald},
  booktitle={International Conference on Pattern Recognition},
  pages={418--434},
  year={2024},
  organization={Springer}
}
```
- If you are interested in sequential feature ordering for tabular data, deep sequential backbones, and early feature ordering-based tabular modeling, please also refer to the TabSeq repository and paper.


----------------------------------------------------------------------------------------------------------------------------------------------------------


### ZAYAN (ICPR 2026)

This repository corresponds to our separate collaborative work on tabular remote sensing and environmental data:
- ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data
- Paper: https://link.springer.com/chapter/10.1007/978-3-032-31397-3_1
- arXiv: https://arxiv.org/abs/2604.27606
- GitHub: https://github.com/zadid6pretam/ZAYAN

```bibtex
@inproceedings{habib2026zayan,
  title     = {ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data},
  author    = {Habib, Al Zadid Sultan Bin and Tasnim, Tanpia and Islam, Md. Ekramul and Tabasum, Muntasir},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition},
  year      = {2026},
  address   = {Lyon, France}
}
```
- ZAYAN focuses on feature-level contrastive learning and Transformer-based classification for tabular remote sensing and environmental datasets.
- Unlike my PhD dissertation projects on high-dimensional tabular learning and HDLSS modeling, ZAYAN was developed as a separate collaboration.

## Contact

For any questions, issues, or suggestions related to this repository, please feel free to contact us or open an issue on GitHub.


## Repository Structure

A typical layout is:

```text
.
├── istructtab/
│   ├── __init__.py
│   └── iStructTab.py                    # Core iStructTab implementation: GEDS + OEMT
├── HAM_iStructTab.ipynb                 # Full HAM10000 experiment with Optuna tuning and diagnostics
├── iStructTab_PIP_Install_Check.ipynb   # Minimal pip-install/import/API check notebook
├── iStructTab_Architecture.png          # High-level architecture diagram
├── requirements.txt                     # Runtime dependencies
├── pyproject.toml                       # Build system and PyPI metadata
├── setup.cfg                            # Optional setuptools configuration
├── LICENSE                              # MIT license
├── .gitignore                           # Git ignore rules
└── README.md                            # Project overview, installation, usage, and citation



