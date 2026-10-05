from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from .iSyncTab import iSyncTab, set_seed


# ============================================================
# Dataset
# ============================================================

class _iSyncTabDataset(Dataset):
    def __init__(
        self,
        X_tab,
        X_img,
        y=None,
        image_size=128,
        image_root=None,
    ):
        self.X_tab = torch.as_tensor(
            np.asarray(X_tab),
            dtype=torch.float32,
        )

        self.X_img = X_img
        self.y = None if y is None else torch.as_tensor(
            y,
            dtype=torch.long,
        )

        self.image_size = int(image_size)
        self.image_root = (
            Path(image_root)
            if image_root is not None
            else None
        )

        if len(self.X_tab) != len(self.X_img):
            raise ValueError(
                "Tabular and image inputs must contain "
                "the same number of samples."
            )

        if self.y is not None and len(self.y) != len(self.X_tab):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

    def __len__(self):
        return len(self.X_tab)

    def _load_image(self, item):
        # --------------------------------------------------------
        # Image path
        # --------------------------------------------------------
        if isinstance(item, (str, Path)):
            path = Path(item)

            if not path.is_absolute() and self.image_root is not None:
                path = self.image_root / path

            with Image.open(path) as img:
                img = img.convert("RGB")
                img = TF.resize(
                    img,
                    [self.image_size, self.image_size],
                    antialias=True,
                )
                return TF.to_tensor(img)

        # --------------------------------------------------------
        # NumPy image
        # --------------------------------------------------------
        if isinstance(item, np.ndarray):
            img = torch.as_tensor(item)

        # --------------------------------------------------------
        # Torch tensor
        # --------------------------------------------------------
        elif torch.is_tensor(item):
            img = item.clone()

        else:
            raise TypeError(
                "Images must be paths, NumPy arrays, "
                "or torch tensors."
            )

        # H x W -> 1 x H x W
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # H x W x C -> C x H x W
        elif (
            img.ndim == 3
            and img.shape[-1] in (1, 3)
            and img.shape[0] not in (1, 3)
        ):
            img = img.permute(2, 0, 1)

        if img.ndim != 3:
            raise ValueError(
                "Each image must have shape HxW, CHW, or HWC."
            )

        original_dtype = img.dtype
        img = img.float()

        if original_dtype == torch.uint8:
            img = img / 255.0

        # iSyncTab's image encoder can internally expand grayscale
        # images to RGB.
        if img.shape[0] not in (1, 3):
            raise ValueError(
                f"Expected 1 or 3 image channels, got {img.shape[0]}."
            )

        img = TF.resize(
            img,
            [self.image_size, self.image_size],
            antialias=True,
        )

        return img

    def __getitem__(self, idx):
        x_tab = self.X_tab[idx]
        x_img = self._load_image(self.X_img[idx])

        if self.y is None:
            return x_tab, x_img

        return x_tab, x_img, self.y[idx]


# ============================================================
# sklearn-style classifier
# ============================================================

class iSyncTabClassifier(ClassifierMixin, BaseEstimator):
    """
    sklearn-style estimator wrapper for image-tabular iSyncTab.

    X can be provided as either:

        (X_tab, X_img)

    or:

        {
            "tabular": X_tab,
            "images": X_img,
        }

    The underlying iSyncTab architecture is unchanged.
    """

    def __init__(
        self,
        # --------------------------------------------------------
        # iSyncTab architecture
        # --------------------------------------------------------
        d_model=128,
        num_clusters=4,
        metric="variance",
        linformer_depth=4,
        linformer_heads=4,
        linformer_k=32,
        lambda_fs=0.1,
        num_memory_tokens=1,
        pretrained_resnet=False,

        # NS-PFS
        nspfs_bins=32,
        nspfs_mi_chunk=128,
        nspfs_sync_temperature=1.0,
        nspfs_energy_weight=1.0,
        nspfs_centroid_weight=1.0,
        nspfs_pair_order="sync",
        nspfs_within_cluster_order="metric_desc",

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------
        epochs=20,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-4,

        # --------------------------------------------------------
        # Data / runtime
        # --------------------------------------------------------
        image_size=128,
        standardize_tabular=True,
        image_root=None,
        device="auto",
        num_workers=0,
        random_state=42,
        verbose=False,
    ):
        self.d_model = d_model
        self.num_clusters = num_clusters
        self.metric = metric
        self.linformer_depth = linformer_depth
        self.linformer_heads = linformer_heads
        self.linformer_k = linformer_k
        self.lambda_fs = lambda_fs
        self.num_memory_tokens = num_memory_tokens
        self.pretrained_resnet = pretrained_resnet

        self.nspfs_bins = nspfs_bins
        self.nspfs_mi_chunk = nspfs_mi_chunk
        self.nspfs_sync_temperature = nspfs_sync_temperature
        self.nspfs_energy_weight = nspfs_energy_weight
        self.nspfs_centroid_weight = nspfs_centroid_weight
        self.nspfs_pair_order = nspfs_pair_order
        self.nspfs_within_cluster_order = (
            nspfs_within_cluster_order
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.image_size = image_size
        self.standardize_tabular = standardize_tabular
        self.image_root = image_root
        self.device = device
        self.num_workers = num_workers
        self.random_state = random_state
        self.verbose = verbose

    # ========================================================
    # Input handling
    # ========================================================

    @staticmethod
    def _unpack_X(X):
        if isinstance(X, dict):
            X_tab = X.get("tabular", X.get("x_tab"))
            X_img = X.get(
                "images",
                X.get("image", X.get("x_img")),
            )

            if X_tab is None or X_img is None:
                raise ValueError(
                    "Dictionary input must contain tabular/x_tab "
                    "and images/image/x_img."
                )

            return X_tab, X_img

        if isinstance(X, (tuple, list)) and len(X) == 2:
            return X[0], X[1]

        raise TypeError(
            "X must be either (X_tab, X_img) or a dictionary "
            "containing tabular and image inputs."
        )

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "to_numpy"):
            return X.to_numpy()

        if torch.is_tensor(X):
            return X.detach().cpu().numpy()

        return np.asarray(X)

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        return torch.device(self.device)

    # ========================================================
    # Fit
    # ========================================================

    def fit(self, X, y):
        set_seed(self.random_state)

        self.device_ = self._resolve_device()

        X_tab, X_img = self._unpack_X(X)

        X_tab = self._to_numpy(X_tab).astype(np.float32)

        if X_tab.ndim != 2:
            raise ValueError(
                "Tabular input must have shape "
                "(n_samples, n_features)."
            )

        self.n_features_in_ = X_tab.shape[1]

        # ----------------------------------------------------
        # Target encoding
        # ----------------------------------------------------
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(
            np.asarray(y)
        ).astype(np.int64)

        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                "iSyncTabClassifier requires at least two classes."
            )

        # ----------------------------------------------------
        # Tabular scaling
        # ----------------------------------------------------
        if self.standardize_tabular:
            self.scaler_ = StandardScaler()
            X_tab = self.scaler_.fit_transform(
                X_tab
            ).astype(np.float32)
        else:
            self.scaler_ = None

        # ----------------------------------------------------
        # Core iSyncTab model
        # ----------------------------------------------------
        self.model_ = iSyncTab(
            num_tab_features=self.n_features_in_,
            num_classes=self.n_classes_,

            d_model=self.d_model,
            num_clusters=self.num_clusters,
            metric=self.metric,

            linformer_depth=self.linformer_depth,
            linformer_heads=self.linformer_heads,
            linformer_k=self.linformer_k,

            lambda_fs=self.lambda_fs,
            num_memory_tokens=self.num_memory_tokens,
            pretrained_resnet=self.pretrained_resnet,

            nspfs_bins=self.nspfs_bins,
            nspfs_mi_chunk=self.nspfs_mi_chunk,
            nspfs_sync_temperature=(
                self.nspfs_sync_temperature
            ),
            nspfs_energy_weight=self.nspfs_energy_weight,
            nspfs_centroid_weight=(
                self.nspfs_centroid_weight
            ),
            nspfs_pair_order=self.nspfs_pair_order,
            nspfs_within_cluster_order=(
                self.nspfs_within_cluster_order
            ),

            device=self.device_,
        ).to(self.device_)

        dataset = _iSyncTabDataset(
            X_tab=X_tab,
            X_img=X_img,
            y=y_encoded,
            image_size=self.image_size,
            image_root=self.image_root,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device_.type == "cuda",
        )

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------
        self.history_ = []

        for epoch in range(self.epochs):
            self.model_.train()

            total_loss = 0.0
            total_samples = 0

            for x_tab_batch, x_img_batch, y_batch in loader:
                x_tab_batch = x_tab_batch.to(
                    self.device_,
                    non_blocking=True,
                )
                x_img_batch = x_img_batch.to(
                    self.device_,
                    non_blocking=True,
                )
                y_batch = y_batch.to(
                    self.device_,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                out = self.model_(
                    x_tab_batch,
                    x_img_batch,
                    y=y_batch,
                )

                loss = out["loss"]

                loss.backward()
                optimizer.step()

                n = y_batch.size(0)

                total_loss += loss.detach().item() * n
                total_samples += n

            epoch_loss = total_loss / max(total_samples, 1)

            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                }
            )

            if self.verbose:
                print(
                    f"Epoch {epoch + 1:03d}/{self.epochs} "
                    f"| loss={epoch_loss:.6f}"
                )

        self.is_fitted_ = True

        return self

    # ========================================================
    # Probability prediction
    # ========================================================

    @torch.no_grad()
    def predict_proba(self, X):
        check_is_fitted(
            self,
            ["model_", "is_fitted_"],
        )

        X_tab, X_img = self._unpack_X(X)

        X_tab = self._to_numpy(X_tab).astype(np.float32)

        if X_tab.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} tabular features, "
                f"got {X_tab.shape[1]}."
            )

        if self.scaler_ is not None:
            X_tab = self.scaler_.transform(
                X_tab
            ).astype(np.float32)

        dataset = _iSyncTabDataset(
            X_tab=X_tab,
            X_img=X_img,
            y=None,
            image_size=self.image_size,
            image_root=self.image_root,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device_.type == "cuda",
        )

        self.model_.eval()

        all_probs = []

        for x_tab_batch, x_img_batch in loader:
            x_tab_batch = x_tab_batch.to(
                self.device_,
                non_blocking=True,
            )
            x_img_batch = x_img_batch.to(
                self.device_,
                non_blocking=True,
            )

            out = self.model_(
                x_tab_batch,
                x_img_batch,
            )

            probs = torch.softmax(
                out["logits"],
                dim=1,
            )

            all_probs.append(
                probs.cpu().numpy()
            )

        return np.concatenate(
            all_probs,
            axis=0,
        )

    # ========================================================
    # Hard predictions
    # ========================================================

    def predict(self, X):
        probs = self.predict_proba(X)

        class_ids = probs.argmax(axis=1)

        return self.label_encoder_.inverse_transform(
            class_ids
        )
