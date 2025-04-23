from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class MultimodalAMDDataset(Dataset):
    """
    Multisource AMD dataset:
      • 1-to-1 (old behaviour): tabular_path + image_root_dir
      • many-to-many : data_sources={tabular_path: image_root_dir, ...}
    Each row ⇒ one B-scan + duplicated tabular metadata.
    """

    # ---------------------------- ctor -------------------------------- #
    def __init__(
        self,
        tabular_path: Optional[str] = None,
        image_root_dir: Optional[str] = None,
        *,
        data_sources: Optional[Dict[str, str]] = None,
        transforms=None,
    ):
        # ------------------- resolve data sources --------------------- #
        if data_sources is None:
            if tabular_path is None:
                raise ValueError("tabular_path required when data_sources is None")
            data_sources = {tabular_path: image_root_dir}

        # load and tag every annotation file
        dfs = []
        img_roots: set[Path] = set()
        for tab_path, img_root in data_sources.items():
            if img_root is None:
                raise ValueError(f"image_root_dir missing for {tab_path}")
            df = pd.read_excel(tab_path).dropna(subset=["stage"]).copy()
            df["__img_root__"] = str(img_root)  # keep per-row reference
            dfs.append(df)
            img_roots.add(Path(img_root))
        self.original_df = pd.concat(dfs, ignore_index=True)

        # ---------------- tabular columns ----------------------------- #
        self.categorical_cols = [
            "laterality",
            "SEX",
            "CIGARETTES_YN_final",
            "SMOKING_TOB_USE_NAME_final",
            "SMOKELESS_TOB_USE_NAME_final",
            "TOBACCO_USER_NAME_final",
            "ALCOHOL_USE_NAME_final",
            "PRIMARY_DX_YN",
            "ICD_primary",
        ]
        self.continuous_cols = ["AGE_AT_VISIT", "VA_continuous"]
        self.label_col = "stage"

        # ---------------- build patient_dir_map ----------------------- #
        # maps patient_id → [Path(...), Path(...)] (handles duplicate IDs across sites)
        self.patient_dir_map: dict[int, list[Path]] = defaultdict(list)
        for root in img_roots:
            for d in root.iterdir():
                if d.is_dir() and d.name.isdigit():
                    self.patient_dir_map[int(d.name)].append(d)

        # ---------------- expand with images -------------------------- #
        self.expected_volume_ids = {
            f"{int(r.research_id)}_{r.laterality}_{r.visit_date}"
            for r in self.original_df.itertuples()
        }
        self.loaded_volume_ids: set[str] = set()
        self.transforms = transforms
        self.df = self._expand_with_images()

        # -------------- encode labels & feature tensors --------------- #
        self._encode_and_tensorise()

    # ---------------------- helper: find B-Scans ---------------------- #
    @staticmethod
    def _find_b_scans_directory(root: Path) -> Optional[Path]:
        for dirpath, _dnames, fnames in os.walk(root):
            if (
                Path(dirpath).name.lower() == "b-scans"
                and any(f.lower().endswith((".jpg", ".png")) for f in fnames)
            ):
                return Path(dirpath)
        return None

    # ---------------- expand each volume into rows ------------------- #
    def _expand_with_images(self) -> pd.DataFrame:
        expanded: list[dict] = []

        for (pid, eye, vdate), rows in self.original_df.groupby(
            ["research_id", "laterality", "visit_date"]
        ):
            pid_int = int(pid)
            vdate = str(vdate)
            img_roots = self.patient_dir_map.get(pid_int, [])
            if not img_roots:
                continue

            found_any = False
            for patient_dir in img_roots:  # try each site until we find scans
                scan_root = patient_dir / eye / vdate
                if not scan_root.exists():
                    continue
                b_scans_dir = scan_root / "B-Scans"
                if not b_scans_dir.is_dir():
                    b_scans_dir = self._find_b_scans_directory(scan_root)
                if not b_scans_dir:
                    continue

                volume_id = f"{pid_int}_{eye}_{vdate}"
                self.loaded_volume_ids.add(volume_id)

                for img_file in sorted(
                    f for f in b_scans_dir.iterdir() if f.suffix.lower() in {".jpg", ".png"}
                ):
                    for _, tab_row in rows.iterrows():
                        row = tab_row.to_dict()
                        row["image_path"] = str(img_file)
                        row["volume_id"] = volume_id
                        expanded.append(row)
                found_any = True
                break  # stop after first matching site

            if not found_any:
                # fall back: keep tabular rows without images
                for _, tab_row in rows.iterrows():
                    row = tab_row.to_dict()
                    row["image_path"] = None
                    row["volume_id"] = f"{pid_int}_{eye}_{vdate}"
                    expanded.append(row)

        print(
            f"Created {len(expanded)} rows "
            f"({len(self.loaded_volume_ids)}/{len(self.expected_volume_ids)} volumes)"
        )
        return pd.DataFrame(expanded)

    # ------------------- encode + tensorise --------------------------- #
    def _encode_and_tensorise(self):
        self.label_encoder = LabelEncoder()
        self.df[self.label_col] = self.label_encoder.fit_transform(
            self.df[self.label_col].astype(str)
        )

        df_cat = pd.get_dummies(
            self.df[self.categorical_cols].fillna("missing").astype(str), drop_first=False
        ).astype("float32")

        df_cont = (
            self.df[self.continuous_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(method="ffill")
            .astype("float32")
        )

        df_tab = pd.concat([df_cont, df_cat], axis=1)

        self.X_cont = torch.tensor(df_tab.values, dtype=torch.float32)
        self.X_categ = torch.empty((len(self.df), 0), dtype=torch.long)  # placeholder
        self.y = torch.tensor(self.df[self.label_col].values, dtype=torch.long)

    # ----------------------- Dataset API ------------------------------ #
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "categorical": self.X_categ[idx],
            "continuous": self.X_cont[idx],
            "label": self.y[idx],
        }
        img_path = self.df.iloc[idx]["image_path"]
        if img_path and Path(img_path).exists():
            with Image.open(img_path).convert("RGB") as img:
                if self.transforms:
                    img = self.transforms(img)
                item["image"] = img
        return item

    # ---------------- utility getters -------------------------------- #
    def get_category_dims(self) -> List[int]:
        return [self.df[c].nunique() for c in self.categorical_cols]

    def get_label_map(self) -> List[str]:
        return self.label_encoder.classes_

    def get_num_classes(self) -> int:
        return len(self.label_encoder.classes_)

    def get_class_distribution(self) -> pd.Series:
        """
        Returns a Series whose index contains the **original label strings**
        and whose values are sample counts.
        """
        counts = self.df[self.label_col].value_counts().sort_index()
        labels = [self.label_encoder.inverse_transform([i])[0] for i in counts.index]
        return pd.Series(counts.values, index=labels)

    def get_volume_label(self, volume_id: str):
        return self.df.loc[self.df["volume_id"] == volume_id, self.label_col].iloc[0]

    # ---------------- diagnostics ------------------------------------ #
    def report_missing_volumes(self):
        missing = self.expected_volume_ids - self.loaded_volume_ids
        print(
            f"Expected: {len(self.expected_volume_ids)} | "
            f"Loaded: {len(self.loaded_volume_ids)} | Missing: {len(missing)}"
        )
        for vid in sorted(missing):
            print("  •", vid)
