from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset


class MultimodalAMDDataset(Dataset):
    """AMD multimodal dataset that joins tabular metadata with OCT B‑scans.

    **Key order of operations**
    1. Load & minimally clean annotation sheets.
    2. Derive raw‑row features (e.g. ICD frequency).
    3. **Expand rows → one row per B‑scan _before_ any imputation/encoding.**
    4. Impute & encode using the expanded dataframe so statistics include every row.
    5. Tensorise for PyTorch.
    """

    # --------------------------------------------------------------
    # ctor
    # --------------------------------------------------------------

    def __init__(
        self,
        tabular_path: Optional[str] = None,
        image_root_dir: Optional[str] = None,
        *,
        data_sources: Optional[Dict[str, str]] = None,
        transforms=None,
    ):
        # 1. Resolve data sources ------------------------------------
        if data_sources is None:
            if tabular_path is None:
                raise ValueError("tabular_path required when data_sources is None")
            data_sources = {tabular_path: image_root_dir}

        dfs: list[pd.DataFrame] = []
        img_roots: set[Path] = set()

        for tab_path, img_root in data_sources.items():
            if img_root is None:
                raise ValueError(f"image_root_dir missing for {tab_path}")

            # 1.1 Load spreadsheet
            df = pd.read_excel(tab_path).copy()

            # 1.2 Tidy column names → snake_case
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # 1.3 Legacy support: rename 'age' → 'age_at_visit'
            if "age" in df.columns and "age_at_visit" not in df.columns:
                df.rename(columns={"age": "age_at_visit"}, inplace=True)

            # 1.4 Keep reference to image root for later expansion
            df["__img_root__"] = str(img_root)

            # 1.5 Drop rows missing labels early
            df.dropna(subset=["stage"], inplace=True)

            dfs.append(df)
            img_roots.add(Path(img_root))

        # 2. Raw‑row feature engineering -----------------------------
        df_all = pd.concat(dfs, ignore_index=True)
        icd_freq = df_all["icd_primary"].value_counts()
        for _df in dfs:
            _df["icd_freq"] = _df["icd_primary"].map(icd_freq).fillna(0)

        # 3. Feature schemas -----------------------------------------
        self.categorical_cols: List[str] = [
            "laterality",
            "sex",
            "primary_dx_yn",
            "cigarettes_yn_final",
            "smoking_tob_use_name_final",
            "smokeless_tob_use_name_final",
            "tobacco_user_name_final",
            "alcohol_use_name_final",
            "ill_drug_user_name_final",
        ]
        self.continuous_cols: List[str] = ["age_at_visit", "va_continuous", "icd_freq"]
        self.label_col: str = "y"  # numeric after encoding

        # 4. Concatenate raw sheets → original_df (no processing yet)
        self.original_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

        # 5. Build patient‑dir map -----------------------------------
        self.patient_dir_map: dict[int, List[Path]] = defaultdict(list)
        for root in img_roots:
            for d in root.iterdir():
                if d.is_dir() and d.name.isdigit():
                    self.patient_dir_map[int(d.name)].append(d)

        # 6. Expand each OCT volume into one row per B‑scan ----------
        self.original_df["visit_date"] = self.original_df["visit_date"].astype(str)
        self.expected_volume_ids = {
            f"{int(r.research_id)}_{r.laterality}_{r.visit_date}"
            for r in self.original_df.itertuples()
        }
        self.loaded_volume_ids: set[str] = set()
        self.transforms = transforms
        self.df = self._expand_with_images()

        # 7. Now process tabular columns -----------------------------
        self._process_tabular()

        # 8. Tensorise ----------------------------------------------
        self._tensorise()

    # --------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------

    def _process_tabular(self):
        """Impute numeric/categorical values and encode labels _after_ expansion."""
        # Impute
        self.df[self.continuous_cols] = SimpleImputer(strategy="mean").fit_transform(
            self.df[self.continuous_cols]
        )
        self.df[self.categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
            self.df[self.categorical_cols]
        )

        # Ordinal‑encode categoricals (unknown→‑1, then +1 so UNK==0)
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.df[self.categorical_cols] = (
            ord_enc.fit_transform(self.df[self.categorical_cols]).astype(int) + 1
        )

        # Label‑encode targets
        self.label_encoder = LabelEncoder()
        self.df[self.label_col] = self.label_encoder.fit_transform(
            self.df["stage"].astype(str)
        )

    @staticmethod
    def _find_b_scans_directory(root: Path) -> Optional[Path]:
        for dirpath, _d, fnames in os.walk(root):
            if Path(dirpath).name.lower() == "b-scans" and any(
                f.lower().endswith((".jpg", ".png")) for f in fnames
            ):
                return Path(dirpath)
        return None

    def _expand_with_images(self) -> pd.DataFrame:
        """Duplicate every tabular row once per B‑scan image."""
        expanded: list[dict] = []

        for (pid, eye, vdate), rows in self.original_df.groupby(
            ["research_id", "laterality", "visit_date"]
        ):
            pid_int = int(pid)
            vdate = str(vdate)
            eye_str = str(eye)  # <‑‑ ensure string before path join

            img_roots = self.patient_dir_map.get(pid_int, [])
            if not img_roots:
                continue

            found = False
            for patient_dir in img_roots:  # handle multi‑site patients
                scan_root = patient_dir / eye_str / vdate
                if not scan_root.exists():
                    continue

                b_scans_dir = scan_root / "B-Scans"
                if not b_scans_dir.is_dir():
                    b_scans_dir = self._find_b_scans_directory(scan_root)
                if not b_scans_dir:
                    continue

                volume_id = f"{pid_int}_{eye_str}_{vdate}"
                self.loaded_volume_ids.add(volume_id)

                for img_f in sorted(
                    p for p in b_scans_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}
                ):
                    for _, tab_row in rows.iterrows():
                        row = tab_row.to_dict()
                        row["image_path"] = str(img_f)
                        row["volume_id"] = volume_id
                        expanded.append(row)
                found = True
                break  # stop after first matching site

            if not found:  # fallback: keep row without image
                for _, tab_row in rows.iterrows():
                    row = tab_row.to_dict()
                    row["image_path"] = None
                    row["volume_id"] = f"{pid_int}_{eye_str}_{vdate}"
                    expanded.append(row)

        print(
            f"Created {len(expanded)} rows ("  # diagnostic summary
            f"{len(self.loaded_volume_ids)}/{len(self.expected_volume_ids)} volumes)"
        )
        return pd.DataFrame(expanded)

    def _tensorise(self):
        self.X_categ = torch.tensor(self.df[self.categorical_cols].values, dtype=torch.long)
        self.X_cont = torch.tensor(self.df[self.continuous_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.df[self.label_col].values, dtype=torch.long)

    # --------------------------------------------------------------
    # Dataset protocol
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Convenience utilities
    # --------------------------------------------------------------

    def get_category_dims(self) -> List[int]:
        return [int(self.df[c].max()) + 1 for c in self.categorical_cols]

    def get_label_map(self) -> List[str]:
        return list(self.label_encoder.classes_)

    def get_num_classes(self) -> int:
        return len(self.label_encoder.classes_)

    def get_class_distribution(self) -> pd.Series:
        counts = self.df[self.label_col].value_counts().sort_index()
        labels = [self.label_encoder.inverse_transform([i])[0] for i in counts.index]
        return pd.Series(counts.values, index=labels)

    def get_volume_label(self, volume_id: str):
        return self.df.loc[self.df["volume_id"] == volume_id, self.label_col].iloc[0]

    # --------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------

    def report_missing_volumes(self):
        missing = self.expected_volume_ids - self.loaded_volume_ids
        print(
            f"Expected: {len(self.expected_volume_ids)} | "
            f"Loaded: {len(self.loaded_volume_ids)} | Missing: {len(missing)}"
        )
        for vid in sorted(missing):
            print("  •", vid)
