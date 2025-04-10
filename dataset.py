import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class MultimodalAMDDataset(Dataset):
    def __init__(self, tabular_path, image_root_dir=None, transforms=None):
        """
        Multimodal dataset combining tabular data and OCT images for AMD.
        Optimized for Tab Transformer with proper handling of missing values.
        
        Args:
            tabular_path (str): Path to the Excel file with tabular data and labels
            image_root_dir (str, optional): Root directory containing OCT images
            transforms (callable, optional): Optional transforms to apply to images
        """
        # Define column groups
        self.categorical_cols = [
            'Laterality', 'SEX', 'CIGARETTES_YN', 'SMOKING_TOB_USE_NAME',
            'SMOKELESS_TOB_USE_NAME', 'TOBACCO_USER_NAME', 'ALCOHOL_USE_NAME',
            'ILL_DRUG_USER_NAME', 'VA (Closest to Dx)', 'PRIMARY_DX_YN'
        ]
        
        self.continuous_cols = [
            'BIRTH_YEAR', 'BIRTH_MONTH', 'BIRTH_DAY',
            'VISION_YEAR', 'VISION_MONTH', 'VISION_DAY',
            'DIAGNOSIS_YEAR', 'DIAGNOSIS_MONTH', 'DIAGNOSIS_DAY'
        ]
        
        self.label_col = 'Diagnosis Label'
        
        # Load tabular data - only drop rows with missing label
        self.original_df = pd.read_excel(tabular_path)
        self.original_df = self.original_df.dropna(subset=[self.label_col])
        
        # Image data setup
        self.image_root_dir = image_root_dir
        self.transforms = transforms
        self.has_images = image_root_dir is not None and os.path.exists(image_root_dir)
        self.loaded_volume_ids = set()
        self.expected_volume_ids = set()
        
        # Create expanded dataset with one row per B-scan
        if self.has_images:
            # Compute expected volume_ids from label file
            self.expected_volume_ids = set(
                self.original_df.apply(
                    lambda row: f"{int(row['Patient Number'])}_{row['Laterality']}_{int(row['Diagnosis Date'])}",
                    axis=1
                )
            )
            self._create_expanded_dataset()
        else:
            self.df = self.original_df.copy()
        
        # Encode categorical data with special handling for missing values
        self.encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Fill NaN with a special string before encoding
            filled_values = self.df[col].fillna('MISSING_VALUE').astype(str)
            self.df[col] = le.fit_transform(filled_values)
            self.encoders[col] = le
        
        # Encode label column (should not have missing values)
        le = LabelEncoder()
        self.df[self.label_col] = le.fit_transform(self.df[self.label_col].astype(str))
        self.encoders[self.label_col] = le
        
        # Create mask for missing values in continuous columns
        self.has_cont_mask = True
    
    def _find_b_scans_directory(self, root_path):
        """Find directory containing B-scan images"""
        for dirpath, _, filenames in os.walk(root_path):
            if any(fname.lower().endswith(('.jpg', '.png')) for fname in filenames):
                return dirpath
        return None
    
    def _create_expanded_dataset(self):
        """
        Create an expanded dataset where each row corresponds to a single B-scan image.
        This duplicates tabular data rows to match each B-scan.
        """
        expanded_rows = []
        missing_count = 0
        found_count = 0
        
        for idx, row in self.original_df.iterrows():
            try:
                patient_id = int(row['Patient Number'])
                eye = row['Laterality']
                diagnosis_date = int(row['Diagnosis Date'])
                
                # Construct path to patient's images
                patient_path = os.path.join(self.image_root_dir, str(patient_id))
                if not os.path.isdir(patient_path):
                    missing_count += 1
                    continue
                
                eye_path = os.path.join(patient_path, eye)
                if not os.path.isdir(eye_path):
                    missing_count += 1
                    continue
                
                date_path = os.path.join(eye_path, str(diagnosis_date))
                if not os.path.isdir(date_path):
                    missing_count += 1
                    continue
                
                # Find B-scans directory using the improved function
                b_scans_path = self._find_b_scans_directory(date_path)
                if not b_scans_path:
                    missing_count += 1
                    continue
                
                # Record that we successfully loaded this volume
                volume_id = f"{patient_id}_{eye}_{diagnosis_date}"
                self.loaded_volume_ids.add(volume_id)
                
                # Get all B-scan images
                b_scan_images = []
                for img_name in os.listdir(b_scans_path):
                    if img_name.lower().endswith(('.jpg', '.png')):
                        img_path = os.path.join(b_scans_path, img_name)
                        b_scan_images.append(img_path)
                
                if not b_scan_images:
                    missing_count += 1
                    continue
                
                # Create a new row for each B-scan image
                for img_path in b_scan_images:
                    new_row = row.copy()
                    new_row['image_path'] = img_path
                    expanded_rows.append(new_row)
                
                found_count += 1
            except (ValueError, TypeError, KeyError) as e:
                missing_count += 1
                continue
        
        print(f"Created expanded dataset: {found_count} volumes matched with tabular data")
        print(f"Missing matches: {missing_count} tabular records could not be matched with images")
        print(f"Total expected volumes in label file: {len(self.expected_volume_ids)}")
        print(f"Volumes successfully loaded from image folders: {len(self.loaded_volume_ids)}")
        
        if expanded_rows:
            self.df = pd.DataFrame(expanded_rows)
        else:
            print("Warning: No matching B-scan images found. Using original dataset.")
            self.df = self.original_df.copy()
            self.df['image_path'] = None
            self.has_images = False
    
    def report_missing_volumes(self):
        """Print volumes that are in label file but not loaded due to missing images"""
        missing = self.expected_volume_ids - self.loaded_volume_ids
        print(f"Total expected volumes in label file: {len(self.expected_volume_ids)}")
        print(f"Volumes successfully loaded from image folders: {len(self.loaded_volume_ids)}")
        print(f"Missing volumes: {len(missing)}")
        for vid in list(sorted(missing)):
            print(" -", vid)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get categorical data
        X_categ = torch.tensor(self.df.iloc[idx][self.categorical_cols].values, dtype=torch.long)
        
        # Get continuous data with mask for missing values
        X_cont_values = self.df.iloc[idx][self.continuous_cols].values
        X_cont_values = X_cont_values.astype(np.float32)
        
        # Convert to float and handle NaN values
        X_cont = torch.tensor(X_cont_values, dtype=torch.float32)
        
        # Create mask for missing continuous values (1 for present, 0 for missing)
        if self.has_cont_mask:
            X_cont_mask = torch.tensor(~np.isnan(X_cont_values), dtype=torch.float32)
        
        # Replace NaN with 0 for the actual values
        X_cont = torch.nan_to_num(X_cont, nan=0.0)
        
        y = torch.tensor(self.df.iloc[idx][self.label_col], dtype=torch.long)
        
        result = {
            'categorical': X_categ,
            'continuous': X_cont,
            'label': y
        }
        
        if self.has_cont_mask:
            result['continuous_mask'] = X_cont_mask
        
        # Get image data if available
        if self.has_images and 'image_path' in self.df.columns:
            img_path = self.df.iloc[idx]['image_path']
            if img_path and os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                    if self.transforms:
                        image = self.transforms(image)
                    result['image'] = image
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        return result
    
    def get_category_dims(self):
        """Return the number of unique values for each categorical column"""
        return [self.df[col].nunique() for col in self.categorical_cols]
    
    def get_label_map(self):
        """Return the mapping from encoded labels to original labels"""
        return self.encoders[self.label_col].classes_
    
    def get_num_classes(self):
        """Return the number of unique classes in the label column"""
        return len(self.encoders[self.label_col].classes_)
    
    def get_class_distribution(self):
        """Return the distribution of classes in the dataset"""
        return self.df[self.label_col].value_counts().sort_index()
