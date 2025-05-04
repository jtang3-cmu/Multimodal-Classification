import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# <-- Use your custom gated TabTransformer here
from gated_tabTransformer import TabTransformer

from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# 1.── Read two Excel files ──
df1 = pd.read_excel('vol_anno_new.xlsx')
df2 = pd.read_excel('vol_anno_ori.xlsx')

# 2.── Standardize column names: strip whitespace, lowercase, spaces → underscores ──
for df in (df1, df2):
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(' ', '_')
    )

# 3.── If df2 has 'age' column instead of 'age_at_visit', rename it ──
if 'age' in df2.columns and 'age_at_visit' not in df2.columns:
    df2.rename(columns={'age': 'age_at_visit'}, inplace=True)

# 4.── Frequency-encode 'icd_primary' across both dataframes and treat as a continuous feature ──
df_all = pd.concat([df1, df2], ignore_index=True)
freq = df_all['icd_primary'].value_counts()
for df in (df1, df2):
    df['icd_freq'] = df['icd_primary'].map(freq).fillna(0)

# 5.── Define categorical and continuous columns ──
cat_cols = [
    'laterality',
    'sex',
    'primary_dx_yn',
    'cigarettes_yn_final',
    'smoking_tob_use_name_final',
    'smokeless_tob_use_name_final',
    'tobacco_user_name_final',
    'alcohol_use_name_final',
    'ill_drug_user_name_final'
]
cont_cols = ['age_at_visit', 'va_continuous', 'icd_freq']

# 6.── Combine and prepare X, y ──
df = pd.concat([df1, df2], ignore_index=True)
df['y'] = df['stage'].astype('category').cat.codes
df = df[df['y'] >= 0].reset_index(drop=True)

# 6.1 Impute continuous features with mean
df[cont_cols] = SimpleImputer(strategy='mean').fit_transform(df[cont_cols])

# 6.2 Impute categorical features with mode and ordinal-encode
df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_cols] = enc.fit_transform(df[cat_cols])

X_cat = df[cat_cols].astype(int).values
X_cont = df[cont_cols].astype(float).values
y      = df['y'].values
num_classes = len(np.unique(y))

print("Number of classes:", num_classes)
print("Class mapping:", dict(enumerate(df['stage'].astype('category').cat.categories)))

# 7.── Stratified train-test split ──
Xc_tr, Xc_te, Xn_tr, Xn_te, y_tr, y_te = train_test_split(
    X_cat, X_cont, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 8.── Compute class weights ──
classes = np.unique(y_tr)
cw = compute_class_weight('balanced', classes=classes, y=y_tr)
class_weights = torch.tensor(cw, dtype=torch.float32)

# 9.── Create a weighted sampler to handle class imbalance ──
class_count       = np.bincount(y_tr)
weights_per_class = 1.0 / class_count
sample_weights    = weights_per_class[y_tr]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# 10.── Build DataLoader ──
def to_tensor(x, dtype):
    return torch.tensor(x, dtype=dtype)

bs = 64
train_ds = TensorDataset(
    to_tensor(Xc_tr, torch.long),
    to_tensor(Xn_tr, torch.float32),
    to_tensor(y_tr,  torch.long),
)
test_ds  = TensorDataset(
    to_tensor(Xc_te, torch.long),
    to_tensor(Xn_te, torch.float32),
    to_tensor(y_te,  torch.long),
)
train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler)
test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

# 11.── Model and loss ──
categories = [len(c) for c in enc.categories_]  # number of categories for each of 9 features
model = TabTransformer(
    categories=categories,
    num_continuous=len(cont_cols),
    dim=64, depth=4, heads=4,
    dim_out=num_classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 12.── Scheduler ──
scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

# 13.── Training loop ──
epochs = 200
loss_history = []

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for xc, xn, yy in train_dl:
        logits = model(xc, xn)
        loss   = criterion(logits, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dl)
    loss_history.append(avg_loss)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch:2d} — loss: {avg_loss:.4f} — lr: {current_lr:.6f}")

# 14.── Plot the loss curve ──
plt.figure(figsize=(12,4))
plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# 15.── Evaluation on test set ──
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xc, xn, yy in test_dl:
        out = model(xc, xn)
        all_pred.append(out.argmax(dim=1).cpu().numpy())
        all_true.append(yy.cpu().numpy())

pred = np.concatenate(all_pred)
true = np.concatenate(all_true)

print('Test accuracy:', accuracy_score(true, pred))
print(classification_report(true, pred))

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Assuming true and pred are already defined
classes = ['Early AMD', 'GA', 'Int AMD', 'Not AMD', 'Scar', 'Wet']
num_classes = len(classes)

# Compute confusion matrix
cm = confusion_matrix(true, pred, labels=range(num_classes))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest')

# Title and axis labels
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

# Tick marks and labels
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)

# Annotate cells with counts, switching text color based on background
thresh = cm.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        color = 'white' if cm[i, j] > thresh else 'black'
        ax.text(j, i, cm[i, j], ha='center', va='center', color=color)

fig.tight_layout()
plt.show()

import os
# Save the model weights to a 'weights' directory
os.makedirs('weights', exist_ok=True)
save_path = os.path.join('weights', 'gated_tabtransformer_weights_6classes.pth')
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")
