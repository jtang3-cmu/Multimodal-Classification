# === New function: Fine-tuning TabTransformer on tabular data only ===
def tune_tab_transformer_model(args, device):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tab_transformer_pytorch import TabTransformer
    from tqdm import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    # Define feature columns for AMD tabular data
    categorical_cols_amd = [
        'Laterality', 'SEX', 'CIGARETTES_YN', 'SMOKING_TOB_USE_NAME',
        'SMOKELESS_TOB_USE_NAME', 'TOBACCO_USER_NAME', 'ALCOHOL_USE_NAME',
        'ILL_DRUG_USER_NAME', 'PRIMARY_DX_YN', 'CURRENT_ICD10_LIST'
    ]
    continuous_cols_amd = ['Age', 'VA']
    label_col = 'stage'

    # Load data (using the provided tabular data file)
    df = pd.read_excel(args.tab_data_path)
    # Fill continuous columns with mean
    for col in continuous_cols_amd:
        df[col] = df[col].fillna(df[col].mean())
    # Fill categorical columns with mode
    for col in categorical_cols_amd:
        df[col] = df[col].fillna(df[col].mode()[0])
    amd_df = df.copy()

    # Encode categorical data
    encoders = {}
    for col in categorical_cols_amd:
        le = LabelEncoder()
        amd_df[col] = le.fit_transform(amd_df[col].astype(str))
        encoders[col] = le

    # Encode label separately
    label_le = LabelEncoder()
    amd_df[label_col] = label_le.fit_transform(amd_df[label_col])
    encoders[label_col] = label_le

    # Train/test split
    train_df, test_df = train_test_split(amd_df, test_size=0.2, stratify=amd_df[label_col], random_state=42)

    # Get category dimensions and number of classes
    category_dims_amd = [len(encoders[col].classes_) for col in categorical_cols_amd]
    num_classes = amd_df[label_col].nunique()

    # Define local Dataset for tabular data
    class AMDDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.X_categ = torch.tensor(df[categorical_cols_amd].values, dtype=torch.long)
            self.X_cont = torch.tensor(df[continuous_cols_amd].values, dtype=torch.float32)
            self.y = torch.tensor(df[label_col].values, dtype=torch.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X_categ[idx], self.X_cont[idx], self.y[idx]

    train_dataset = AMDDataset(train_df)
    test_dataset = AMDDataset(test_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.tab_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.tab_batch_size)

    # Initialize TabTransformer
    model = TabTransformer(
        categories=category_dims_amd,
        num_continuous=len(continuous_cols_amd),
        dim=32,
        depth=4,
        heads=8,
        dim_out=num_classes,
        use_shared_categ_embed=False
    ).to(device)

    # Load pretrained weights (encoder only)
    try:
        pretrained_dict = torch.load("tab_transformer_heart.pth", map_location=device)
        transformer_weights = {k: v for k, v in pretrained_dict.items() if k.startswith("transformer.")}
        model.load_state_dict(transformer_weights, strict=False)
        print(f"Loaded pretrained transformer weights: {len(transformer_weights)} keys")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fine-tuning loop
    epochs = args.tab_epochs
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for x_categ, x_cont, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_categ, x_cont)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1} - Loss: {running_loss:.4f} - Accuracy: {acc:.4f}")

    # Evaluation on test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_categ, x_cont, y in test_loader:
            x_categ, x_cont = x_categ.to(device), x_cont.to(device)
            logits = model(x_categ, x_cont)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=label_le.classes_))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_le.classes_, yticklabels=label_le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Save the trained model
    model_save_path = "tab_transformer_tuned.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained TabTransformer model saved to {model_save_path}")