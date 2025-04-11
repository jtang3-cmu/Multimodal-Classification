import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel
from tab_transformer_pytorch import TabTransformer
import sys
sys.path.append('RETFound_MAE')  # Adjust path as needed
from models_vit import RETFound_mae

class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=2048):
        super(ResNet50Encoder, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer
        self.output_dim = output_dim

    def forward(self, x):
        return self.resnet(x)
    
class RETFoundEncoder(nn.Module):
    def __init__(self, pretrained=True, weights_path="RETFound_MAE/RETFound_mae_natureOCT.pth"):
        super(RETFoundEncoder, self).__init__()
        self.model = RETFound_mae(img_size=224, num_classes=0)
        
        if pretrained:
            # Load pretrained weights
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
            state_dict = {k: v for k, v in checkpoint['model'].items() 
                         if not k.startswith("decoder") and "mask_token" not in k}
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Loaded RETFound MAE pretrained weights from {weights_path}")
        
        self.output_dim = 1024  # RETFound output dimension

    def forward(self, x):
        return self.model(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        return self.layer_norm(query + attn_output)

class MultiModalFusionBERT(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        return self.layer_norm(query + attn_output)

class MultiModalFusionBERT(nn.Module):
    def __init__(self, category_dims, num_continuous, image_feature_dim=2048, tab_feature_dim=64,
                 hidden_dim=768, num_heads=8, num_classes=6, finetune_last_bert_layer=False,
                 image_encoder_type='resnet50'):
        super().__init__()
        
        # Select image encoder based on type
        if image_encoder_type == 'resnet50':
            self.image_encoder = ResNet50Encoder(pretrained=True)
        elif image_encoder_type == 'retfound':
            self.image_encoder = RETFoundEncoder(pretrained=True)
        else:
            raise ValueError(f"Unknown image encoder type: {image_encoder_type}")
        
        self.image_fc = nn.Linear(self.image_encoder.output_dim, hidden_dim)

        # Properly initialize TabTransformer with the correct parameters
        self.tab_transformer = TabTransformer(
        categories=category_dims,
        num_continuous=num_continuous,
        dim=32,
        depth=4,
        heads=8,
        dim_out=tab_feature_dim
        )

        self.tab_fc = nn.Linear(tab_feature_dim, hidden_dim)
        self.cross_attn_fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads)

        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        if finetune_last_bert_layer:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            for param in self.bert_encoder.encoder.layer[-1].parameters():
                param.requires_grad = True
            if hasattr(self.bert_encoder, 'pooler'):
                for param in self.bert_encoder.pooler.parameters():
                    param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, categorical, continuous, continuous_mask=None):
        # Process image
        image_features = self.image_encoder(image)
        img_embed = self.image_fc(image_features).unsqueeze(1)

        # Process tabular data - note the TabTransformer only takes categorical and continuous inputs
        tab_features = self.tab_transformer(categorical, continuous)
        tab_embed = self.tab_fc(tab_features).unsqueeze(1)

        # Fusion and BERT processing
        fused = self.cross_attn_fusion(query=tab_embed, key=img_embed, value=img_embed)
        combined_seq = torch.cat([fused, tab_embed], dim=1)
        attention_mask = torch.ones(combined_seq.size()[:2], dtype=torch.long, device=combined_seq.device)
        bert_output = self.bert_encoder(inputs_embeds=combined_seq, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]

        return self.classifier(cls_token)

# Model creation functions
def create_model(args, dataset):
    if args.model_type == 'multimodal':
        return create_multimodal_model(args, dataset)
    elif args.model_type == 'image_only':
        return create_image_model(args, dataset)
    elif args.model_type == 'tabular_only':
        return create_tabular_model(args, dataset)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

def create_multimodal_model(args, dataset):
    category_dims = dataset.get_category_dims()
    num_continuous = len(dataset.continuous_cols)
    num_classes = dataset.get_num_classes()
    
    model = MultiModalFusionBERT(
        category_dims=category_dims,
        num_continuous=num_continuous,
        image_feature_dim=2048 if args.image_encoder_type == 'resnet50' else 1024,
        tab_feature_dim=args.tab_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=num_classes,
        finetune_last_bert_layer=True,
        image_encoder_type=args.image_encoder_type
    )

    # Load pre-trained weights
    try:
        pretrained_weights = torch.load('tab_transformer_heart.pth', map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        
        # Create a mapping from pretrained keys to model keys
        # This maps the standalone TabTransformer keys to the nested structure
        pretrained_dict = {}
        for k, v in pretrained_weights.items():
            # Create the new key with the proper prefix
            new_key = f"transformer.{k}"
            # Only add if the key exists in the model and shapes match
            if new_key in model_dict and model_dict[new_key].shape == v.shape:
                pretrained_dict[new_key] = v
        
        # Update model with filtered weights
        if pretrained_dict:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")
        else:
            print("No matching layers found in pretrained weights")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")

    if args.freeze_encoders:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        for param in model.tab_transformer.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model


def create_image_model(args, dataset):
    num_classes = dataset.get_num_classes()
    
    if args.image_encoder_type == 'resnet50':
        encoder = ResNet50Encoder(pretrained=True)
        output_dim = 2048
    elif args.image_encoder_type == 'retfound':
        encoder = RETFoundEncoder(pretrained=True)
        output_dim = 1024
    else:
        raise ValueError(f"Unknown image encoder type: {args.image_encoder_type}")
    
    model = nn.Sequential(
        encoder,
        nn.Linear(output_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    
    if args.freeze_encoders:
        for param in model[0].parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model

def create_tabular_model(args, dataset):
    category_dims = dataset.get_category_dims()
    num_continuous = len(dataset.continuous_cols)
    num_classes = dataset.get_num_classes()

    # === initialize TabTransformer ===
    tab_transformer = TabTransformer(
        categories=category_dims,
        num_continuous=num_continuous,
        dim=32,
        depth=4,
        heads=8,
        dim_out=args.tab_dim
    )

     # Create classifier head
    classifier = nn.Sequential(
        nn.Linear(args.tab_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    
    # Create the full model
    class TabularModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
            
        def forward(self, x_categ, x_cont):
            features = self.encoder(x_categ, x_cont)
            return self.classifier(features)
    
    model = TabularModel(tab_transformer, classifier)
    
    try:
        pretrained_weights = torch.load('tab_transformer_heart.pth', map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        
        # Create a mapping from pretrained keys to model keys
        # This maps the standalone TabTransformer keys to the nested structure
        pretrained_dict = {}
        for k, v in pretrained_weights.items():
            # Create the new key with the proper prefix
            new_key = f"transformer.{k}"
            # Only add if the key exists in the model and shapes match
            if new_key in model_dict and model_dict[new_key].shape == v.shape:
                pretrained_dict[new_key] = v
        
        # Update model with filtered weights
        if pretrained_dict:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")
        else:
            print("No matching layers found in pretrained weights")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
    
    if args.freeze_encoders:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model

