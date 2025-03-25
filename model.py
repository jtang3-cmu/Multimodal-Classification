import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
import timm
# ===============================
# 1. Image Model (RETFound Encoder)
# ===============================
class RETFoundEncoder(nn.Module):
    def __init__(self, model_name="timm/retfound"):
        super(RETFoundEncoder, self).__init__()
        # Load the RETFound model and remove the classification head
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.feature_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)  # Return image features

# ===============================
# 2. Tabular Data Model (TabTransformer)
# ===============================
class TabTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ===============================
# 3. Cross-Attention Fusion Module
# ===============================
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        # Use PyTorch's MultiheadAttention to implement cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        """
        query: [batch, seq_len, hidden_dim] - e.g., tabular features
        key, value: [batch, seq_len, hidden_dim] - e.g., image features
        """
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        fused = self.layer_norm(query + attn_output)
        return fused

# ===============================
# 4. Multi-Modal Model: First perform cross-attention fusion,
#    then use a pre-trained/fine-tuned BERT as the decoder
# ===============================
class MultiModalFusionBERT(nn.Module):
    def __init__(self, image_feature_dim, tabular_feature_dim, hidden_dim=768, num_heads=8, num_classes=2):
        super(MultiModalFusionBERT, self).__init__()
        # Map features from each modality to the same dimension
        self.image_fc = nn.Linear(image_feature_dim, hidden_dim)
        self.tabular_fc = nn.Linear(tabular_feature_dim, hidden_dim)
        
        # Cross-attention fusion module: using tabular features as query and image features as key/value
        self.cross_attn_fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Use a pre-trained BERT as the decoder (load pre-trained weights)
        self.bert_decoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Classifier head: based on the [CLS] token output from BERT decoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image_features, tabular_features):
        # 1. Map both modalities to the same hidden_dim
        img_embed = self.image_fc(image_features)   # [batch, hidden_dim]
        tab_embed = self.tabular_fc(tabular_features) # [batch, hidden_dim]
        
        # 2. Expand vectors to sequences: treat each as a single token
        # Here, the tabular token acts as query, and the image token acts as key/value
        tab_seq = tab_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        img_seq = img_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # 3. Fuse features via cross-attention: let tabular info attend to image info
        fused_tab = self.cross_attn_fusion(query=tab_seq, key=img_seq, value=img_seq)  # [batch, 1, hidden_dim]
        
        # 4. Construct an input sequence for the BERT decoder (e.g., concatenate the fused token with the original tabular token)
        combined_seq = torch.cat([fused_tab, tab_seq], dim=1)  # [batch, 2, hidden_dim]
        batch_size, seq_length, _ = combined_seq.size()
        attention_mask = torch.ones(batch_size, seq_length, device=combined_seq.device, dtype=torch.long)
        
        # 5. Pass through the pre-trained BERT decoder
        bert_outputs = self.bert_decoder(inputs_embeds=combined_seq, attention_mask=attention_mask)
        # Take the first token's output as the global [CLS] representation
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
        
        # 6. Classification
        logits = self.classifier(cls_output)
        return logits
