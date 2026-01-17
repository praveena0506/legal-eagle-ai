import torch
import torch.nn as nn
import math


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, h=4, d_ff=256, N=2, num_classes=2, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=d_ff,
            batch_first=True,
            activation='gelu',  # Modern activation
            norm_first=True  # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N)

        # Output Head
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Transformer expects mask: True = Ignore
        # We might need to invert PyTorch's default mask behavior depending on version,
        # but usually key_padding_mask handles True as padding.
        x = self.transformer(x, src_key_padding_mask=mask)

        # Global Average Pooling (take average of all words)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


def create_padding_mask(seq, pad_token_id=0):
    # Create mask where Pad Token is True (Transformer ignores True)
    return (seq == pad_token_id)