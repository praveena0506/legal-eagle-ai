import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 1. Positional Encoding (Unchanged) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# --- 2. Scaled Dot-Product Attention (Unchanged) ---
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    d_k = q.size(-1)
    scaled_attention_scores = matmul_qk / math.sqrt(d_k)
    if mask is not None:
        scaled_attention_scores = scaled_attention_scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scaled_attention_scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


# --- 3. Multi-Head Attention (Unchanged) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.h, self.d_k)
        return x.transpose(1, 2)

    def forward(self, q_in, k_in, v_in, mask):
        batch_size = q_in.size(0)
        q = self.w_q(q_in)
        k = self.w_k(k_in)
        v = self.w_v(v_in)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        context, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output


# --- 4. Position-wise Feed-Forward Network (UPDATED: GELU) ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 🔥 UPDATE 1: Replaced ReLU with GELU
        # Modern LLMs (BERT, GPT) use GELU for smoother gradient flow.
        self.activation = nn.GELU()

    def forward(self, x):
        intermediate = self.activation(self.w_1(x))
        intermediate = self.dropout(intermediate)
        output = self.w_2(intermediate)
        return output


# --- 5. Encoder Layer (UPDATED: Pre-LayerNorm) ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 🔥 UPDATE 2: Pre-LayerNorm Architecture
        # Original: x + Sublayer(x) -> Norm
        # Modern:   x + Sublayer(Norm(x))

        # --- Sub-layer 1: Self-Attention ---
        residual = x
        x_norm = self.norm1(x)  # Normalize BEFORE attention
        attn_output = self.self_attn(q_in=x_norm, k_in=x_norm, v_in=x_norm, mask=mask)
        x = residual + self.dropout1(attn_output)  # Add residual

        # --- Sub-layer 2: Feed-Forward ---
        residual = x
        x_norm = self.norm2(x)  # Normalize BEFORE feed-forward
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout2(ffn_output)  # Add residual

        return x


# --- 6. The Full Encoder (Unchanged logic, kept final norm) ---
class Encoder(nn.Module):
    def __init__(self, d_model, h, d_ff, N, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        # In Pre-LN, we need a final normalization at the end of the stack
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# --- 7. The Full Transformer Classifier (Unchanged) ---
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, h, d_ff, N, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, h, d_ff, N, dropout)
        self.classifier_head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_input, mask):
        x = self.embedding(x_input)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, mask)
        x_pooled = x.mean(dim=1)
        output = self.classifier_head(x_pooled)
        return output


def create_padding_mask(input_ids, pad_token_id):
    mask = (input_ids != pad_token_id)
    return mask.unsqueeze(1).unsqueeze(2)