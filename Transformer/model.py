import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoders,
        num_decoders,
        src_vocab_size,
        tgt_vocab_size,
        max_seq_len,
        d_model,
        d_ff,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pe = PositionalEncoding(max_seq_len, d_model)

        self.encoder = Encoder(num_encoders, d_model, d_ff, num_heads, dropout)
        self.decoder = Decoder(num_decoders, d_model, d_ff, num_heads, dropout)

        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # embedding
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        # position embedding
        src = self.pe(src)
        tgt = self.pe(tgt)

        # EncoderLayer pass
        enc_output = self.encoder(src)

        # DecoderLayer pass
        dec_output = self.decoder(tgt, enc_output)

        # linear pass
        out = self.linear(dec_output)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.mha_norm(self.mha(x, mask) + x)
        out = self.ffn_norm(self.ffn(x) + x)
        return out


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for encoder in self.layers:
            x = encoder(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, num_heads, dropout=0.1) for _ in range(num_layers)]
        )

    def forward(self, x, enc_inp, mask=None):
        for decoder in self.layers:
            x = decoder(x, enc_inp, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.mmha = MultiHeadAttention(d_model, num_heads, dropout)
        self.mmha_norm = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_inp, mask=None):
        # masked self attention
        q = self.mmha_norm(self.mmha(x, mask) + x)

        # crossed attention
        k, v = enc_inp, enc_inp
        o = self.mha_norm(self.mha(q, k, v) + q)

        # feed forward
        out = self.ffn_norm(self.ffn(o) + o)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = d_model // num_heads
        self.heads = nn.ModuleList(
            [
                SelfAttention(d_model, self.att_size, dropout)
                for _ in range(self.num_heads)
            ]
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None, mask=None):
        """multi head attentions

        Args:
            x [batch, seq_len, d_model]:

        Returns:
            out [batch, seq_len, d_model]
        """
        mha = torch.concat(
            [attn_block(q, k, v, mask) for attn_block in self.heads], dim=-1
        )
        return self.linear(mha)


class SelfAttention(nn.Module):
    def __init__(self, d_model, output_dim, dropout=0.1):
        super().__init__()
        self.Wq = nn.Linear(d_model, output_dim)
        self.Wk = nn.Linear(d_model, output_dim)
        self.Wv = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None, mask=None):
        """Attention(Q,K,V) = softmax(Q @ K^T / √d_k) @ V
        Q (Query): "What am I looking for?"
        K (Key): "What information is available?"
        V (Value): "What information to return?"
        Args:
            x [batch, seq_len, d_model]: input to self attention
        returns:
            out [batch, seq_len, output_dim]
        """
        k = k if k is not None else q
        v = v if v is not None else q
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        dim_k = k.shape[-1]

        scores = torch.bmm(q, torch.transpose(k, 1, 2)) / math.sqrt(dim_k)
        if mask is not None:
            scores.masked_fill_(mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = weights @ v
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class PositionalEncoding(nn.Module):
    """
    PE(pos,2i)​=sin(pos/ (10000^(2i/dmodel)​​),
    PE(pos,2i+1)​=cos(pos/ 10000^(2i/dmodel​)​)

    div_term=e^(−ln(10000)⋅(2i/dmodel​)) = 10000^(−2i/dmodel​)

    Args:
        x [batch, seq_len, d_model]:
    """

    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model).unsqueeze(0)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(-1)
        i = torch.arange(0, d_model, 2)
        div_term = torch.exp(-math.log(10000) * i / d_model).unsqueeze(0)
        pe[:, :, 0::2] = torch.sin(pos * div_term)
        pe[:, :, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    d_model = 512  # embedding size
    d_ff = 2048
    seq_len = 100  # kind of context length (max token to attend)
    num_heads = 8  # number of heads in multi-head attention
    num_encoders = 2
    num_decoders = 2
    src_vocab_size = 10000  # vocabulary size
    tgt_vocab_size = 5000
    max_seq_len = 5000

    # s = SelfAttention(d_model, d_model)
    # inp = torch.randn(12, seq_len, d_model)
    # o = s(inp)

    # f = FeedForward(d_model)
    # inp = torch.randn(12, d_model)
    # o = f(inp)

    # mha = MultiHeadAttention(d_model, num_heads)
    # o = mha(inp)

    # EncoderLayer = EncoderLayer(d_model, d_ff, num_heads)
    # enc_out = EncoderLayer(inp)

    # DecoderLayer = DecoderLayer(d_model, d_ff, num_heads)
    # o = DecoderLayer(inp, enc_out)

    t = Transformer(
        num_encoders,
        num_decoders,
        src_vocab_size,
        tgt_vocab_size,
        max_seq_len,
        d_model,
        d_ff,
        num_heads,
    )
    src = torch.arange(500).unsqueeze(0)
    tgt = torch.arange(500).unsqueeze(0)
    o = t(src, tgt)
    print(o.shape)
