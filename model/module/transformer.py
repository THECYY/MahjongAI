from py_utils import *
from model.module.self_atten import *

class TransformerEncoderLayer(nn.Module):

    def __init__(self, h_dim, forward_dim, head, dropout, act="relu"):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(h_dim)
        self.self_atten = SelfAttention(head, h_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(h_dim)
        self.linear1 = nn.Linear(h_dim, forward_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(forward_dim, h_dim)
        self.dropout3 = nn.Dropout(dropout)

        self.activatioin = get_activate_fn(act)

    # embeddings [seq(14), batch, h_dim]
    def forward(self, embeddings, padding_mask, training=True):
        norm1 = self.layer_norm1(embeddings)
        atten = self.self_atten(norm1, padding_mask, training)

        embeddings = embeddings + self.dropout1(atten)

        norm2 = self.layer_norm2(embeddings)
        forward = self.linear2(self.dropout2(self.activatioin(self.linear1(norm2))))
        aembedding = embeddings + self.dropout3(forward)
        return aembedding

class TransformerEncoder(nn.Module):
    
    def __init__(self, num_layers, h_dim, forward_dim, head, dropout, act):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(h_dim, forward_dim, head, dropout, act) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(h_dim)

    def forward(self, embeddings, padding_mask, training):
        embeddings = embeddings.transpose(0, 1)
        for layer in self.layers:
            embeddings = layer(embeddings, padding_mask, training)
        return self.layer_norm(embeddings).transpose(0, 1)

        