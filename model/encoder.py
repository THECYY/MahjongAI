from py_utils import *
from model.module.transformer import *

class MCsEncoder(nn.Module):

    def __init__(self, n_mc, h_dim, layers, forward_dim, head, dropout, act):
        super().__init__()
        self.mcs_embedding = nn.Embedding(n_mc, h_dim)
        self.transformer_encoder = TransformerEncoder(layers, h_dim, forward_dim, head, dropout, act)
        
    def forward(self, mcs, padding_mask, training):
        embeddings = self.mcs_embedding(mcs)
        trans_emb = self.transformer_encoder(embeddings, padding_mask, training)
        
        # embeddings [batch, seq_len, h_dim]
        # padding [batch, seq_len]
        # z [batch, h_dim]
        z = trans_emb * padding_mask.unsqueeze(-1)
        mask = padding_mask.sum(dim=-1).unsqueeze(-1)
        mask[torch.where(mask == 0)] = 1
        z = z.sum(dim = 1) / mask
        return z 

# Eat, Bump, Gang1, Gang4
class CardGroupEncoder(nn.Module):

    def __init__(self, h_dim, n_mc, transformer_layers, head, dropout, act):
        super().__init__()
        self.mc_embedding = nn.Embedding(n_mc, h_dim)
        self.type_embedding = nn.Embedding(4, h_dim)
        self.transformer_encoder = TransformerEncoder(transformer_layers, h_dim, h_dim, head, dropout, act)

    # types [batch, 4] 
    # mask [batch, 4]
    # mcss [batch, 4, 3]
    def forward(self, types, mcss, padding_mask, training):
        type_embs = self.type_embedding(types) # [batch, 4, h_dim]
        mc_embs = self.mc_embedding(mcss) # [batch, 4, 3, h_dim]
        embeddings = mc_embs.sum(dim=2) * type_embs # [batch, 4, h_dim]
        embeddings = self.transformer_encoder(embeddings, padding_mask, training)
        
        # [batch, seq_len, h_dim]
        z = embeddings * padding_mask.unsqueeze(-1)
        mask = padding_mask.sum(dim=-1).unsqueeze(-1)
        mask[torch.where(mask == 0)] = 1
        z = z.sum(dim = 1) / mask
        return z

class CardPlayMLPDecoder(nn.Module):
    def __init__(self, act, h_dim, layers):
        super().__init__()
        self.mlp = mlp(layers, act, h_dim, 14, 2 * h_dim, False)

    # [batch, 14]
    def forward(self, z):
        z = self.mlp(z)
        return z