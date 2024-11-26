from py_utils import *

class SelfAttention(nn.Module):

    def __init__(self, head, h_dim, dropout):
        super().__init__()
        self.head = head
        self.qkv_linear = nn.Linear(h_dim, 3 * h_dim)
        self.dropout = dropout

    # embedding [seq_len, batch, h_dim]
    def forward(self, embedding, key_padding_mask=None, training=True):
        seq_len, batch_size, h_dim = embedding.size()
        head_dim = h_dim // self.head                # 每一个注意力头的维度
        scaling = float(head_dim) ** -0.5
        q, k, v = self.qkv_linear(embedding).chunk(3, dim=-1)
        q = q * scaling

        # [batch * head, seq_len, h_dim]
        q = q.contiguous().view(seq_len, batch_size * self.head, head_dim).transpose(0, 1)
        k = k.contiguous().view(seq_len, batch_size * self.head, head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_len, batch_size * self.head, head_dim).transpose(0, 1)

        # [batch, head, seq_len, seq_len]
        attn_output_weight = torch.bmm(q, k.transpose(1, 2))

        if key_padding_mask is not None:
            key_padding_mask[:, -1] = 0
            attn_output_weight = attn_output_weight.view(batch_size, self.head, seq_len, seq_len)
            key_padding_mask = torch.logical_not(key_padding_mask)
            attn_output_weight = attn_output_weight.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weight = attn_output_weight.view(batch_size * self.head, seq_len, seq_len)

        attn_output_weight_soft = F.softmax(attn_output_weight, dim=-1)
        attn_output_weight_soft = F.dropout(attn_output_weight_soft, p=self.dropout, training=training)
        attn_output_weight_soft = attn_output_weight_soft.masked_fill(
            torch.isnan(attn_output_weight_soft),
            float(0)
        )

        # v [seq_len, batch_size * head, head_dim]
        # attn_output_weight [batch_size * head, seq_len, seq_len]
        attn_output = torch.bmm(attn_output_weight_soft, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, batch_size, h_dim)
        
        return attn_output
