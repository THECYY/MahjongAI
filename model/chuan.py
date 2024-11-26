from py_utils import *
from model.encoder import *

class ChuanCardPlayAI(nn.Module):

    def __init__(self, h_dim, transformer_layers, table_dim, river_dim, c_dim, head, dropout, act, decoder_layers, player_count=4):
        super().__init__()
        self.hand_cards_encoder = MCsEncoder(27, h_dim, transformer_layers, 2 * h_dim, head, dropout, act)
        self.other_table_card_encoder = CardGroupEncoder(table_dim, 27, 2, 4, dropout, act)
        self.self_table_card_encoder = CardGroupEncoder(table_dim, 27, 2, 4, dropout, act)
        self.other_card_river_encoder = MCsEncoder(27, river_dim, transformer_layers, river_dim * 2, head, dropout, act)
        self.self_river_encoder = MCsEncoder(27, river_dim, transformer_layers, river_dim * 2, head, dropout, act)
        self.won_encoder = nn.Embedding(2, 8)

        self.card_river_mlp = mlp(2, act, river_dim, river_dim, 2 * river_dim, False, True)
        self.table_card_mlp = mlp(2, act, table_dim, table_dim, 2 * table_dim, False, True)
        self.other_concate_mlp = mlp(2, act, (player_count - 1) * (table_dim + river_dim + 8), c_dim, (player_count - 1) * (table_dim + river_dim + 8), False, True)
        self.self_concate_mlp = mlp(2, act, table_dim + river_dim, c_dim, table_dim + river_dim, False, True)

        self.mlp = mlp(3, act, c_dim * 2 + h_dim, h_dim, h_dim)

        self.decoder = CardPlayMLPDecoder(act, h_dim, decoder_layers)
        
    # hand_cards的seq_len为14
    # table_cards的seq_len为4
    # card_river的seq_len为 (36 * 3 - 2 * (13 + 14)) / 2 == 27
    
    # hand_mcs                      [batch, 14]
    # hand_mcs_padding_mask         [batch, 14]

    # table_cards_types             [batch, 3, 4]
    # table_cards_mcs               [batch, 3, 4, 3]
    # table_cards_padding_mask      [batch, 3, 4]

    # self_table_cards_types        [batch, 4]
    # self_table_cards_mcs          [batch, 4, 3]
    # self_table_cards_padding_mask [batch, 4]

    # other_card_rivers             [batch, 3, 27]
    # other_card_river_padding_mask [batch, batch, 3, 27]

    # self_card_rivers              [batch, 27]
    # self_card_rivers_padding_mask [batch, 27]
    def forward(
            self, 
            hand_mcs,             
            hand_mcs_padding_mask,        
            table_cards_types,            
            table_cards_mcs,              
            table_cards_padding_mask,     
            self_table_cards_types,       
            self_table_cards_mcs,         
            self_table_cards_padding_mask,
            other_card_rivers,            
            other_card_river_padding_mask,
            self_card_rivers,             
            self_card_rivers_padding_mask,
            won,        # [batch, 3]     
            training
        ):
        # player num
        batch_mul_player = table_cards_mcs.shape[0] * table_cards_mcs.shape[1]
        batch = table_cards_mcs.shape[0]
        other_player = table_cards_mcs.shape[1]

        # embeddings 
        # [batch, h_dim]
        h_emb = self.hand_cards_encoder(hand_mcs, hand_mcs_padding_mask, training)
        ott = table_cards_types.reshape([batch_mul_player, table_cards_types.shape[2]])
        otmc = table_cards_mcs.reshape([batch_mul_player, table_cards_mcs.shape[2], table_cards_mcs.shape[3]])
        otm = table_cards_padding_mask.reshape([batch_mul_player, table_cards_padding_mask.shape[2]])
        ot_emb = self.other_table_card_encoder(ott, otmc, otm, training)
        
        # [batch, other_player, table_dim]
        ot_emb = ot_emb.reshape([batch, other_player, ot_emb.shape[1]])

        # [batch, table_dim]
        st_emb = self.self_table_card_encoder(self_table_cards_types, self_table_cards_mcs, self_table_cards_padding_mask, training)
        
        ori = other_card_rivers.reshape([batch_mul_player, other_card_rivers.shape[2]])
        orim = other_card_river_padding_mask.reshape([batch_mul_player, other_card_river_padding_mask.shape[2]])
        ori_emb = self.other_card_river_encoder(ori, orim, training)
        
        
        
        # [batch, other_player, river_dim]
        ori_emb = ori_emb.reshape([batch, other_player, ori_emb.shape[1]])
        
        # [batch, river_dim]
        sri_emb = self.self_river_encoder(self_card_rivers, self_card_rivers_padding_mask, training)

        # [batch, 3, c_dim]
        w_emb = self.won_encoder(won)

        # MLP
        ori_emb = self.card_river_mlp(ori_emb)
        sri_emb = self.card_river_mlp(sri_emb)
        ot_emb = self.table_card_mlp(ot_emb)
        st_emb = self.table_card_mlp(st_emb)

        embeddings = []
        for i in range(other_player):
            embeddings.append(ori_emb[:, i])
            embeddings.append(ot_emb[:, i])
            embeddings.append(w_emb[:, i])
        other_embeddings = torch.concatenate(embeddings, dim=1)
        self_embeddings = torch.concatenate([sri_emb, st_emb], dim=-1)

        others = self.other_concate_mlp(other_embeddings)
        selfs = self.self_concate_mlp(self_embeddings)
        z = torch.concatenate([h_emb, others, selfs], dim=-1)

        z = self.mlp(z)

        logistic = self.decoder(z)

        if torch.any(torch.isnan(logistic)):
            raise Exception("Nan")
        return logistic
