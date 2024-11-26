from py_utils import *
from majong_trainer import *
from model.chuan import *
from chuan_data_generator import *
from mahjong.players import *
from dataset.ChuanDataset import *
from mahjong.players import *

class ChuanMajongAITrainer(MajongFlowTrainer):

    def __init__(self, cfg):
        super().__init__("MajongAI", "chuan_play", cfg)
        
    def init_players(self):
        ai_player = ChuanAIPlayer("cyy_ai")
        ai_player.mate_play_brain(self.model)
        self.players = [
            ChuanHeuristicPlayer("lyx_heruistic"),
            ChuanPurityColorPlayer("wtf_purity"),
            ChuanSevenPairsPlayer("lm_seven"),
            ai_player
        ]

    def init_model(self):
        self.model = ChuanCardPlayAI(self.cfg.h_dim, self.cfg.transformer_layers, \
            self.cfg.t_dim, self.cfg.r_dim, self.cfg.c_dim, 4, 0, "relu", 4, 4)
        self.init_players()
        self.softmax = nn.Softmax(dim=-1)
    
    def calculate_loss(self, data, output):
        _, hm, _, _, _, _, _, _, _, _, _, _, _, scores, indices = data
        total_loss = 0
        for i in range(scores.shape[0]):
            count = hm[i].sum()
            label = torch.ones([count], dtype=torch.float32, device=self.device)
            if scores[i] > 0:
                label = label * 0.2 / (count - 1)
                label[int(indices[i])] = 0.8
            else:
                label = label / (count - 1)
                label[int(indices[i])] = 0
            out = output[i][:count]
            pred = self.softmax(out)
            loss = F.cross_entropy(label, pred)
            total_loss += loss * abs(int(scores[i])) / 20
        return total_loss, {"loss": total_loss.item()}

    def forward(self, data):
        h, hm, tt, tmc, tm, r, rm, ott, otmc, otm, ori, orm, won, _, _ = data
        output = self.model(h, hm, ott, otmc, otm, tt, tmc, tm, ori, orm, r, rm, won,True)
        return output

    def init_flow_dataset(self):
        memory = "memory_{}_{}".format(int_to_fix_len_str(self.round // 1000, 8, "0"), int_to_fix_len_str(self.round % 1000, 4, "0"))
        base_dir = "./data/{}/{}/{}".format(self.cfg.ai_name, self.app_name, memory)
        logger_dir = "./log/{}/{}/{}".format(self.cfg.ai_name, self.app_name, memory)
        make_sure_dir(logger_dir)
        make_sure_dir(base_dir)
        file_count = len(os.listdir(base_dir))
        if file_count < self.cfg.flow_size:
            generate_chuan_data(self.cfg, base_dir, self.players, self.cfg.flow_size, logger_dir)
        self.flow_dataset = ChuanDataset(self.cfg, base_dir)
        self.flow_dataloader = self.flow_dataset.get_dataloader(1)

    def eval(self):
        print("=========== eval ai ==========")
        game_name = "{}_eval_{}".format(self.cfg.ai_name, self.round // 1000)
        logger_path = "eval/{}/{}.txt".format(self.cfg.ai_name, game_name)
        make_sure_dir_of_file(logger_path)
        config = ChuanConfig()
        config.max_iter = 8
        game = ChuanMajong(config, game_name)
        players = [ChuanAIPlayer("{}_{}".format(self.cfg.ai_name, i)) for i in range(config.max_player)]
        for player in players:
            player.mate_play_brain(self.model)
            game.add_player(player)
        logger = Logger(logger_path, True)
        game.set_logger(logger)
        game.start_game()
        print("========== end eval ==========")

if __name__ == "__main__":
    cfg = get_cfg()
    trainer = ChuanMajongAITrainer(cfg)
    trainer.train()