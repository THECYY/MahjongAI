from py_utils import *
from mahjong.players import *
from mahjong.chuan_majong import *
from mahjong.sts4_mahjong import *

class ChuanProxyMajong(ChuanMajong):

    def o_init_round_basic_props(self):
        super().o_init_round_basic_props()
        self.tensor_record = [[] for _ in range(self.max_player)]
        self.init_score = [score for score in self.scores]

    def _on_someone_is_win(self, winner_list, items, sources):
        scroes = super()._on_someone_is_win(winner_list, items, sources)
        
        for winner in winner_list:
            for record in self.tensor_record[winner]:
                record["win"] = 1
                record["score"] = scroes[winner]
        
        # 如果不是自摸，将惩罚
        self.tensor_record[sources][-1]["win"] = 0
        self.tensor_record[sources][-1]["listen"] = 1
        self.tensor_record[sources][-1]["score"] = scroes[sources]
        return scroes

    def o_end_round(self):
        super().o_end_round()
        for i in range(self.max_player):
            if self.not_listen_score[i] < 0:
                for record in self.tensor_record[i]:
                    record["win"] = 0
                    record["listen"] = 0
                    record["score"] = self.not_listen_score[i]
    

class PlayerProxy(Player):

    def __init__(self, player):
        super().__init__(player.name)
        self.player = player
    
    def give_cards(self):
        self.player.hand_cards = self.hand_cards
        self.player.table_cards = self.table_cards
        self.player.card_river = self.card_river

    def o_make_decision_to_change_three(self, cardss):
        self.give_cards()
        return self.player.o_make_decision_to_change_three(cardss)

    def o_make_decision_to_play(self, game:MGame, playable_cards):
        self.give_cards()
        index = self.player.o_make_decision_to_play(game, playable_cards)
        play_data = game_shotcum_to_play_card_data(game)
        play_data["index"] = index
        play_data["play_lack"] = MCType.is_type(playable_cards[index].mc, self.player.lack)
        self_index = game._find_player(self)
        game.tensor_record[self_index].append(play_data)
        return index

    def o_make_decision_to_cpg(self, game, card, choice, eat, dump, gang):
        self.give_cards()
        return self.player.o_make_decision_to_cpg(game, card, choice, eat, dump, gang)

    def o_make_decision_to_gang4(self, game, gangs):
        self.give_cards()
        return self.player.o_make_decision_to_gang4(game, gangs)

    def o_make_decision_to_lack(self, game):
        self.give_cards()
        return self.player.o_make_decision_to_lack(game)

    def o_make_decision_to_win(self, game, win_types, package):
        self.give_cards()
        return self.player.o_make_decision_to_win(game, win_types, package)

    def o_make_decision_to_stand(self, game, playable_cards):
        self.give_cards()
        return self.player.o_make_decision_to_stand(game, playable_cards)

def data_mask(data, mask):
    return data[mask]

def record_view(record):
    print("==========================")
    print("Your hand: {}".format(MC.to_str(list(data_mask(record["hand"], record["hand_mask"])))))
    for i in range(int(record["table_mask"].sum())):
        mcs = list(record["table_mcs"][i])
        print("Your {} {}".format(Op.str(int(record["table_types"][i])), MC.to_str(mcs)))
    print("Your card river: {}".format(MC.to_str(list(data_mask(record["river"], record["river_mask"])))))
    
    for i in range(record["other_table_mask"].shape[0]): # 3玩家
        print("Player {} river {}".format(i + 1, MC.to_str(list(data_mask(record["other_river"][i], record["other_river_mask"][i])))))
        for k in range(record["other_table_mask"][i].sum()):
            mcs = list(record["other_table_mcs"][i][k])
            print("Player {} {} {}".format(i + 1, record["other_table_types"][i][k], MC.to_str(mcs)))
    print("You choose {}".format(record["index"]))
    print("Score {}".format(record["score"]))

def get_game(game_type, name, logger):
    if game_type == "chuan":
        config = ChuanConfig()
        config.max_iter = 1
        game = ChuanProxyMajong(config, name)
    elif game_type == "sts4":
        print("No such game")
    else:
        print("No such game")
    game.set_logger(logger)
    return game

# 每一场就作为一个数据供应
def generate_chuan_data(cfg, base_dir, players, count, log_dir):
    print("Generate chuan majong data")
    data_dir = base_dir
    make_sure_dir(data_dir)
    players = [PlayerProxy(player) for player in players]

    # 开始生成数据
    total_data = 0
    importent_path = logger_path = "{}/game_record.txt".format(log_dir)
    if os.path.exists(importent_path):
        os.remove(importent_path)
    while True:

        # 启动一个game, 并完成游戏
        game_name = "game_{}".format(int_to_fix_len_str(total_data, 8, "0"))
        logger_path = "{}/{}.txt".format(log_dir, game_name)
        logger = Logger(logger_path, True)
        game = get_game(cfg.game_type, game_name, logger)
        for player in players:
            game.add_player(player)
        game.start_game()
        for player in players:
            player.clear_cards()

        # 查找所有有用纪录并保存
        records = []
        for i in range(game.max_player):
            for record in game.tensor_record[i]:
                if record.get("win") is not None and not record.get("play_lack"):
                    if not cfg.only_win:
                        records.append(record)
                    elif record.get("win") or record.get("listen"):
                        records.append(record)
        if len(records) != 0:
            save_path = data_dir + "/{}.pt".format(game_name)
            torch.save(records, save_path)
            logger.log_to_file(logger_path, True)
            logger.importent_log_to_file(importent_path)
            total_data += 1
            #for record in records:
            #    record_view(record)
        if total_data == count:
            break

def load_chuan_data(dir):
    data_dir = dir
    files = os.listdir(data_dir)
    datas = []
    for file in files:
        data = torch.load(data_dir + "/" + file)
        datas.append(data)
    return datas
    
