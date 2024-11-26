from py_utils import *
from .game import *

class Sts4Rule(Rule):
    ops_for_card = [Op.Bump, Op.Gang1, Op.Eat]
    ops_in_turn = [Op.Play, Op.Gang4, Op.Stand]

    activate_winable_types = [
        WinableType.Stand,                  # √
        WinableType.Broken_Unitary_Nine,    # √    
        WinableType.Clear_River_Self,       # √
        WinableType.Self_Wind,              # √
        WinableType.Ground_Wind,            # √
        WinableType.Zhong,                  # √
        WinableType.Bai,                    # √
        WinableType.Fa,                     # √
        WinableType.Avg,                    # √
        WinableType.One_Cup,                # √
        WinableType.Grab_Gang,              # √
        WinableType.Bloom,                  # √
        WinableType.Torch_Moon,             # √        
        WinableType.Torch_Fish,             # √
        WinableType.Once,                   # √
        WinableType.W_Stand,                # √         
        WinableType.Bump3,                  # √
        WinableType.Three_Gang,             # √               
        WinableType.All_Bump,               # √
        WinableType.Three_Hidden_Bump,      # √     
        WinableType.Small_Yuan,             # √
        WinableType.Mix_Older,              # √
        WinableType.Seven_Pairs,            # √
        WinableType.Mix_Unitary_Nine,       # √
        WinableType.One_To_Nine,            # √
        WinableType.Avg3,                   # √
        WinableType.Two_Cup,                # √
        WinableType.Purity_Unitary_Nine,    # √ 
        WinableType.Mix_Color,              # √
        WinableType.Purity_Color,           # √ 
        WinableType.Flow,                     
        WinableType.Sky,                    # √
        WinableType.Land,                   # √
        WinableType.Big_Yuan,               # √ 
        WinableType.Words,                  # √
        WinableType.Green,                  # √ 
        WinableType.Purity_Old,             # √ 
        WinableType.All_Unitary_Nine,       # √ 
        WinableType.Small_Wind,             # √
        WinableType.Four_Gang,              # √ 
        WinableType.Flower,                 # √ 
        WinableType.Four_Hidden_Bump,       # √
        WinableType.Four_Hidden_Bump_Single,# √
        WinableType.All_Unitary_Nine_Full,  # √
        WinableType.Purity_Flower,          # √ 
        WinableType.Big_Wind                # √
    ]

    times_map = {
        WinableType.Stand: 1,        
        WinableType.Broken_Unitary_Nine: 1,    
        WinableType.Clear_River_Self: 1,
        WinableType.Self_Wind: 1,
        WinableType.Ground_Wind: 1,             
        WinableType.Zhong: 1,
        WinableType.Bai: 1,
        WinableType.Fa: 1,             
        WinableType.Avg: 1,             
        WinableType.One_Cup: 1,                 
        WinableType.Grab_Gang: 1,               
        WinableType.Bloom: 1,              
        WinableType.Torch_Moon: 1,              
        WinableType.Torch_Fish: 1,            
        WinableType.Once: 1,              
        WinableType.W_Stand: 2,                 
        WinableType.Bump3: 2,                
        WinableType.Three_Gang: 2,           
        WinableType.All_Bump: 2,           
        WinableType.Three_Hidden_Bump: 2,    
        WinableType.Small_Yuan: 2,
        WinableType.Mix_Older: 2,        
        WinableType.Seven_Pairs: 2,         
        WinableType.Mix_Unitary_Nine: 2,      
        WinableType.One_To_Nine: 2,     
        WinableType.Avg3: 2,     
        WinableType.Two_Cup: 3,               
        WinableType.Purity_Unitary_Nine: 3,  
        WinableType.Mix_Color: 3,
        WinableType.Purity_Color: 6,            
        WinableType.Flow: 5
    }

    table_sub = [
        WinableType.Mix_Unitary_Nine,
        WinableType.One_To_Nine, 
        WinableType.Avg3,
        WinableType.Purity_Unitary_Nine, 
        WinableType.Mix_Color, 
        WinableType.Purity_Color  
    ]

    restrain = {
        WinableType.Purity_Flower: [WinableType.Flower],
        WinableType.All_Unitary_Nine_Full: [WinableType.All_Unitary_Nine],
        WinableType.Four_Hidden_Bump_Single: [WinableType.Four_Hidden_Bump],
        WinableType.Purity_Color: [WinableType.Mix_Color],
        WinableType.Purity_Unitary_Nine: [WinableType.Mix_Unitary_Nine, WinableType.Mix_Older],
        WinableType.Two_Cup: [WinableType.One_Cup],
        WinableType.Mix_Older: [WinableType.Mix_Unitary_Nine],
        WinableType.W_Stand: [WinableType.Stand]
    }

    full_map = {
        WinableType.Sky: 1,         
        WinableType.Land: 1,                    
        WinableType.Big_Yuan: 1,                
        WinableType.Words: 1,               
        WinableType.Green: 1,                  
        WinableType.Purity_Old: 1,              
        WinableType.All_Unitary_Nine: 1,        
        WinableType.Small_Wind: 1,       
        WinableType.Four_Gang: 1,             
        WinableType.Flower: 1,              
        WinableType.Four_Hidden_Bump: 1,  
        WinableType.Four_Hidden_Bump_Single: 2,
        WinableType.All_Unitary_Nine_Full: 2,
        WinableType.Purity_Flower: 2,  
        WinableType.Big_Wind: 2
    }

    times_score = {
        1: 1000,
        2: 2000,
        3: 4000,
        4: 6000,
        5: 8000,
        6: 12000,
        7: 12000,
        8: 16000,
        9: 16000,
        10: 16000,
        11: 24000,
        12: 24000,
        13: 32000
    }

    def o_winable(self, player, game, package, hand_can_be_well=None):
        
        # 首先保证去掉任意一张牌都听牌
        if not self.acceptable(player, game, package, hand_can_be_well):
            return False, [], None

        results, detector = self.get_winable_map(player.hand_cards, player.table_cards, game.active_card, package, None, hand_can_be_well)
        assert(isinstance(detector, MahjongDetector))

        # 首先查看役满
        has_full = False
        for type in self.full_map.keys():
            if type not in self.activate_winable_types:
                continue
            if results[type] != 0:
                has_full = True
                break

        # 有役满的情况下就抑制所有非役满
        if has_full:                
            for type in self.times_map.keys():  
                results[type] = 0

        # 去除所有的没有达成的类型
        items = []
        for type in results.keys():
            if results[type]:
                for _ in range(results[type]):
                    items.append(type)
        return len(items) != 0, items, detector

    def o_calculate_score(self, game: MGame, winner_list, items, source_player_index):
        fulls = []
        sums = []
        clear_table = [len(player.hand_cards) >= 13 for player in game.players]
        for i in range(len(items)):
            fulls.append(items[i][0] in self.full_map.keys())
            sum = 0
            for item in items[i]:
                if fulls[i]:
                    sum += self.full_map[item]
                else:
                    sum += self.times_map[item]
                if item in self.table_sub and not clear_table[winner_list[i]]:
                    sum -= 1
            sums.append(sum)

        # 宝牌与里宝牌与红宝牌
        treasure = game.card_pile.get_mcs_of_indices([i + 4 for i in range(game.treasure)])
        inner_treasure = game.card_pile.get_mcs_of_indices([i + 9 for i in range(game.treasure)])
        treasure = [self.get_next_mc(item) for item in treasure]
        inner_treasure = [self.get_next_mc(item) for item in inner_treasure]
        for i in range(len(fulls)):
            if fulls[i]:
                continue
            hand_cards = game.players[winner_list[i]].hand_cards
            tr = 0
            for card in hand_cards:
                if card.mc in treasure:
                    tr += 1

            itr = 0
            if game.standing[winner_list[i]]:
                for card in hand_cards:
                    if card.mc in inner_treasure:
                        itr += 1
            self.logger.trace("Player {} treasure {}; inner treasure {}".format(game.players[winner_list[i]].name, tr, itr))
            sums[i] += tr + itr

        scores = []
        for i in range(len(fulls)):
            if fulls[i]:
                score = 32000 * sums[i]
            else:
                times = min(sums[i], 13)
                score = self.times_score[times]
            
            # 庄家
            self_main = False
            if game.main_player == winner_list[i]:
                score = int(score * 1.5)
                self_main = True

            # 自摸分配三家
            if source_player_index == winner_list[i]:
                if self_main:
                    score_list = [-score // 3, -score // 3, -score // 3, -score // 3]
                else:
                    score_list = [-score // 4, -score // 4, -score // 4, -score // 4]
                    score_list[game.main_player] = -score // 2
                score_list[winner_list[i]] = score

            # 单胡一家
            else:
                score_list = [0, 0,  0,  0]
                score_list[winner_list[i]] = score
                score_list[source_player_index] = -score

            scores.append(score_list)
        self.logger.trace("Add score: " + ",".join(["{}".format(score) for score in scores]))
        return scores

    def is_flow(self, player: Player):
        flow = True
        for card in player.card_river:
            if card.mc not in MC.Unitary_Nine:
                flow = False
                break
        return flow

    def o_judge_round_is_over(self, game):
        return game.win_info.get(game.get_map_key()) != None 

class Sts4Config(Config):
    max_player = 4                              # 玩家数
    max_iter = 8                                # 最大游戏轮数
    start_scores = [25000, 25000, 25000, 25000] # 初始分数

class Sts4MGame(MGame):
    def __init__(self, config: Sts4Config, name="test"):
        super().__init__(config, name)

    def o_on_cpg(self, op, player_index, source_index, card):
        self.treasure += 1
        return super().o_on_cpg(op, player_index, source_index, card)

    def _on_gang4(self, player_index, gang):
        self.treasure += 1
        super()._on_gang4(player_index, gang)
    
    def o_init_rule(self):
        self.rule = Sts4Rule()

    def o_init_card_pile(self):
        self.card_pile = MCardPile(14)
        self.card_pile.init_by_default()
        self.card_pile.shuffle()

    def o_init_round_basic_props(self):
        self.conti_main_player = {}
        self.treasure = 1
        for i in range(self.max_iter):
            self.conti_main_player[i] = 0
        self.ground_wind = (self.iter // 4) + MC.Z1
        self.self_wind = [(i - self.iter + 4) % 4 + MC.Z1 for i in range(self.max_player)]

    def _make_package(self, by_self, grab_gang, bloom, player_index):
        package = super()._make_package(by_self, grab_gang, bloom, player_index)
        package["self_wind"] = self.self_wind[player_index]
        package["ground_wind"] = self.ground_wind
        return package

    def get_map_key(self):
        return "{}_{}".format(self.iter, self.conti_main_player[self.iter])

    def o_record_winner_list(self, winner_list, items, packages):
        key = self.get_map_key()
        for i in range(len(winner_list)):
            if self.win_info.get(key) is None:
                self.win_info[key] = []
            self.win_info[key].append([winner_list[i], items[i]])

    def o_end_round(self):

        # 计算流局满贯
        map_key = self.get_map_key()
        if self.win_info.get(map_key) == None:
            for i in range(self.max_player):
                flow = self.rule.is_flow(self.players[i])
                if flow:
                    if self.win_info.get(map_key) == None:
                        self.win_info[map_key] = []
                    self.win_info[map_key].append((i, WinableType.Flow))
        super().o_end_round()

    # 流局
    def o_flow_round(self):
        if self.rule.o_judge_round_is_over(self):
            return
        listen = []
        listen_sum = 0
        for i in range(self.max_player):
            lst, _ = self.rule.listen(self.players[i].hand_cards)
            listen.append(lst)
            listen_sum += 1 if lst else 0

        if listen_sum == 3:
            for i in range(len(listen)):
                if not listen[i]:
                    self.scores[i] -= 3000
                else:
                    self.scores[i] += 1000
        elif listen_sum == 2:
            for i in range(len(listen)):
                if not listen[i]:
                    self.scores[i] -= 1500
                else:
                    self.scores[i] += 1500
        elif listen_sum == 1:
            for i in range(len(listen)):
                if not listen[i]:
                    self.scores[i] -= 1000
                else:
                    self.scores[i] += 3000

    def o_decision_next_main_player(self):

        # 没有流局的话
        key = self.get_map_key()
        if self.win_info.get(key) != None:
            winner_list = [item[0] for item in self.win_info[key]]
            if self.main_player in winner_list:      # 庄家获胜
                self.conti_main_player[self.iter] += 1
            else:
                self.iter += 1
                self.next_main_player = (self.main_player + 1) % self.max_player

        # 流局的话, 查看庄家是否听牌
        else:
            listen = []
            for i in range(self.max_player):
                lst, _ = self.rule.listen(self.players[i].hand_cards)
                listen.append(lst)
            if listen[self.main_player]:
                self.conti_main_player[self.iter] += 1
            else:
                self.next_main_player = (self.main_player + 1) % self.max_player
                self.iter += 1
            