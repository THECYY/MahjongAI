from .game import *
from .chuan_majong import *

class ConsolePlayer(Player):
    
    def __init__(self, name):
        super().__init__(name)

    def get_inputs(self, str, num):
        inputs = input("Index to play: ")
        inputs = inputs.split(" ")
        rs = [0 for i in range(num)]
        for i in range(min(len(inputs), num)):
            rs[i] = str_to_int(input[i])
        return num

    def console_player_info_each_by_each(self, game: MGame):
        self_index = game._find_player(self)
        for i in range(1, game.max_player):
            print("Player {}:".format(game.players[i].name))
            print(game.players[(self_index + i) % game.max_player].showable_info_by_others())
        print("Yourself:")
        print(self.showable_info_by_self())

    def o_make_decision_to_gang4(self, game: MGame, gangs):
        print("Making choice to gang4")
        self.console_player_info_each_by_each(game)
        print("Gang: {}".format(CardList.array2_to_str(gangs)))
        while True:
            inputs = self.get_inputs("Gang4? (0/1), index: ", 2)
            choice, index = inputs[0], inputs[1]
            if choice != NONE:
                if choice and (index > len(gangs) or index < 0):
                    print("For gang4, you only have {} choices, but you choose {}".format(len(gangs), index))
                    continue
                break
        return choice, index

    def o_make_decision_to_play(self, game: MGame, playable_cards):
        print("Making choice to play")
        self.console_player_info_each_by_each(game)
        print("Play: {}".format(CardList.to_str(playable_cards)))
        while True:
            index = self.get_inputs("Index to play: ", 1)[0]
            if index < 0 or index >= len(playable_cards):
                print("For play, you only have {} choices, but you choose {}".format(len(playable_cards), index))
                continue
            else:
                break
        return index
            
    def o_make_decision_to_win(self, game, win_types):
        print("Making choice to win")
        print("Yourself: {}".format(self.showable_info_by_self()))
        print("Win types:", WinableType.to_str(win_types))
        choice = NONE
        while True:
            choice = self.get_inputs("Win? (1/0): ", 1)[0]
            if choice != NONE:
                break
        return choice

    def o_make_decision_to_stand(self, game, playable_cards):
        print("Making choice to stand")
        self.console_player_info_each_by_each(game)

        print("If you stand, you can play card:", CardList.to_str(playable_cards))
        while True:
            choice = self.get_inputs("Stand? (1/0): ", 1)[0]
            if choice != NONE:
                break
        return choice            

    def o_make_decision_to_cpg(self, game: MGame, card: Card, choice, eat, bump, gang):
        print("Making choice to cpg")
        self_index = game._find_player(self)
        for i in range(1, game.max_player):
            print("Next player {}:".format(i))
            print(game.players[(self_index + i) % game.max_player].showable_info_by_others())
        print("Yourself:")
        print(self.showable_info_by_self())
        print("Processing card: {}".format(card.get_name()))
        op = Op.Empty
        index = -1
        card_lists = None
        if Op.Eat in choice and op == Op.Empty:
            while True:
                print("Eat: {}".format(CardList.array2_to_str(eat)))
                inputs = self.get_inputs("Eat? (0 or 1) index: ", 2)
                e, eat_index = str_to_int(inputs[0]), str_to_int(inputs[1])
                if eat_index >= len(eat):
                    print("For eat, you have only {} choices, but your choice is {}".format(len(eat), eat_index))
                    continue
                op = Op.Eat if e else op
                index = eat_index
                card_lists = eat[index]
                break

        if Op.Gang1 in choice and op == Op.Empty:
            while True:
                print("Gang1: {}".format(CardList.array2_to_str(gang)))
                inputs = self.get_inputs("Gang1? (0 or 1) index: ", 2)
                g, gang_index = str_to_int(inputs[0]), str_to_int(inputs[1])
                if gang_index >= len(gang):
                    print("For gang1, you have only {} choices, but your choice is {}".format(len(gang), gang_index))
                    continue
                op = Op.Gang1 if g else op
                index = gang_index
                card_lists = gang[index]
                break

        if Op.Bump in choice and op == Op.Empty:
            while True:
                print("Bump: {}".format(CardList.array2_to_str(bump)))
                inputs = self.get_inputs("Bump? (0 or 1) index: ", 2)
                b, bump_index = str_to_int(inputs[0]), str_to_int(inputs[1])
                if bump_index >= len(bump):
                    print("For bump, you have only {} choices, but your choice is {}".format(len(bump), bump_index))
                    continue
                op = Op.Bump if b else op
                index = bump_index
                card_lists = bump[index]
                break

        return op, index, card_lists

    def o_make_decision_to_lack(self, game):
        while True:
            inputs = self.get_inputs("Lack type, I(0), O(1), W(2): ", 1)
            lack = inputs[0]
            if lack >= len(3) or lack < 0:
                print("For lack, you have only 3 choices, but your choice is {}".format(lack))
                continue
            break
        return lack

    def o_make_decision_to_change_three(self, cardss):
        while True:
            print("Cardss: {}".format(CardList.array2_to_str(cardss)))
            inputs = self.get_inputs("Change three cards: index of cards, index1, index2, index3", 4)
            index_of_cards = inputs[0]
            if index_of_cards >= len(cardss) or index_of_cards < 0:
                print("For cards, you have only {} choices, but your choice is {}".format(len(cardss), index_of_cards))
                continue
            cards = cardss[index_of_cards]
            if inputs[1] > len(cards):
                print("For card, you have only {} choices, but your choice is {}".format(len(cards), inputs[1]))
                continue
            if inputs[2] > len(cards):
                print("For card, you have only {} choices, but your choice is {}".format(len(cards), inputs[2]))
                continue
            if inputs[3] > len(cards):
                print("For card, you have only {} choices, but your choice is {}".format(len(cards), inputs[3]))
                continue
            break
        return index_of_cards, inputs[1], inputs[2], inputs[3]
    
class RandomPlayer(Player):

    def __init__(self, name):
        super().__init__(name)

    def o_make_decision_to_cpg(self, game, card, choice, eat, bump, gang):
        if Op.Gang1 in choice:
            return Op.Gang1, 0, gang[0]
        elif Op.Bump in choice:
            return Op.Bump, 0, bump[0]
        else:
            return Op.Eat, 0, eat[0]

    def o_make_decision_to_gang4(self, game, gangs):
        return True, 0

    def o_make_decision_to_play(self, game, playable_cards):
        return random.randint(0, len(playable_cards) - 1)

    def o_make_decision_to_stand(self, game, playable_cards):
        return True
    
    def o_make_decision_to_win(self, game, win_types, package):
        return True

    def o_make_decision_to_lack(self, game):
        return random.randint(0, 2)

    def o_make_decision_to_change_three(self, cardss):
        return 0, 0, 1, 2

class ChuanHeuristicPlayer(Player):

    gs = [1.5, 1.2, 1]

    def __init__(self, name):
        self.lack = None
        super().__init__(name)

    # 选取最少的当作自己的定缺
    def o_make_decision_to_lack(self, game):
        types = CardList.stata_types(self.hand_cards)
        min_num = 99
        type = MC.Z
        for i in range(3):
            if types[i] < min_num:
                min_num = types[i]
                type = i
        self.lack = type
        self.rule = game.rule
        return type

    # 一定要杠（非缺）
    def o_make_decision_to_gang4(self, game, gangs):
        index = 0
        for gang in gangs:
            if MCType.is_type(gang[0].mc, self.lack):
                index += 1
                continue
            return True, index
        return False, NONE

    def _get_first_lack_index(self, cards):
        index = 0
        for card in cards:
            if MCType.is_type(card.mc, self.lack):
                return index
            index += 1
        return NONE

    def _get_loss_sub_score(self, card):
        return -1

    def get_mc_score(self, cards):

        class MCWithScore:
            def __init__(self, mc, score):
                self.mc = mc
                self.score = score

        scores = [0 for _ in range(27)]
        for card in cards:
            mc = card.mc
            if mc in MC.Unitary_Nine:
                scores[mc] += ChuanHeuristicPlayer.gs[2]
            else:
                scores[mc] += ChuanHeuristicPlayer.gs[0]
            l = self.rule.get_previous_adj_mc(mc)
            if l != NONE:
                scores[l] += ChuanHeuristicPlayer.gs[1]
                ll = self.rule.get_previous_adj_mc(l)
                if ll != NONE:
                    scores[ll] += ChuanHeuristicPlayer.gs[2]
            r = self.rule.get_next_adj_mc(mc)
            if r != NONE:
                scores[r] += ChuanHeuristicPlayer.gs[1]
                rr = self.rule.get_previous_adj_mc(r)
                if rr != NONE:
                    scores[rr] += ChuanHeuristicPlayer.gs[2]
        mc_set = set()
        for card in cards:
            if not mc_set.__contains__(card.mc):
                scores[card.mc] += self._get_loss_sub_score(card)
                mc_set.add(card.mc)
        mc_with_score = []
        for card in cards:
            mc_with_score.append(MCWithScore(card.mc, scores[card.mc]))
        sorted(mc_with_score, key=lambda x: x.score)
        return mc_with_score

    # 首先打缺，然后追求牌效
    def o_make_decision_to_play(self, game, playable_cards):
        index = self._get_first_lack_index(playable_cards)
        if index != NONE:
            return index
        
        mc_with_score = self.get_mc_score(playable_cards)
        index = 0
        for card in playable_cards:
            if card.mc == mc_with_score[0].mc:
                return index
            index += 1
        raise Exception("???")

    # 如果有杠一定要杠（非缺），然后再判断是否bump
    def o_make_decision_to_cpg(self, game, card, choice, eat, bump, gang):
        index = 0
        for g in gang:
            if not MCType.is_type(g[0].mc, self.lack):
                return Op.Gang1, index, g
            index += 1 
        
        index = 0
        for b in bump:
            return Op.Bump, index, b

        return Op.Empty, NONE, []

    # 决断自摸与胡牌
    def o_make_decision_to_win(self, game: MGame, win_types, package):

        # 自摸直接胡
        if package["by_self"]:
            return True

        # 开局比较早
        river = 0
        for player in game.players:
            river += len(player.card_river)
        if river > 40:
            return True

        # 已经有人胡
        for win in game.won:
            if win:
                return True

        # 其他家的牌不是特别大
        player_index = game._find_player(self)
        for i in range(game.max_player - 1):
            index = (i + 1 + player_index) % game.max_player
            if not game.won[index]:
                analysis = ChuanJudgeOtherPlayerWinTypeOfPlayer(game.players[index], game, self.lack)
                items = analysis.perform()
                sum = 0
                for item in items:
                    sum += ChuanRule.times_map[item]
                if sum >= 3:
                    return True

        return False

    def o_make_decision_to_change_three(self, cardss):
        index, min = [], 9999
        for i in range(len(cardss)):
            if min > len(cardss[i]):
                min = len(cardss[i])
                index = i
        return index, 0, 1, 2

class ChuanPurityColorPlayer(ChuanHeuristicPlayer):

    def __init__(self, name):
        self.purity_type = NONE
        super().__init__(name)
        self._choose_purity_type()

    def _choose_purity_type(self):
        types = CardList.stata_types(self.hand_cards)
        max_num = NONE
        for i in range(3):
            if types[i] > max_num:
                max_num = types[i]
                self.purity_type = i

    def o_make_decision_to_gang4(self, game: MGame, gangs):
        index = 0
        for gang in gangs:
            if MCType.is_type(gang[0].mc, self.purity_type):
                return True, index
            index += 1
        return True, 0

    # 有杠就杠，不然就不做
    def o_make_decision_to_cpg(self, game, card, choice, eat, bump, gang):
        index = 0
        for g in gang:
            if not MCType.is_type(g[0].mc, self.lack):
                return Op.Gang1, index, g
            index += 1 

        index = 0
        for b in bump:
            if MCType.is_type(b[0].mc, self.purity_type):
                return Op.Bump, index, b
        return Op.Empty, NONE, []

    # 打缺，打非清
    def o_make_decision_to_play(self, game, playable_cards):
        index = self._get_first_lack_index(playable_cards)
        if index != NONE:
            return index
        
        index = 0
        for card in playable_cards:
            if not MCType.is_type(card.mc, self.purity_type):
                return index
            index += 1

        mc_with_score = self.get_mc_score(playable_cards)
        index = 0
        for card in playable_cards:
            if card.mc == mc_with_score[0].mc:
                return index
            index += 1

        raise Exception("???")

    def o_make_decision_to_win(self, game: MGame, win_types, package):
        return True

class ChuanSevenPairsPlayer(ChuanHeuristicPlayer):

    def o_make_decision_to_gang4(self, game: MGame, gangs):
        return False, 0

    def o_make_decision_to_win(self, game: MGame, win_types, package):
        return True

    def o_make_decision_to_cpg(self, game, card, choice, eat, bump, gang):
        return Op.Empty, NONE, []

    # 七对子思路
    def o_make_decision_to_play(self, game, playable_cards):

        # 首先还是打缺
        index = self._get_first_lack_index(playable_cards)
        if index != NONE:
            return index

        # 首先应该打出的是场上已经出现了三次的手牌中有的牌/手牌中三张但是场上有的牌
        game_mc_map = GameStata.get_mc_map_of_card_river_and_groups(game)
        hand_map = CardList.get_mc_map(self.hand_cards)
        
        for mc in hand_map.keys():
            count = hand_map[mc]
            if count == 1 or count == 3:
                game_mc = game_mc_map.get(mc)
                game_mc = 0 if game_mc is None else game_mc
                if count + game_mc == 4:
                    index = 0
                    for card in playable_cards:
                        if card.mc == mc:
                            return index
                        index += 1
        
        # 其次打出的应该是手牌有一个，场上有两个的
        for mc in hand_map.keys():
            count = hand_map[mc]
            if count == 1:
                game_mc = game_mc_map.get(mc)
                game_mc = 0 if game_mc is None else game_mc
                if count + game_mc == 3:
                    index = 0
                    for card in playable_cards:
                        if card.mc == mc:
                            return index
                        index += 1
        
        # 除了对子，那么就按照评分进行打出
        cards = []
        for card in playable_cards:
            if hand_map.get(card.mc) != 2:
                cards.append(card)
        mc_with_score = self.get_mc_score(cards)
        
        # 场上有一个就加一分
        for ms in mc_with_score:
            count = game_mc_map.get(ms.mc)
            count = count if count is not None else 0
            if count == 1:
                ms.score += ChuanHeuristicPlayer.gs[2]
            
        index = 0
        for card in playable_cards:
            if card.mc == mc_with_score[0].mc:
                return index
            index += 1
        raise Exception("???")

# AI麻将
class ChuanAIPlayer(ChuanHeuristicPlayer):

    def __init__(self, name):
        super().__init__(name)
        self.play_brain = None

    def mate_play_brain(self, model):
        self.play_brain = model
        
    def o_make_decision_to_play(self, game, playable_cards):
        
        # 先打缺
        index = self._get_first_lack_index(playable_cards)
        if index != NONE:
            return index
        
        if self.play_brain is None:
            return super().o_make_decision_to_play(game, playable_cards)
        with torch.no_grad():
            data = game_shotcum_to_play_card_data(game)
            h, hm, tt, tmc, tm, r, rm, ott, otmc, otm, ori, orm, won = data["hand"], data["hand_mask"], data["table_types"], \
                data["table_mcs"], data["table_mask"], data["river"], data["river_mask"], \
                    data["other_table_types"], data["other_table_mcs"], data["other_table_mask"], \
                     data["other_river"], data["other_river_mask"], data["won"]
            h, hm, tt, tmc, tm = h.unsqueeze(0), hm.unsqueeze(0), tt.unsqueeze(0), tmc.unsqueeze(0), tm.unsqueeze(0)
            r, rm, ott, otmc, otm = r.unsqueeze(0), rm.unsqueeze(0), ott.unsqueeze(0), otmc.unsqueeze(0), otm.unsqueeze(0)
            ori, orm, won = ori.unsqueeze(0), orm.unsqueeze(0), won.unsqueeze(0)

            # [1, 14]
            output = self.play_brain(h, hm, ott, otmc, otm, tt, tmc, tm, ori, orm, r, rm, won, False)
            output = output.squeeze(0)[:len(self.hand_cards)]
            output = torch.softmax(output, dim=-1)
            index = torch.argmax(output)
        return int(index)

    ## 不平衡
    #def o_make_decision_to_cpg(self, game, card, choice, eat, bump, gang):
    #    pass

    ## 不平衡
    #def o_make_decision_to_gang4(self, game, gangs):
    #    pass

    ## 不平衡
    #def o_make_decision_to_win(self, game: MGame, win_types, package):
    #    pass

    ## 可以启发式
    #def o_make_decision_to_lack(self, game):
    #    pass


player_type_map = {
    "random": RandomPlayer,
    "console": ConsolePlayer,
    "chuan_heu": ChuanHeuristicPlayer,
    "chuan_pp": ChuanPurityColorPlayer,
    "chuan_sp": ChuanSevenPairsPlayer,
    "chuan_ai": ChuanAIPlayer
}

def get_player(type, name):
    return player_type_map[type](name)