from .card import *
from mglobal import *

class Heuristic:
    def perform(self):
        raise NotImplementedError()

class HandCardCanBeWell(Heuristic):

    def __init__(self, hand_cards, only_bump=False):
        self._hand_cards = copy.deepcopy(hand_cards)
        self.results = []
        self.only_bump = only_bump

    '''
        返回所有听牌的数组：
        item = (need_card, head_mc, mcs_groups)
    '''
    def perform(self):
        if len(self._hand_cards) % 3 != 1:
            return None
        if len(self.results) == 0:
            self._heruistic_search_can_be_hand_well()
        return self.results

    def _heruistic_search_can_be_hand_well(self):
        mcs = [0 for _ in range(len(MC.I) + len(MC.O) + len(MC.W) + len(MC.Z))]
        min = NONE
        for card in self._hand_cards:
            mcs[card.mc] += 1
            if card.mc < min or min == NONE:
                min = card.mc
        return self._dfs_search(mcs, min)

    def _next_index(self, array, index):
        for i in range(index , len(array)):
            if array[i] != 0:
                return i
        return len(array)

    def _dfs_search(self, mcs, index=0, head=NONE, groups=[], exhaushed=False, need_mc=NONE):
        if index == len(mcs):
            assert(exhaushed and head != NONE)
            exist = False
            for result in self.results:
                if result[0] == need_mc:
                    exist = True
            if not exist:
                self.results.append([need_mc, [[item for item in group.mcs] for group in groups], head])
            return

        # As head
        if head == NONE:

            # 消耗递归
            if not exhaushed:
                head = index
                mcs[index] -= 1
                next_index = self._next_index(mcs, index)
                self._dfs_search(mcs, next_index, head, groups, True, index)
                mcs[index] += 1
                head = NONE

            # 没有消耗掉需要的牌也能够当雀头
            if mcs[index] >= 2:
                head = index
                mcs[index] -= 2
                next_index = self._next_index(mcs, index)
                self._dfs_search(mcs, next_index, head, groups, exhaushed, need_mc)
                mcs[index] += 2
                head = NONE

        # As bump
        if mcs[index] >= 2:

            # 消耗递归
            if not exhaushed:
                bump_group = MCGroup([index, index, index], Op.Bump)
                mcs[index] -= 2
                next_index = self._next_index(mcs, index)
                groups.append(bump_group)
                self._dfs_search(mcs, next_index, head, groups, True, index)
                groups.remove(bump_group)
                mcs[index] += 2

            # 非消耗递归
            if mcs[index] >= 3:
                bump_group = MCGroup([index, index, index], Op.Bump)
                mcs[index] -= 3
                next_index = self._next_index(mcs, index)
                groups.append(bump_group)
                self._dfs_search(mcs, next_index, head, groups, exhaushed, need_mc)
                groups.remove(bump_group)
                mcs[index] += 3

        # 仅仅检测bump就结束算法
        if self.only_bump:
            return

        # As first of avg
        if MC.same_group(index, index + 1, index + 2) and index < MC.Z1:

            # 非消耗递归
            if mcs[index] and mcs[index + 1] and mcs[index + 2]:
                mcs[index] -= 1
                mcs[index + 1] -= 1
                mcs[index + 2] -= 1
                next_index = self._next_index(mcs, index)
                group = MCGroup([index, index + 1, index + 2], Op.Eat)
                groups.append(group)
                self._dfs_search(mcs, next_index, head, groups, exhaushed, need_mc)
                groups.remove(group)
                mcs[index] += 1
                mcs[index + 1] += 1
                mcs[index + 2] += 1

            # 消耗index + 1递归
            if mcs[index] and mcs[index + 2] and not exhaushed:
                mcs[index] -= 1
                mcs[index + 2] -= 1
                next_index = self._next_index(mcs, index)
                group = MCGroup([index, index + 1, index + 2], Op.Eat)
                groups.append(group)
                self._dfs_search(mcs, next_index, head, groups, True, index + 1)
                groups.remove(group)
                mcs[index] += 1
                mcs[index + 2] += 1

            # 消耗index + 2递归
            if mcs[index] and mcs[index + 1] and not exhaushed:
                mcs[index] -= 1
                mcs[index + 1] -= 1
                next_index = self._next_index(mcs, index)
                group = MCGroup([index, index + 1, index + 2], Op.Eat)
                groups.append(group)
                self._dfs_search(mcs, next_index, head, groups, True, index + 2)
                groups.remove(group)
                mcs[index] += 1
                mcs[index + 1] += 1

        # As second of avg
        if MC.same_group(index - 1, index, index + 1) and index < MC.Z1:
                    
            # 只能进行消耗性递归
            if mcs[index] and mcs[index + 1] and not exhaushed:
                mcs[index] -= 1
                mcs[index + 1] -= 1
                next_index = self._next_index(mcs, index)
                group = MCGroup([index - 1, index, index + 1], Op.Eat)
                groups.append(group)
                self._dfs_search(mcs, next_index, head, groups, True, index - 1)
                groups.remove(group)
                mcs[index] += 1
                mcs[index + 1] += 1

class GameStata:
    
    def get_mc_map_of_card_river_and_groups(game):
        map = {}
        for player in game.players:
            for card in player.card_river:
                map[card.mc] = 1 if map.get(card.mc) is None else map[card.mc] + 1

            for group in player.table_cards:
                for card in group.cards:
                    map[card.mc] = 1 if map.get(card.mc) is None else map[card.mc] + 1
        return map

# 分析牌的散列程度
class CardsDivergence(Heuristic):
    def __init__(self, cards):
        super().__init__()
        self.mcs = [card.mc for card in cards]
        self.i = [0 for _ in MC.I]
        self.o = [0 for _ in MC.O]
        self.w = [0 for _ in MC.W]    
        for mc in self.mcs:
            if MCType.is_type(mc, MCType.I):
                self.i.append(mc)
            elif MCType.is_type(mc, MCType.O):
                self.o.append(mc)
            elif MCType.is_type(mc, MCType.W):
                self.w.append(mc)

    # 1-3 4-6 7-9
    def _seg_analysis(self, list):
        arr = []
        arr.append(list[0] or list[1] or list[2])
        arr.append(list[3] or list[4] or list[5])
        arr.append(list[6] or list[7] or list[8])
        return arr, arr[0] + arr[1] + arr[2]

    def _slice_analysis(self, list):
        arr = []
        arr.append(list[0] or list[3] or list[6])
        arr.append(list[1] or list[4] or list[7])
        arr.append(list[2] or list[5] or list[8])
        return arr, arr[0] + arr[1] + arr[2]

    def perform(self):
        analysis = {
            "seg": [
                self._seg_analysis(self.i),
                self._seg_analysis(self.o),
                self._seg_analysis(self.w)
            ],
            "slice":
            [
                self._slice_analysis(self.i),
                self._slice_analysis(self.o),
                self._slice_analysis(self.w),
            ]
        }
        return analysis

# 川麻意图
class ChuanJudgeOtherPlayerWinTypeOfPlayer(Heuristic):

    def __init__(self, player, game, lack):
        super().__init__()
        self.player = player
        self.game = game
        self.lack = lack

    def _stat(self):
        self.divergence = CardsDivergence(self.player.card_river)
        self.card_river_analysis = self.divergence.perform()
        self.card_river_stat = CardList.stata_types(self.player.card_river)
        self.table_card_stat = [0, 0, 0]
        for group in self.player.table_cards:
            for card in group.cards:
                self.table_card_stat[MCType.get_type_of_mc(card.mc)] += 1

        self.unlack = (self.card_river_stat[0] if self.lack != MCType.I else 0) + \
                (self.card_river_stat[1] if self.lack != MCType.O else 0) + \
                    (self.card_river_stat[2] if self.lack != MCType.W else 0)
        self.game_stat = GameStata.get_mc_map_of_card_river_and_groups(self.game)

    def _judge_all_bump(self):
        return len(self.player.table_cards) >= 3

    def _judge_seven_pairs(self):
        
        i_seg = self.card_river_analysis["seg"][0][1] >= 2 if self.lack != MCType.I else 0
        i_seg_three = self.card_river_analysis["seg"][0][1] >= 3 if self.lack != MCType.I else 0
        o_seg = self.card_river_analysis["seg"][1][1] >= 2 if self.lack != MCType.O else 0
        o_seg_three = self.card_river_analysis["seg"][1][1] >= 3 if self.lack != MCType.O else 0
        w_seg = self.card_river_analysis["seg"][1][1] >= 2 if self.lack != MCType.W else 0
        w_seg_three = self.card_river_analysis["seg"][1][1] >= 3 if self.lack != MCType.W else 0
            
        three = i_seg_three + o_seg_three + w_seg_three
        seg = i_seg + o_seg + w_seg
        
        return len(self.player.table_cards) == 0 and three >= 1 and seg >= 5 # 出牌散度比较大

    def _judge_putrity_color(self):
        
        # 首先保证出牌数量（非缺）的大于等于5
        if self.unlack <= 5:
            return False

        # 所有table一色（非缺）
        if self.table_card_stat[0] and not self.table_card_stat[1] and not self.table_card_stat[2]:
            purity_type = MCType.I
        elif not self.table_card_stat[0] and self.table_card_stat[1] and not self.table_card_stat[2]:
            purity_type = MCType.O
        elif not self.table_card_stat[0] and not self.table_card_stat[1] and self.table_card_stat[2]:
            purity_type = MCType.W
        else:
            return False
        
        # 确保出牌某一色小于等于20%
        return self.card_river_stat[purity_type] / self.unlack < 0.2

    def _judge_gold_hook(self):
        return len(self.player.table_cards) == 4

    def _judge_purity_unitary_nine(self):

        # 幺九系列牌小于等于20%
        i1 = self.card_river_analysis["seg"][0][1]
        o1 = self.card_river_analysis["seg"][1][1]
        w1 = self.card_river_analysis["seg"][2][1]
        sum = (i1 if self.lack != MCType.I else 0) + (o1 if self.lack != MCType.O else 0) + (w1 if self.lack != MCType.W else 0)
        
        if self.unlack == 0:
            return False

        if self.unlack < 10 and sum / self.unlack < 0.35:
            return False

        for group in self.player.table_cards:
            if group.op != Op.Eat and not group.cards[0].mc in MC.Unitary_Nine:
                return False
        
        return True

    def _judge_four_gang(self):
        gang = 0
        for group in self.player.table_cards:
            if group.op in [Op.Gang1, Op.Gang4]:
                gang += 1
            else:
                if self.game_stat.get(group.cards[0].mc) == 4:
                    return False
        return gang >= 3

    def _judge_major_bump(self):
        if not self._judge_all_bump():
            return False
        for group in self.player.table_cards:
            if group.cards[0].mc not in MC.Major:
                return False
        return True

    def perform(self):
        self._stat()
        items = []
        if self._judge_gold_hook():
            items.append(WinableType.Gold_Hook)
        elif self._judge_all_bump():
            items.append(WinableType.All_Bump)
        if self._judge_seven_pairs():
            items.append(WinableType.Seven_Pairs)
        if self._judge_purity_unitary_nine():
            items.append(WinableType.Purity_Unitary_Nine)
        if self._judge_putrity_color():
            items.append(WinableType.Purity_Color)
        if self._judge_major_bump():
            items.append(WinableType.Major_Bump)
        if self._judge_four_gang():
            items.append(WinableType.Four_Gang)
        return items
    

