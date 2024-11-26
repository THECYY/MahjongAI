from .card import *
from mglobal import *
from .heuristic import *

def default_MD_package():
    return {
        "stand": 0,             # 立直
        "by_self": 0,           # 自摸
        "self_wind": NONE,      # 自风
        "ground_wind": NONE,    # 场风
        "grab_gang": 0,         # 抢杠
        "bloom": 0,             # 岭上开花
        "bottom": 0,            # 河底
        "once": 0,              # 一发
        "w_stand": 0,           # 两立直
        "flow": 0,              # 流局满贯
        "clear_river": 0,       # 空牌河
        "main_player": 0,       # 庄家
    }

class MahjongDetector:

    def __init__(self, hand_cards, card_groups, last_card, hand_mc_map=None, package={}, hand_can_well: HandCardCanBeWell=None):
        self._hand_cards = []        
        self._hand_cards = copy.deepcopy(hand_cards)

        removed = False
        if len(hand_cards) % 3 == 2:
            for card in self._hand_cards:
                if card.equal(last_card):
                    self._hand_cards.remove(card)
                    removed = True
                    break
        elif len(hand_cards) % 3 == 1:
            removed = True

        self.heuristic = HandCardCanBeWell(self._hand_cards, False) if hand_can_well is None else hand_can_well
        if removed:
            CardList.add_card_with_sort(self._hand_cards, last_card)
            
        self._package = package
        table_clear = True
        for group in card_groups:
            if group.op != Op.Gang4:
                table_clear = False
        self._clear_table = table_clear
        self._table_card_groups = card_groups
        self._last_mc = last_card.mc
        self._hand_mc_map = hand_mc_map if hand_mc_map is not None else CardList.get_mc_map(self._hand_cards)
        self._detect()
        self._stat()

    def hand_can_be_well_heuristic(self):
        return self.heuristic
   
    def _detect(self):

        # 雀头mc, 手牌自理, 手牌中bump数量, avg数量, bump mcs, avg mcs
        results = self.heuristic.perform()
        listen_list = [item[0] for item in results]
        self.hand_bump, self.hand_avg, self.bump_mc, self.avgs_mc = 0, 0, [], []
        if self._last_mc not in listen_list:
            self.hand_well = False
            self.head_mc = NONE
        else:
            index_of_result = listen_list.index(self._last_mc)
            result = results[index_of_result]
            self.head_mc = result[2]
            self.hand_well = True
            for mcs_group in result[1]:
                if mcs_group[0] == mcs_group[1]:
                    self.hand_bump += 1
                    self.bump_mc.append(mcs_group[0])
                else:
                    self.hand_avg += 1
                    self.avgs_mc.append(mcs_group[0])

        self.bump, self.avg = self.hand_bump, self.hand_avg
        for group in self._table_card_groups:
            if group.op in [1, 2, 3]:
                self.bump += 1
                self.bump_mc.append(group.cards[0].mc)
                if group.op == 3:
                    self.hand_bump += 1
            else:
                self.avg += 1
                self.avgs_mc.append(min(group.cards[0].mc, group.cards[1].mc, group.cards[2].mc))
    
        self._avgs_mc_map = {}
        for mc in self.avgs_mc:
            if self._avgs_mc_map.get(mc) is None:
                self._avgs_mc_map[mc] = 1
            else:
                self._avgs_mc_map[mc] += 1

    def _stat(self):
        
        # self.well针对普遍形式
        # stat针对特殊形式
        self.hand_pair = 0      # 手牌中对子
        self.hand_gang = 0      # 手牌中的杠
        self.old = 0            # 手牌中的老头牌个数
        self.gang1 = 0          # table上明杠的数量
        self.gang4 = 0          # table上暗杠的数量
        self.w, self.i, self.o, self.z = 0, 0, 0, 0 # 数量统计
        self.unitary_nine = 0
        self.green_count = 0
        for mc in self._hand_mc_map.keys():
            if self._hand_mc_map[mc] == 2:
                self.hand_pair += 1
            elif self._hand_mc_map[mc] == 4:
                self.hand_gang += 1

            if mc in MC.I:
                self.i += self._hand_mc_map[mc]
            elif mc in MC.O:
                self.o += self._hand_mc_map[mc]
            elif mc in MC.W:
                self.w += self._hand_mc_map[mc]
            elif mc in MC.Z:
                self.z += self._hand_mc_map[mc]
            if mc in MC.Green:
                self.green_count += self._hand_mc_map[mc]

            if mc in MC.Old:
                self.old += self._hand_mc_map[mc]
                self.unitary_nine += self._hand_mc_map[mc]
            elif mc in MC.Unitary_Nine:
                self.unitary_nine += self._hand_mc_map[mc]

        for group in self._table_card_groups:
            if group.op == 2:
                self.gang1 += 1
            elif group.op == 3:
                self.gang4 += 1
            for index in range(len(group.cards)):
                if index >= 3:
                    continue
                card = group.cards[index]
                if card.mc in MC.Old:
                    self.old += 1
                    self.unitary_nine += 1
                elif card.mc in MC.Unitary_Nine:
                    self.unitary_nine += 1

                if card.mc in MC.I:
                    self.i += 1
                elif card.mc in MC.O:
                    self.o += 1
                elif card.mc in MC.W:
                    self.w += 1
                elif card.mc in MC.Z:
                    self.z += 1

                if card.mc in MC.Green:
                    self.green_count += 1

    def _is_all_avg_contains_older(self):
        for mc in self.avgs_mc:
            if mc not in [MC.I1, MC.O1, MC.W1, MC.I7, MC.O7, MC.W7]:
                return False
        return True

    def _is_all_bump_contains_unitary_nine(self):
        for mc in self.bump_mc:
            if mc not in MC.Unitary_Nine:
                return False
        return True

    def _is_all_bump_contains_older(self):
        for mc in self.bump_mc:
            if mc not in MC.Old:
                return False
        return True

    # 立直 (In Game)
    def is_stand(self):
        return self._clear_table and self._package["stand"]

    # 断幺九
    def is_broken_unitary_nine(self):
        return self.unitary_nine == 0

    # 门前请自摸胡 (In Game)
    def is_clear_river_self(self):
        return len(self._hand_cards) == 14 and self._package["by_self"]  

    # 自风
    def is_self_wind(self):
        return self._package["self_wind"] in self.bump_mc 

    # 场风
    def is_ground_wind(self):
        return self._package["ground_wind"] in self.bump_mc
    
    # 中
    def is_zhong(self):
        return MC.Z5 in self.bump_mc

    # 白
    def is_bai(self):
        return MC.Z6 in self.bump_mc
    
    # 发
    def is_fa(self):
        return MC.Z7 in self.bump_mc
    
    # 平胡
    def is_avg(self):
        return self._clear_table and self.bump == 0

    # 一杯口
    def is_one_cup(self):
        one_cup = False
        for mc in self._avgs_mc_map:
            if self._avgs_mc_map[mc] >= 2:
                one_cup = 1
                break
        return self._clear_table and one_cup

    # 抢杠 (In Game)
    def is_grab_gang(self):
        return self._package["grab_gang"]

    # 岭上开花
    def is_bloom(self):
        return self._package["bloom"]

    # 海底捞月
    def is_torch_moon(self):
        return self._package["bottom"] and self._package["by_self"]

    # 河底捞鱼
    def is_torch_fish(self):
        return self._package["bottom"] and not self._package["by_self"]

    # 一发 (In Game)
    def is_once(self):
        return self._package["once"] 
    
    # 两立直
    def is_w_stand(self):
        return self._package["w_stand"]

    # 三色同刻
    def is_bump3(self):
        bump3 = False
        for i in range(9):
            if (MC.I1 + i) in self.bump_mc and (MC.O1 + i) in self.bump_mc and (MC.W1 + i) in self.bump_mc:
                bump3 = True
                break
        return bump3

    # 三杠子
    def is_three_gang(self):
        return self.gang1 + self.gang4 >= 3

    # 对对胡
    def is_all_bump(self):
        return len(self.bump_mc) == 4

    # 三暗刻
    def is_three_hidden_bump(self):
        return self.hand_bump >= 3

    # 小三元
    def is_small_yuan(self):
        yuan = self.is_zhong() + self.is_bai() + self.is_fa()
        return self.head_mc in MC.Yuan and yuan >= 2

    # 混老头
    def is_mix_older(self):
        return self._is_all_bump_contains_unitary_nine() and self.head_mc in MC.Unitary_Nine and self.is_mix_unitary_nine() and self.is_all_bump()

    # 七对子
    def is_seven_pairs(self):
        return self.hand_pair + self.hand_gang * 2 == 7

    # 混全带幺九
    def is_mix_unitary_nine(self):
        return self._is_all_avg_contains_older() and self._is_all_bump_contains_unitary_nine() \
            and self.head_mc in MC.Unitary_Nine

    # 一气通贯 
    def is_one_to_nine(self):
        return (MC.I1 in self.avgs_mc and MC.I4 in self.avgs_mc and MC.I7 in self.avgs_mc) or \
                  (MC.O1 in self.avgs_mc and MC.O4 in self.avgs_mc and MC.O7 in self.avgs_mc) or \
                    (MC.W1 in self.avgs_mc and MC.W4 in self.avgs_mc and MC.W7 in self.avgs_mc)

    # 三色同顺
    def is_avg3(self):
        avg3 = False
        for i in range(9):
            if MC.I1 + i in self.avgs_mc and MC.O1 + i in self.avgs_mc and MC.W1 + i in self.avgs_mc:
                avg3 = True
                break
        return avg3

    # 二杯口
    def is_two_cup(self):
        cup = 0
        for mc in self._avgs_mc_map:
            if self._avgs_mc_map[mc] >= 2:
                cup += 1
        return cup >= 2

    # 纯全带幺九
    def is_purity_unitary_nine(self):
        return self._is_all_avg_contains_older() and self._is_all_bump_contains_older() and self.head_mc in MC.Old
            
    # 混一色
    def is_mix_color(self):
        return self.z != 0 and ((self.i == 0 and self.w == 0 and self.o != 0) or \
            (self.i == 0 and self.w != 0 and self.o == 0) or (self.i != 0 and self.w == 0 and self.o == 0))
    
    # 清一色 
    def is_purity_color(self):
        return self.w == 14 or self.i == 14 or self.o == 14

    # 流局满贯 (In Game)
    def is_flow(self):
        return self._package["flow"]

    # 天和 (In Game)
    def is_sky(self):
        return self._package["main_player"] and self._package["clear_river"] and self._package["by_self"]

    # 地和 (In Game)
    def is_land(self):
        return not self._package["main_player"] and self._package["clear_river"] and self._package["by_self"]

    # 大三元
    def is_big_yuan(self):
        return MC.Z5 in self.bump_mc and MC.Z6 in self.bump_mc and MC.Z7 in self.bump_mc

    # 字一色
    def is_words(self):
        return self.i == 0 and self.o == 0 and self.w == 0

    # 绿一色
    def is_green(self):
        return self.green_count == 14

    # 清老头
    def is_purity_old(self):
        return self.old == 14

    # 国士无双
    def is_all_unitary_nine(self):
        if self._clear_table and self.unitary_nine == 14:
            for mc in MC.Unitary_Nine:
                if self._hand_mc_map.get(mc) is None:
                    return False
            return True
        return False

    # 小四喜
    def is_small_wind(self):
        wind = 0
        for mc in self.bump_mc:
            if mc in MC.Wind:
                wind += 1
        return wind == 3 and self.head_mc in MC.Wind

    # 四杠子
    def is_four_gang(self):
        return self.gang1 + self.gang4 == 4

    # 九莲宝灯
    def is_flower(self):
        if self._clear_table and self.is_purity_color():
            for basic in [MC.I1, MC.O1, MC.W1]:
                flower = True
                for i in range(9):
                    mcc = 0 if self._hand_mc_map.get(basic + i) is None else self._hand_mc_map.get(basic + i)
                    if i >= 1 and i <= 7 and mcc == 0:
                        flower = False
                        break
                    if i in [0, 8] and mcc <= 2:
                        flower = False
                        break
                if flower:
                    return flower
        return False

    # 四暗刻
    def is_four_hidden_bump(self):
        return self.bump == 4 and self._clear_table

    # 四暗刻单骑
    def is_four_hidden_bump_single(self):
        return self.is_four_hidden_bump() and self._last_mc == self.head_mc

    # 国士无双十三面
    def is_all_unitary_nine_full(self):
        return self.is_all_unitary_nine() and self._hand_mc_map[self._last_mc] == 2

    # 纯正九莲宝灯
    def is_purity_flower(self):
        return self.is_flower() and (self._hand_mc_map[self._last_mc] == 2 or self._hand_mc_map[self._last_mc] == 4)
    
    # 大四喜
    def is_big_wind(self):
        return MC.Z1 in self.bump_mc and MC.Z2 in self.bump_mc and MC.Z3 in self.bump_mc and MC.Z4 in self.bump_mc

    # 普通胡
    def is_hand_well(self):
        return self.hand_well

    # 龙七对
    def is_dragon_seven_pairs(self):
        return self.is_seven_pairs() and self.hand_gang >= 1

    # 将对
    def is_major_bump(self):
        major_bump = True
        for mc in self.bump_mc:
            if mc not in MC.Major:
                major_bump = False
                break
        return major_bump and self.is_all_bump()

    # 金钩钩
    def is_gold_hook(self):
        return len(self._hand_cards) == 2 and self.is_all_bump()
   
MD_FN_MAP = {
    WinableType.Stand: MahjongDetector.is_stand,
    WinableType.Broken_Unitary_Nine: MahjongDetector.is_broken_unitary_nine,
    WinableType.Clear_River_Self: MahjongDetector.is_clear_river_self,
    WinableType.Self_Wind: MahjongDetector.is_self_wind,
    WinableType.Ground_Wind: MahjongDetector.is_ground_wind,
    WinableType.Zhong: MahjongDetector.is_zhong,
    WinableType.Bai: MahjongDetector.is_bai,
    WinableType.Fa: MahjongDetector.is_fa,
    WinableType.Avg: MahjongDetector.is_avg,
    WinableType.One_Cup: MahjongDetector.is_one_cup,
    WinableType.Grab_Gang: MahjongDetector.is_grab_gang,
    WinableType.Bloom: MahjongDetector.is_bloom,
    WinableType.Torch_Moon: MahjongDetector.is_torch_moon,
    WinableType.Torch_Fish: MahjongDetector.is_torch_fish,
    WinableType.Once: MahjongDetector.is_once,
    WinableType.W_Stand: MahjongDetector.is_w_stand,
    WinableType.Bump3: MahjongDetector.is_bump3,
    WinableType.Three_Gang: MahjongDetector.is_three_gang,
    WinableType.All_Bump: MahjongDetector.is_all_bump,
    WinableType.Three_Hidden_Bump: MahjongDetector.is_three_hidden_bump,
    WinableType.Small_Yuan: MahjongDetector.is_small_yuan,
    WinableType.Mix_Older: MahjongDetector.is_mix_older,
    WinableType.Seven_Pairs: MahjongDetector.is_seven_pairs,
    WinableType.Mix_Unitary_Nine: MahjongDetector.is_mix_unitary_nine,
    WinableType.One_To_Nine: MahjongDetector.is_one_to_nine,
    WinableType.Avg3: MahjongDetector.is_avg3,
    WinableType.Two_Cup: MahjongDetector.is_two_cup,
    WinableType.Purity_Unitary_Nine: MahjongDetector.is_purity_unitary_nine,
    WinableType.Mix_Color: MahjongDetector.is_mix_color,
    WinableType.Purity_Color: MahjongDetector.is_purity_color,
    WinableType.Flow: MahjongDetector.is_flow,
    WinableType.Sky: MahjongDetector.is_sky,
    WinableType.Land: MahjongDetector.is_land,
    WinableType.Big_Yuan: MahjongDetector.is_big_yuan,
    WinableType.Words: MahjongDetector.is_words,
    WinableType.Green: MahjongDetector.is_green,
    WinableType.Purity_Old: MahjongDetector.is_purity_old,
    WinableType.All_Unitary_Nine: MahjongDetector.is_all_unitary_nine,
    WinableType.Small_Wind: MahjongDetector.is_small_wind,
    WinableType.Four_Gang: MahjongDetector.is_four_gang,
    WinableType.Flower: MahjongDetector.is_flower,
    WinableType.Four_Hidden_Bump: MahjongDetector.is_four_hidden_bump,
    WinableType.Four_Hidden_Bump_Single: MahjongDetector.is_four_hidden_bump_single,
    WinableType.All_Unitary_Nine_Full: MahjongDetector.is_all_unitary_nine_full,
    WinableType.Purity_Flower: MahjongDetector.is_purity_flower,
    WinableType.Big_Wind: MahjongDetector.is_big_wind,
    WinableType.Hand_Well: MahjongDetector.is_hand_well,
    WinableType.Dragon_Seven_Pair: MahjongDetector.is_dragon_seven_pairs,
    WinableType.Major_Bump: MahjongDetector.is_major_bump,
    WinableType.Gold_Hook: MahjongDetector.is_gold_hook
}