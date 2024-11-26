from py_utils import *
from mglobal import *

class MCType:
    I = 0
    O = 1
    W = 2
    Z = 3

    Names = ["I", "O", "W", "Z"]

    @staticmethod
    def get_card_list(types):
        cards = []
        for type in types:
            cards += [MC.I, MC.O, MC.W, MC.Z][type]
        return cards

    @staticmethod
    def str(type):
        return MCType.Names[type]

    @staticmethod
    def is_type(mc, type):
        if type == MCType.I:
            array = MC.I
        if type == MCType.O:
            array = MC.O
        if type == MCType.W:
            array = MC.W
        if type == MCType.Z:
            array = MC.Z
        return mc in array

    @staticmethod
    def get_type_of_mc(mc):
        if mc <= MC.I9 and mc >= MC.I1:
            return MCType.I
        elif mc <= MC.O9 and mc >= MC.O1:
            return MCType.O
        elif mc <= MC.W9 and mc >= MC.W1:
            return MCType.W
        else:
            return MCType.Z

class MC:
    I1 = 0
    I2 = 1
    I3 = 2
    I4 = 3
    I5 = 4
    I6 = 5
    I7 = 6
    I8 = 7
    I9 = 8
    
    O1 = 9
    O2 = 10
    O3 = 11
    O4 = 12
    O5 = 13
    O6 = 14
    O7 = 15
    O8 = 16
    O9 = 17
    
    W1 = 18
    W2 = 19
    W3 = 20
    W4 = 21
    W5 = 22
    W6 = 23
    W7 = 24
    W8 = 25
    W9 = 26

    Z1 = 27
    Z2 = 28
    Z3 = 29
    Z4 = 30
    Z5 = 31     # 中
    Z6 = 32     # 白
    Z7 = 33     # 发

    O = [O1, O2, O3, O4, O5, O6, O7, O8, O9]
    I = [I1, I2, I3, I4, I5, I6, I7, I8, I9]
    W = [W1, W2, W3, W4, W5, W6, W7, W8, W9]
    Z = [Z1, Z2, Z3, Z4, Z5, Z6, Z7]
    Wind = [Z1, Z2, Z3, Z4]
    Yuan = [Z5, Z6, Z7]
    Unitary_Nine = [O1, O9, I1, I9, W1, W9, Z1, Z2, Z3, Z4, Z5, Z6, Z7]
    Old = [O1, O9, I1, I9, W1, W9]
    Green = [I2, I3, I4, I6, I8, Z7]
    Major = [I2, I5, I8, O2, O5, O8, W2, W5, W8]

    @staticmethod
    def str(mc):
        if mc <= 8:
            return "I{}".format(mc + 1)
        elif mc <= 17:
            return "O{}".format(mc - 8)
        elif mc <= 26:
            return "W{}".format(mc - 17)
        else:
            return "Z{}".format(mc - 26)
        
    @staticmethod
    def is_O(mc):
        return mc >= MC.O1 and mc <= MC.O9
    
    @staticmethod
    def is_I(mc):
        return mc >= MC.I1 and mc <= MC.I9
    
    @staticmethod
    def is_W(mc):
        return mc >= MC.W1 and mc <= MC.W9
    
    @staticmethod
    def is_Z(mc):
        return mc >= MC.Z1 and mc <= MC.Z7

    @staticmethod
    def same_group(*args):
        g = NONE
        for arg in args:
            if g == NONE:
                g = arg // 9
                continue
            if g != arg // 9:
                return False
        return True

    @staticmethod
    def combine_mcs_with_unique(mcs1, mcs2, *args):
        mc_set = set()
        for mc in mcs1:
            mc_set.add(mc)
        for mc in mcs2:
            mc_set.add(mc)
        for i in range(len(args)):
            for mc in args[i]:
                mc_set.add(mc)
        return list(mc_set)

    @staticmethod
    def to_str(mcs):
        return ",".join([MC.str(mc) for mc in mcs])
    
class Op:
    Eat   = 0
    Bump  = 1
    Gang1 = 2
    Gang4 = 3
    Play  = 4
    Win   = 5
    Head  = 6
    Empty = 7
    Stand = 8

    @staticmethod
    def str(op):
        return ["eat", "bump", "gang1", "gang4", "play", "win", "head", "empty"][op]

class MCGroup:
    def __init__(self, mcs, op):
        self.mcs = mcs
        self.op = op
        
class Card:
    def __init__(self, mc, red):
        self.mc = mc
        self.red = red

    def get_name(self):
        return MC.str(self.mc) if not self.red else "r{}".format(MC.str(self.mc))

    def equal(self, card):
        return card.mc == self.mc and self.red == card.red

class CardGroup:
    
    def __init__(self, cards, op):
        self.cards = cards
        self.op = op

class CardList:

    @staticmethod
    def combine_lists_with_unique_mc(cards1, cards2, *args):
        cs = []
        mc_set = set()
        for card in cards1:
            if not mc_set.__contains__(card.mc):
                mc_set.add(card.mc)
                cs.append(card)
        for card in cards2:
            if not mc_set.__contains__(card.mc):
                mc_set.add(card.mc)
                cs.append(card)
        for i in range(len(args)):
            for card in args[i]:
                if not mc_set.__contains__(card.mc):
                    mc_set.add(card.mc)
                    cs.append(card)
        return cs

    @staticmethod
    def stata_types(cards):
        type_count = [0 for _ in range(4)]
        for card in cards:
            type_count[MCType.get_type_of_mc(card.mc)] += 1
        return type_count

    @staticmethod
    def add_card_with_sort(cards, card):
        binary_insert(cards, card, lambda x: x.mc)

    @staticmethod
    def to_str(cards):
        strs = []
        for card in cards:
            if isinstance(card, Card):
                strs.append(card.get_name())
            elif isinstance(card, CardGroup):
                strs.append(CardList.to_str(card.cards))
        strs = "[" + ",".join(strs) + "]"
        return strs

    @staticmethod
    def find_mc(cards, mc):
        result = []
        for card in cards:
            if card.mc == mc:
                result.append(card)
        return result

    @staticmethod
    def array2_to_str(cards_list):
        strs = []
        for cards in cards_list:
            strs.append(CardList.to_str(cards))
        return "[" + ",".join(strs) + "]"

    @staticmethod
    def sort_cards(cards):
        sorted(cards, key=lambda x: x.mc)

    @staticmethod
    def get_mc_map(cards):
        mc_map = {}
        for card in cards:
            if mc_map.get(card.mc) is None:
                mc_map[card.mc] = 1
            else:
                mc_map[card.mc] += 1
        return mc_map

    @staticmethod
    def from_mc(mcs):
        return [Card(mc, False) for mc in mcs]

    @staticmethod
    def remove_if_find(cards, r_cards):
        r = []
        for i in range(len(r_cards)):
            remove = None
            for inner_card in cards:
                if r_cards[i].equal(inner_card):
                    remove = inner_card
                    break
            assert(remove is not None)
            r.append(remove)
            cards.remove(remove)
        return r

    @staticmethod
    def find(cards, card):
        index = 0
        for c in cards:
            if c.equal(card):
                return index
            index += 1
        return NONE

class WinableType:
    Stand                     = 0   # 立直 (In Game)
    Broken_Unitary_Nine       = 1   # 断幺九
    Clear_River_Self          = 2   # 门前请自摸胡 (In Game)
    Self_Wind                 = 3   # 自风
    Ground_Wind               = 4   # 场风
    Zhong                     = 5   # 中
    Bai                       = 6   # 白
    Fa                        = 7   # 发
    Avg                       = 8   # 平胡
    One_Cup                   = 9   # 一杯口
    Grab_Gang                 = 10  # 抢杠 (In Game)
    Bloom                     = 11  # 岭上开花
    Torch_Moon                = 12  # 海底捞月
    Torch_Fish                = 13  # 河底捞鱼
    Once                      = 14  # 一发 (In Game)
    W_Stand                   = 15  # 两立直
    Bump3                     = 16  # 三色同刻
    Three_Gang                = 17  # 三杠子
    All_Bump                  = 18  # 对对胡
    Three_Hidden_Bump         = 19  # 三暗刻
    Small_Yuan                = 20  # 小三元
    Mix_Older                 = 21  # 混老头
    Seven_Pairs               = 22  # 七对子 (特殊牌型)
    Mix_Unitary_Nine          = 23  # 混全带幺九 
    One_To_Nine               = 24  # 一气通贯 
    Avg3                      = 25  # 三色同顺 
    Two_Cup                   = 26  # 二杯口
    Purity_Unitary_Nine       = 27  # 纯全带幺九 
    Mix_Color                 = 28  # 混一色 
    Purity_Color              = 29  # 清一色  
    Flow                      = 30  # 流局满贯 (In Game) (特殊规则)

    Sky                       = 31  # 天和 (In Game)
    Land                      = 32  # 地和 (In Game)
    Big_Yuan                  = 33  # 大三元
    Words                     = 34  # 字一色
    Green                     = 35  # 绿一色
    Purity_Old                = 36  # 清老头
    All_Unitary_Nine          = 37  # 国士无双 (特殊牌型)
    Small_Wind                = 38  # 小四喜
    Four_Gang                 = 39  # 四杠子
    Flower                    = 40  # 九莲宝灯
    Four_Hidden_Bump          = 41  # 四暗刻
    Four_Hidden_Bump_Single   = 42  # 四暗刻单骑
    All_Unitary_Nine_Full     = 43  # 国士无双十三面 (特殊牌型)
    Purity_Flower             = 44  # 纯正九莲宝灯
    Big_Wind                  = 45  # 大四喜

    Hand_Well                 = 46  # 普通胡
    Dragon_Seven_Pair         = 47  # 龙七对
    Major_Bump                = 48  # 将对
    Gold_Hook                 = 49  # 金钩钩

    CNames = [
        "立直", "断幺九", "门前请自摸胡", "自风", "场风", "中", "白", "发", "平胡", "一杯口", "抢杠",
        "岭上开花", "海底捞月", "河底捞鱼", "一发", "两立直", "三色同刻", "三杠子", "对对胡",
        "三暗刻", "小三元", "混老头", "七对子", "混全带幺九",  "一气通贯", "三色同顺", "二杯口", 
        "纯全带幺九", "混一色", "清一色", "流局满贯", "天和", "地和", "大三元", "字一色",
        "绿一色", "清老头", "国士无双", "小四喜", "四杠子", "九莲宝灯", "四暗刻", "四暗刻单骑",
        "国士无双十三面", "纯正九莲宝灯", "大四喜", "胡", "龙七对", "将对", "金钩钩"
    ]
    Names = [
        "Stand", "Broken_Unitary_Nine", "Clear_River_Self", "Self_Wind", "Ground_Wind", "Zhong",
        "Bai", "Fa", "Avg", "One_Cup", "Grab_Gang", "Bloom", "Torch_Moon", "Torch_Fish", "Once",
        "W_Stand", "Bump3", "Three_Gang", "All_Bump", "Three_Hidden_Bump", "Small_Yuan", "Mix_Older", 
        "Seven_Pairs", "Mix_Unitary_Nine", "One_To_Nine", "Avg3", "Two_Cup", "Purity_Unitary_Nine",
        "Mix_Color", "Purity_Color", "Flow", "Sky", "Land", "Big_Yuan", "Words", "Green", "Purity_Old",
        "All_Unitary_Nine", "Small_Wind", "Four_Gang", "Flower", "Four_Hidden_Bump", "Four_Hidden_Bump_Single",
        "All_Unitary_Nine_Full", "Purity_Flower", "Big_Wind", "Hand_Well", "Dragon_Seven_Pair", 
        "Major_Bump", "Gold_Hook"
    ]



    HAND_WELL_FREE = [Seven_Pairs, All_Unitary_Nine, All_Unitary_Nine_Full]

    @staticmethod
    def str(type):
        return WinableType.Names[type]

    @staticmethod
    def to_str(types):
        types = [WinableType.str(type) for type in types]
        return "[" + ",".join(types) + "]"
