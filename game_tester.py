from mahjong.sts4_mahjong import *
from mahjong.players import *
from mahjong.players import *
import mglobal
from mahjong.chuan_majong import *
from mahjong.mahjong_detector import *

# 001. 可以抢杠        √
# 002. 暗杠不能抢      √
# 003. 吃碰杠顺序确定  √
# 004. 杠上炮海底      √
# 005. 抢断后改变顺序  √
# 006. 吃碰杠完别抽牌  √
# 
# sts4
# 101. 场风           √
# 102. 自风           √
# 103. 立直操作       √
# 104. 立直当回合只能打出立直的牌   √
# 105. 立直后只能打出抽的牌         √
# 106. 一发在鸣牌后取消             √
# 107. 算分机制合理
# 108. 流局满贯测试
# 109. 连庄机制
# 110. 役种          √
# 111. 宝牌

# 1. 断幺判断有误
# 2. 四个字也能立直
# 3. 天胡已经被撤销（测试）
def test_sts4():
    config = Sts4Config()
    game = Sts4MGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def get_sts4game():
    config = Sts4Config()
    game = Sts4MGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    return game

def test_001(game):
    return [
        CardList.from_mc([
            MC.I1, MC.I1, MC.I1, MC.I1
        ]),
        CardList.from_mc([
            MC.I2, MC.I3, 
            MC.I4, MC.I4, MC.I4,
            MC.I5, MC.I5, MC.I5,
            MC.I6, MC.I6, MC.I6,
            MC.I7, MC.I7
        ]),
        [], []
    ]

def test_003(game):
    return [
        CardList.from_mc([
            MC.I1, MC.I2, MC.I3, 
            MC.I4, MC.I4, MC.I4,
            MC.I5, MC.I5, MC.I5,
            MC.I6, MC.I6, MC.O6,
            MC.O2, MC.O2
        ]),
        CardList.from_mc([
            MC.O1, MC.O3, 
            MC.O4, MC.O4, MC.O4,
            MC.O5, MC.O5, MC.O5,
            MC.O6, MC.O6, MC.O6,
            MC.O8, MC.O9
        ]),
        CardList.from_mc([
            MC.O2, MC.O2, MC.O2, 
            MC.W4, MC.W5,
            MC.W5, MC.W5, MC.W5,
            MC.W6, MC.W6, MC.W7,
            MC.W9, MC.W9
        ]),
        []
    ]

def get_random_hand():
    return CardList.from_mc([
        random.randint(0, MC.Z7) for _ in range(13)
    ])

def set_player_hands(game, cards):
    for i in range(4):
        for card in cards[i]:
            game.players[i].add_a_card(card)

def test_101():

    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile.add_card_to_end(Card(MC.I1, True))
    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)

    cards = CardList.from_mc([
        MC.I2, MC.I3, 
        MC.I4, MC.I4, MC.I4,
        MC.I5, MC.I5, MC.I5,
        MC.I6, MC.I6, MC.I6,
        MC.Z1, MC.Z1
    ])
    set_player_hands(game, [cards, get_random_hand(), get_random_hand(), get_random_hand()])
    game.start_game()

def test_101():

    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile.add_card_to_end(Card(MC.I1, True))
            self.standing[0] = 2
    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)

    cards = CardList.from_mc([
        MC.I2, MC.I3, 
        MC.I4, MC.I4, MC.I4,
        MC.I5, MC.I5, MC.I5,
        MC.I6, MC.I6,
        MC.Z1, MC.Z1, MC.Z1
    ])
    set_player_hands(game, [cards, get_random_hand(), get_random_hand(), get_random_hand()])
    game.start_game()


def test_1011():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile.add_card_to_end(Card(MC.I4, True))
            self.standing[0] = 2
    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)

    cards = CardList.from_mc([
        MC.I2, MC.I3, 
        MC.I4, MC.I4, MC.I4,

        MC.Z5, MC.Z5, MC.Z5,
        MC.Z6, MC.Z6, 
        MC.Z7, MC.Z7, MC.Z7
    ])
    set_player_hands(game, [cards, get_random_hand(), get_random_hand(), get_random_hand()])
    game.start_game()

def test_1012():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile.add_card_to_end(Card(MC.I4, True))
            self.standing[0] = 2
    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)

    cards = CardList.from_mc([
        MC.I2, MC.I3, 
        MC.I5, MC.I6, MC.I7,

        MC.I7, MC.I8, MC.I9,
        MC.I7, MC.I8, MC.I9,
        MC.I1, MC.I1
    ])
    CardList.sort_cards(cards)
    set_player_hands(game, [cards, get_random_hand(), get_random_hand(), get_random_hand()])
    game.start_game()

def test_1014():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.I6, True))

        def _init_player_hand_cards_for_round(self):
            cards = CardList.from_mc([
                MC.I6
            ])
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            self.players[0].table_cards = [
                CardGroup(CardList.from_mc([MC.I4, MC.I4, MC.I4, MC.I4]), Op.Gang4),
                CardGroup(CardList.from_mc([MC.I5, MC.I5, MC.I5, MC.I5]), Op.Gang4),
                CardGroup(CardList.from_mc([MC.O4, MC.O4, MC.O4, MC.O4]), Op.Gang4),
                CardGroup(CardList.from_mc([MC.W4, MC.W4, MC.W4]), Op.Bump),
            ]
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()
                
    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_1013():
        
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.I4, True))
    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)

    cards = CardList.from_mc([
        MC.I1, MC.I4, 
        MC.I8, MC.I9, MC.I7,

        MC.I7, MC.I8, MC.I9,
        MC.I7, MC.I8, MC.I9,
        MC.I1, MC.I4
    ])
    CardList.sort_cards(cards)
    set_player_hands(game, [cards, get_random_hand(), get_random_hand(), get_random_hand()])
    game.start_game()

def test_seven_pairs():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.I6, True))

        def _init_player_hand_cards_for_round(self):
            cards = CardList.from_mc([
                MC.I6,
                MC.I8, MC.I8,
                MC.I9, MC.I9,
                MC.I1, MC.I1,
                MC.O2, MC.O2,
                MC.O4, MC.O4,
                MC.Z2, MC.Z2
            ])
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_all_unitary_nine():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.I1, True))

        def _init_player_hand_cards_for_round(self):
            mcs = MC.Unitary_Nine
            mcs.remove(MC.I1)
            mcs.append(MC.Z1)
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_mix_older():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.W1, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I1,MC.I1,MC.I1, 
                MC.O1,MC.O1,MC.O1,
                MC.O9,MC.O9,MC.O9,
                MC.W1
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            self.players[0].table_cards = [
                CardGroup(CardList.from_mc([MC.I9, MC.I9, MC.I9]), Op.Bump),
            ]
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_mix_unitary_nine():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.Z1, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I1,MC.I2,MC.I3, 
                MC.O1,MC.O2,MC.O3,
                MC.O7,MC.O8,MC.O9,
                MC.Z1
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            self.players[0].table_cards = [
                CardGroup(CardList.from_mc([MC.I7, MC.I8, MC.I9]), Op.Eat),
            ]
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()


def test_avg3():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.Z1, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I1,MC.I2,MC.I3, 
                MC.I1,MC.I2,MC.I3,
                MC.W1,MC.W2,MC.W3,
                MC.Z1
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            self.players[0].table_cards = [
                CardGroup(CardList.from_mc([MC.W1, MC.W2, MC.W3]), Op.Eat),
            ]
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_green():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.I4, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I2,MC.I2,MC.I2, 
                MC.I3,MC.I3,MC.I3,
                MC.Z7, MC.Z7, MC.Z7,
                MC.I4
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            self.players[0].table_cards = [
                CardGroup(CardList.from_mc([MC.I6,MC.I6,MC.I6]), Op.Eat),
            ]
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_flower():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.I5, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I1,MC.I1,MC.I4, 
                MC.I6,MC.I6,MC.I7,
                MC.I8,MC.I9,MC.I9,
                MC.I9, MC.I1,MC.I2,MC.I3
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_full():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 13
            self.card_pile.add_card_to_end(Card(MC.Z5, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.Z7,MC.Z7,MC.Z7,
                MC.Z6,MC.Z6,MC.Z6,
                MC.Z3,MC.Z3,MC.Z3,
                MC.Z4,MC.Z4,
                MC.Z5,MC.Z5
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            self.players[0].hand_cards = cards
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_grab_gang():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile.add_card_to_end(Card(MC.Z5, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I1,
                MC.Z3,MC.Z6,MC.Z6,
                MC.O3,MC.O9,
                MC.O4,MC.Z4,
                MC.Z5,MC.Z7
            ]
            mcs2 = [
                MC.I2, MC.I3, 
                MC.I4, MC.I5, MC.I6,
                MC.Z5,MC.Z5,
                MC.Z6,MC.Z6, MC.Z6, 
                MC.Z7, MC.Z7,MC.Z7
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            for card in cards:
                self.players[0].add_a_card(card)
            self.players[0].table_cards = [
                CardGroup(CardList.from_mc([MC.I1, MC.I1, MC.I1]), Op.Bump)
            ]
            for card in CardList.from_mc(mcs2):
                self.players[1].add_a_card(card)
            for i in range(2):
                self.players[i + 2].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()


def test_bloom():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile._index = 14
            for card in self.card_pile.card_lists:
                card.mc = MC.I1
            self.card_pile.add_card_to_end(Card(MC.I1, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I1, MC.I1, MC.I1,
                MC.I2, MC.I3, MC.Z5,
                MC.I4, MC.I5, MC.I6,
                MC.Z5, 
                MC.Z7, MC.Z7,MC.Z7
            ]
            mcs2 = [
                MC.I2, MC.I3, 
                MC.I4, MC.I5, MC.I6,
                MC.Z5,MC.Z5,
                MC.Z6,MC.Z6, MC.Z6, 
                MC.Z7, MC.Z7,MC.Z7
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            for card in cards:
                self.players[1].add_a_card(card)

            self.players[1].card_river = [Card(MC.I1, False)]
            self.players[0].card_river = [Card(MC.I1, False)]
            for card in CardList.from_mc(mcs2):
                self.players[0].add_a_card(card)
            for i in range(2):
                self.players[i + 2].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test_stand():
    
    class Sts4DebugGame(Sts4MGame):
        def o_init_card_pile(self):
            super().o_init_card_pile()
            for card in self.card_pile.card_lists:
                card.mc = MC.I1
            self.card_pile.add_card_to_end(Card(MC.I1, True))

        def _init_player_hand_cards_for_round(self):
            mcs = [
                MC.I1, MC.I1, 
                MC.I2, MC.I3, MC.W1,
                MC.I4, MC.I5, MC.I6,
                MC.Z5, MC.Z5,
                MC.Z7, MC.Z7,MC.Z7
            ]
            cards = CardList.from_mc(mcs)
            CardList.sort_cards(cards)
            for card in cards:
                self.players[0].add_a_card(card)
            for i in range(3):
                self.players[i + 1].hand_cards = get_random_hand()

    config = Sts4Config()
    game = Sts4DebugGame(config)
    players = [ConsolePlayer("Player1"), ConsolePlayer("Player2"), ConsolePlayer("Player3"), ConsolePlayer("Player4")]
    game.add_player(players)
    game.start_game()

def test(key):
    
    map = {
        "001": test_001,
        "002": test_001,
        "003": test_003,
    }
    self_map = {
        "101": test_101,
        "1011": test_1011,
        "1012": test_1012,
        "1013": test_1013,
        "1014": test_1014,
        "seven_pair": test_seven_pairs,
        "all": test_all_unitary_nine,
        "mix": test_mix_older,
        "mix_unitary_nine": test_mix_unitary_nine,
        "avg3": test_avg3,
        "green": test_green,
        "flower": test_flower,
        "full": test_full,
        "grab_gang": test_grab_gang,
        "bloom": test_bloom,
        "stand": test_stand
        
    }
    if map.get(key) != None:
        game = get_sts4game()
        cards = map[key](game)
        for i in range(4):
            for card in cards[i]:
                game.players[i].add_a_card(card)
        game.start_game()
    else:
        self_map[key]()

# hand can be well
def test_hand_can_be_well_mcs(mcs):
    cards = []
    for mc in mcs:
        cards.append(Card(mc, False))
    well = HandCardCanBeWell(cards, False)
    listens = well.perform()
    for mc in listens:
        print(mc)

def test_hand_can_be_well():

    mcs1 = [
        MC.W1,MC.W1,MC.W1,MC.W2,MC.W3,MC.W4,MC.W5,MC.W6,MC.W7,MC.W8,MC.W9,MC.W9,MC.W9
    ]
    mcs2 = [
        MC.W2,MC.W2,MC.W2,MC.W3,MC.W3,MC.W3,MC.W4,MC.W4,MC.W5,MC.W5,MC.W6,MC.W6,MC.W6
    ]
    mcs3 = [
        MC.Z2,MC.Z2,MC.Z2,MC.Z3,MC.Z3,MC.Z3,MC.Z4,MC.Z4,MC.Z5,MC.Z5,MC.Z6,MC.Z6,MC.Z6
    ]
    mcs4 = [
        MC.W2,MC.W2, MC.W2,MC.W3,MC.W3,MC.W4,MC.W5,MC.W6,MC.W6,MC.W7,MC.W8,MC.W8,MC.W8,
    ]
    test_hand_can_be_well_mcs(mcs1)
    print()
    test_hand_can_be_well_mcs(mcs2)
    print()
    test_hand_can_be_well_mcs(mcs3)
    print()
    test_hand_can_be_well_mcs(mcs4)

# ====================================================
def logger_test():
    config = Sts4Config()
    game = Sts4MGame(config, "random")
    names = ["lyx", "wtf", "lm", "cyy"]
    for i in range(4):
        game.add_player(RandomPlayer(names[i]))
    game.start_game()
    logger.log_to_file("log/test.txt")

# ====================================================
def test_chuan():
    config = ChuanConfig()
    game = ChuanMajong(config)
    names = ["lyx", "wtf", "lm", "cyy"]
    for i in range(4):
        game.add_player(RandomPlayer(names[i]))
    game.start_game()
    logger.log_to_file("log/chuan_test.txt")

# ===================================================
def test_player():
    class DebugGame(ChuanMajong):
        def _on_someone_is_win(self, winner_list, items, sources):
            for winner in winner_list:
                print("===")
                print(CardList.to_str(self.players[winner].hand_cards))
            return super()._on_someone_is_win(winner_list, items, sources)

    config = ChuanConfig()
    config.basic_score = 1
    config.start_scores = [10000, 10000, 10000, 10000]
    game = ChuanMajong(config)

    game.add_player(ChuanHeuristicPlayer("lyx-Heuristic"))
    game.add_player(ChuanPurityColorPlayer("wtf-Purity"))
    game.add_player(ChuanSevenPairsPlayer("lm-SevenPairs"))
    game.add_player(RandomPlayer("cyy-Random"))

    game.start_game()
    logger.log_to_file("log/chuan_player.txt")

def purity_color():
    class DebugGame(ChuanMajong):

        def o_init_card_pile(self):
            super().o_init_card_pile()
            self.card_pile.add_card_to_end(Card(MC.I6, False))
            self.card_pile.add_card_to_end(Card(MC.O4, False))
            self.card_pile.add_card_to_end(Card(MC.O4, False))
            self.card_pile.add_card_to_end(Card(MC.O6, False))
            self.card_pile.add_card_to_end(Card(MC.O8, False))

        def _on_someone_is_win(self, winner_list, items, sources):
            return super()._on_someone_is_win(winner_list, items, sources)

    config = ChuanConfig()
    config.max_iter = 1
    logger = Logger("log/test.txt", True)
    game = DebugGame(config, "purity")
    game.set_logger(logger)
    players = [ChuanPurityColorPlayer("p1"), ChuanPurityColorPlayer("p2"), ChuanPurityColorPlayer("p3"), ChuanPurityColorPlayer("p4")]
    players[0].hand_cards = CardList.from_mc([
        MC.I2, MC.I2, MC.I2, 
        MC.I3, MC.I3, MC.I3, 
        MC.I5, MC.I5, MC.I5, 
        MC.I6
    ])

    cards = CardList.from_mc([
        MC.W1, MC.W1, MC.W2, MC.W2, MC.W3,
        MC.W5, MC.W5, MC.W5, MC.W6, 
        MC.W8, MC.W9, MC.W9, MC.W9  
    ])
    for i in range(3):
        players[i + 1].hand_cards = copy.deepcopy(cards)
    players[0].table_cards = [CardGroup(CardList.from_mc([
        MC.I1, MC.I1, MC.I1, MC.I1
    ]), Op.Gang4)]
    #detector = MahjongDetector(players[0].hand_cards, players[0].table_cards, Card(MC.I6, False), None, default_MD_package(), None)


    for player in players:
        game.add_player(player)
    game.start_game()
    #logger.log_to_file("log/test.txt")



purity_color()