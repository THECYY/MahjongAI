from py_utils import *
from mglobal import *
from .card import *
from .mahjong_detector import *
from .heuristic import *

class Rule:
    activate_winable_types = []     # 可以使用的获胜条件
    restrain = {}                   # 获胜类型之间的抑制方法
    ops_in_turn = []                # 在玩家回合能够进行的操作列表
    ops_for_card = []               # 对别的玩家打出的牌能够响应

    # 获取map，key是WinabeType，value是Detector的函数返回的值
    def get_winable_map(self, hand_cards, table_cards, last_card, package, detector=None, hand_well=None):
        if detector == None:
            detector = MahjongDetector(hand_cards, table_cards, last_card, None, package, hand_can_well=hand_well)
        results = {}
        for type in self.activate_winable_types:
            results[type] = MD_FN_MAP[type](detector)
        if not detector.hand_well:
            for key in results.keys():
                if key not in WinableType.HAND_WELL_FREE:
                    results[key] = 0
        for type in self.restrain.keys():
            if results[type]:
                for r_type in self.restrain[type]:
                    results[r_type] = 0
        remove = []
        for type in results.keys():
            if not results[type]:
                remove.append(type)
        for r in remove:
            results.pop(r)
        return results, detector

    def hand_well_listenable(self, hand_cards, hand_can_be_well: HandCardCanBeWell=None):
        if hand_can_be_well is None:
            hand_can_be_well = HandCardCanBeWell(hand_cards, False)
        hand_well_result = hand_can_be_well.perform()
        if len(hand_well_result) != 0:
            return True, [result[0] for result in hand_well_result]
        return False, []

    def seven_pair_listenable(self, hand_cards):
        mc_map = CardList.get_mc_map(hand_cards)
        pairs = 0
        only_one = NONE
        for key in mc_map.keys():
            if mc_map[key] == 2:
                pairs += 1
            else:
                only_one = key
        return pairs == 6, [only_one] if pairs == 6 else []
    
    def all_unitary_nine_listenable(self, hand_cards):
        if len(hand_cards) != 13:
            return False, []

        mcs = [0 for _ in MC.Unitary_Nine]
        for card in hand_cards:
            if card.mc not in MC.Unitary_Nine:
                return False, []
            mcs[MC.Unitary_Nine.index(card.mc)] += 1
        only_zero = NONE
        for i in range(len(mcs)):
            if not mcs[i]:
                only_zero = i
                break
        return True, [only_zero] if only_zero != NONE else MC.Unitary_Nine

    def dragon_seven_pair_litenable(self, hand_cards, hand_can_be_well):
        if len(hand_cards) != 13:
            return False, []
        
        map = CardList.get_mc_map(hand_cards)
        four, three, pair, one = 0, 0, 0, NONE
        for key in map.keys():
            if map[key] == 2:
                pair += 1
            elif map[key] == 3:
                three += 1
                one = key
            elif map[key] == 4:
                four += 1
            elif map[key] == 1:
                one = key        

        if pair + four * 2 == 6 and four != 0:
            return True, [one]
        elif pair + four * 2 == 5 and three == 1:
            return True, [one]
        else:
            return False, []

    def listen(self, hand_cards, hand_can_be_well: HandCardCanBeWell=None, args=None):
        r, mcs = self.hand_well_listenable(hand_cards, hand_can_be_well)
        #if len(mcs) != 0:
        #    logger.trace("Listen hand well {}".format(MC.to_str(mcs)))
        
        mcs_seven = []
        if WinableType.Seven_Pairs in self.activate_winable_types:
            rs, mcs_seven = self.seven_pair_listenable(hand_cards)
            r = rs or r
            #if rs:
            #    logger.trace("Listen seven pairs {}".format(MC.to_str(mcs_seven)))

        mcs_dseven = []
        if WinableType.Dragon_Seven_Pair in self.activate_winable_types:
            rs, mcs_dseven = self.dragon_seven_pair_litenable(hand_cards, hand_can_be_well)
            r = rs or r
            #if rs:
            #    logger.trace("Listen dragon seven pairs {}".format(MC.to_str(mcs_dseven)))

        mcs_all_un = []
        if WinableType.All_Unitary_Nine_Full in self.activate_winable_types or \
            WinableType.All_Unitary_Nine in self.activate_winable_types:
            rs, mcs_all_un = self.all_unitary_nine_listenable(hand_cards)
            r = rs or r
            #if rs:
            #    logger.trace("Listen all unitary nine {}".format(MC.to_str(mcs_all_un)))

        mcs = MC.combine_mcs_with_unique(mcs, mcs_dseven, mcs_seven, mcs_all_un)
        #if len(mcs) != 0:
        #    logger.trace("Above cards {}".format(CardList.to_str(hand_cards)))
        return r, mcs

    def get_next_mc(self, mc):
        if mc >= MC.O1 and mc <= MC.O9:
            return ((mc - MC.O1 + 1) % 9) + MC.O1
        if mc >= MC.I1 and mc <= MC.I9:
            return ((mc - MC.I1 + 1) % 9) + MC.I1
        if mc >= MC.W1 and mc <= MC.W9:
            return ((mc - MC.W1 + 1) % 9) + MC.W1
        if mc >= MC.Z1 and mc <= MC.Z7:
            return ((mc - MC.Z1 + 1) % 7) + MC.Z1

    def get_previous_mc(self, mc):
        if mc >= MC.O1 and mc <= MC.O9:
            return ((mc - MC.O1 + 8) % 9) + MC.O1
        if mc >= MC.I1 and mc <= MC.I9:
            return ((mc - MC.I1 + 8) % 9) + MC.I1
        if mc >= MC.W1 and mc <= MC.W9:
            return ((mc - MC.W1 + 8) % 9) + MC.W1
        if mc >= MC.Z1 and mc <= MC.Z7:
            return ((mc - MC.Z1 + 6) % 7) + MC.Z1

    def get_next_adj_mc(self, mc):
        if mc >= MC.O1 and mc <= MC.O8:
            return mc + 1
        if mc >= MC.I1 and mc <= MC.I8:
            return mc + 1
        if mc >= MC.W1 and mc <= MC.W8:
            return mc + 1
        return -1

    def get_previous_adj_mc(self, mc):
        if mc >= MC.O2 and mc <= MC.O9:
            return mc - 1
        if mc >= MC.I2 and mc <= MC.I9:
            return mc - 1
        if mc >= MC.W2 and mc <= MC.W9:
            return mc - 1
        return -1

    def eatable(self, player, card):
        l = self.get_previous_adj_mc(card.mc)
        ll = self.get_previous_adj_mc(l) if l != NONE else NONE
        r = self.get_next_adj_mc(card.mc)
        rr = self.get_next_adj_mc(r) if r != NONE else NONE
        llc = CardList.find_mc(player.hand_cards, ll) if ll != NONE else []
        lc = CardList.find_mc(player.hand_cards, l) if l != NONE else []
        rc = CardList.find_mc(player.hand_cards, r) if r != NONE else []
        rrc = CardList.find_mc(player.hand_cards, rr) if rr != NONE else []
        eat = []
        if len(llc) > 0 and len(lc) > 0:
            eat.append([llc[0], lc[0]])
        if len(lc) > 0 and len(rc) > 0:
            eat.append([lc[0], rc[0]])
        if len(rc) > 0 and len(rrc) > 0:
            eat.append([rc[0], rrc[0]])
        return len(eat) > 0, eat

    def bumpable(self, player, card):
        count = player.query_mc_num(card.mc)
        bumple = count is not None and count >= 2
        bump_array = []
        if bumple:
            cards = CardList.find_mc(player.hand_cards, card.mc)
            red_card = None 
            for c in cards:
                if c.red:
                    red_card = c
                    break
            if red_card is None or count == 2:
                bump_array.append(cards[:2])
            else:
                # with red
                for c in cards:
                    if not c.red:
                        bump_array.append([c])
                        bump_array.append([c])
                        break
                bump_array[0].append(red_card)
                # without red
                for c in cards:
                    if not c.red and c is not bump_array[1][0]:
                        bump_array[1].append(c)
                        break
        return bumple, bump_array

    def gang1able(self, player, card):
        count = player.query_mc_num(card.mc)
        gang1able = count is not None and count >= 3
        gang = []
        if gang1able:
            gs = CardList.find_mc(player.hand_cards, card.mc)
            assert(len(gs) == 3)
            gang.append(gs)
        return gang1able, gang

    def gang4able(self, player, card_pile):
        if card_pile.exhaushed():
            return False, []

        # 暗杠
        gang4 = []
        for key in player.mcs():
            if player.query_mc_num(key) >= 4:
                gang4.append(CardList.find_mc(player.hand_cards, key))

        # 加杠
        bumps = [group.cards[0].mc if group.op == Op.Bump else NONE for group in player.table_cards]
        for key in player.mcs():
            if key in bumps:
                gang4.append(CardList.find_mc(player.hand_cards, key))
        return len(gang4) > 0, gang4

    def standable(self, hand_cards):
        playable_cards = []
        if len(hand_cards) != 14:
            return False, []
        for i in range(len(hand_cards)):
            copy_cards = copy.deepcopy(hand_cards)
            copy_cards.remove(copy_cards[i])
            hand_card_can_be_well = HandCardCanBeWell(copy_cards)
            results = hand_card_can_be_well.perform()
            if len(results) != 0:
                playable_cards.append(hand_cards[i])
                continue
            result, cards = self.seven_pair_listenable(copy_cards)
            if result:
                playable_cards.append(hand_cards[i])
                continue
            result, cards = self.all_unitary_nine_listenable(copy_cards)
            if result:
                playable_cards.append(hand_cards[i])
                continue

        return len(playable_cards) != 0, playable_cards

    def acceptable(self, player, game, package, hand_can_be_well=None, args=None):
        cards = copy.deepcopy(player.hand_cards)
        if len(cards) % 3 == 2:
            first = cards[0]
            cards.remove(first)
            listenable, listen_list = self.listen(cards, None, args=args)
            if not listenable or first.mc not in listen_list:
                return False
        elif len(cards) % 3 == 1:
            listenable, listen_list = self.listen(cards, None, args=args)
            if not listenable or game.active_card.mc not in listen_list:
                return False, [], None
        return True

    # winable, items, detector
    def o_winable(self, player, game, package, hand_can_be_well=None):
        raise NotImplementedError()

    # 计算分数, 必须重载
    def o_calculate_score(self, game, winner_list, items, source_player_index):
        r'''
            需要返回一个列表, 对应每一个玩家的得分
            每一次有人胡牌都会调用
        '''
        raise NotImplementedError()

    # 判断游戏是否已经结束
    def o_judge_round_is_over(self, game):
        r'''
            需要返回布尔值，代表一轮游戏是否结束
            每一次有人胡牌都会调用
        '''
        raise NotImplementedError()

class MCardPile:
    def __init__(self, masked=0):   # 有多少牌被隐藏
        self.card_lists = []
        self._index = 0
        self.masked = masked

    def get_mcs_of_indices(self, indices):
        mcs = []
        for i in indices:
            mcs.append(self.card_lists[i].mc)
        return mcs

    def add_card_to_end(self, card):
        self._index += 1
        self.card_lists.insert(self._index, card)

    def init_by_types(self, types):
        mc_list =  MCType.get_card_list(types)
        for mc in mc_list:
            self.card_lists.append(Card(mc, False))
            self.card_lists.append(Card(mc, False))
            self.card_lists.append(Card(mc, False))
            self.card_lists.append(Card(mc, False))
        self._index = len(self.card_lists) - 1

    def init_by_default(self):
        self.init_by_types([MCType.I, MCType.O, MCType.W, MCType.Z])
        self._index = len(self.card_lists) - 1

    def init_by_cards(self, cards):
        self.card_lists = cards
        self._index = len(self.card_lists) - 1

    def shuffle(self):
        random.shuffle(self.card_lists)

    def exhaushed(self):
        return self._index < self.masked

    def draw_cards(self, count=1):
        if self._index - count + 1 >= self.masked:
            cards = self.card_lists[(self._index - count + 1):(self._index + 1)]
            self._index -= count
            return cards
        else:
            return None

    def draw_a_card_from_maksed(self):
        card = self.card_lists[0]
        self.card_lists[0] = self.card_lists[self.masked]
        for i in range(self.masked, self._index):
            self.card_lists[i] = self.card_lists[i + 1]
        self._index -= 1
        return card

    def get_masked_cards(self):
        return self.card_lists[:self.masked]

    def get_unmasked_cards(self):
        return self.card_lists[self.masked:]

class Player:
    
    def __init__(self, name):
        self.name = name
        self.clear_cards()
        
    def clear_cards(self):
        self.hand_cards = []        # Cards
        self.table_cards = []       # CardGroups
        self.card_river = []        # Cards

    @property
    def _mc_map(self):
        return CardList.get_mc_map(self.hand_cards)

    def mcs(self):
        return self._mc_map.keys()

    def get_gang_num(self):
        count, hand = 0, 0
        for group in self.table_cards:
            if group.op in [Op.Gang1, Op.Gang4]:
                count += 1
        map = self._mc_map
        for key in map.keys():
            if map[key] >= 4:
                hand += 1
        return count, hand  

    def query_mc_num(self, mc):
        count = self._mc_map.get(mc)
        return 0 if count is None else count

    def draw_a_card(self, card_pile: MCardPile):
        card = card_pile.draw_cards(1)[0]
        return self.add_a_card(card)

    def add_a_card(self, card: Card):
        CardList.add_card_with_sort(self.hand_cards, card)
        return card

    def showable_info_by_self(self):
        return  [
            " Hand cards: {}".format(CardList.to_str(self.hand_cards)),
            "Table cards: {}".format(CardList.to_str(self.table_cards)),
            " Card river: {}".format(CardList.to_str(self.card_river)),
        ]
    
    def showable_info_by_others(self):
        return  [
            "Table cards: {}".format(CardList.to_str(self.table_cards)),
            " Card river: {}".format(CardList.to_str(self.card_river)),
        ]

    def play_a_card(self, card):
        if isinstance(card, int):
            c = self.hand_cards[card]
        else:
            assert(isinstance(card, Card))
            c = card
        CardList.remove_if_find(self.hand_cards, [c])
        self.card_river.append(c)
        return c

    def gang4(self, gang, card_pile: MCardPile):
        if len(gang) == 4:      # 暗杠
            CardList.remove_if_find(self.hand_cards, gang)
            self.table_cards.append(CardGroup(gang, Op.Gang4))
        elif len(gang) == 1:
            CardList.remove_if_find(self.hand_cards, gang)
            for group in self.table_cards:
                if group.op == Op.Bump and group.cards[0].mc == gang[0].mc:
                    group.cards.append(gang[0])
                    group.op = Op.Gang1
        new_card = card_pile.draw_a_card_from_maksed()
        self.add_a_card(new_card)
        return new_card

    def hand2table(self, cards, card, op):
        CardList.remove_if_find(self.hand_cards, cards)
        cards.append(card)
        self.table_cards.append(CardGroup(cards, op))

    def gang1(self, gang, card:Card, card_pile: MCardPile):
        self.hand2table(gang, card, Op.Gang1)
        new_card = card_pile.draw_a_card_from_maksed()
        self.add_a_card(new_card)
        return new_card

    def bump(self, bump, card):
        self.hand2table(bump, card, Op.Bump)

    def eat(self, eat, card):
        self.hand2table(eat, card, Op.Eat)

    def o_make_decision_to_cpg(self, game, card, choice, eat, dump, gang):
        raise NotImplementedError()
    
    def o_make_decision_to_gang4(self, game, gangs):
        raise NotImplementedError()

    def o_make_decision_to_play(self, game, playable_cards):
        raise NotImplementedError()

    def o_make_decision_to_win(self, game, win_types, package):
        raise NotImplementedError()

    def o_make_decision_to_stand(self, game, playable_cards):
        raise NotImplementedError()

    def o_make_decision_to_lack(self, game):
        raise NotImplementedError()

    def o_make_decision_to_change_three(self, cardss):
        raise NotImplementedError()

class Config:
    max_player = 0      # 玩家数
    max_iter = 0        # 最大游戏轮数
    start_scores = [] # 初始分数

class MGame:

    def __init__(self, config, name="test"):
        self.config = config
        self.name = name
        self.o_init_rule()
        self._init_game_info()
        assert(Op.Bump in self.rule.ops_for_card)

    def set_logger(self, logger):
        self.logger = logger

    # 初始化当前的规则
    def o_init_rule(self):
        r'''
            需要更新一个参数
            1. self.rule 当前的游戏规则对象
        '''
        self.rule = Rule()
        raise NotImplementedError()
        
    def _init_game_info(self):
        self.players = []                           # 玩家
        self.max_player = self.config.max_player    # 玩家数量
        self.max_iter = self.config.max_iter        # 最大游戏轮数
        self.iter = 0                               # 当前游戏轮数
        self.main_player = 0                        # 当前主玩家
        self.next_main_player = 0                   # 下一轮游戏的主玩家
        self.win_info = {}                          # 游戏胜利信息
        self.scores = self.config.start_scores[:self.max_player] # 初始分数

    def _game_can_be_start(self):
        return len(self.players) == self.max_player
        
    # 启动游戏
    def start_game(self):
        if not self._game_can_be_start():
            self.logger.warning("Failed to start game")
        else:
            self.logger.trace("================ Game Start ================")
            self.logger.importent(" ".join([self.players[i].name for i in range(self.max_player)]))
            self._run_game()

    # 运行游戏
    def _run_game(self):
        while self.iter < self.max_iter:
            self.logger.importent("---------------- Start {}/{} ----------------".format(self.iter + 1, self.max_iter))
            self._init_round()
            end = self._run_round()
            if end: break
            self.change_main_player()
        self.o_end_game()

    # 交换主玩家，指示当前的主玩家, 默认轮流做庄家
    def change_main_player(self):
        self.main_player = self.next_main_player

    # 添加分数
    def add_scores(self, scores):
        self.logger.importent("Add scores: " + ",".join(["{}".format(score) for score in scores]))
        end = False
        for i in range(len(scores)):
            self.scores[i] += scores[i]
            end = end or self.scores[i] < 0
        return 0 if not end else 1

    # 交换当前进行回合的玩家
    def change_round(self):
        self.current = self.next
        self.next = (self.current + 1) % self.max_player

    # 默认是轮流做庄
    def o_decision_next_main_player(self):
        r'''
            需要更新两个参数
            1. self.next_main_player 决定下一个主玩家的下标 (默认为轮流坐庄)
            2. self.iter 迭代计数, 是否完成了一轮 (用于连庄)
        '''
        if self.win_info.get(self.iter) != None:
            if len(self.win_info[self.iter]) == 1:
                self.next_main_player = self.win_info[self.iter][0][0]
                self.iter += 1
            else:
                self.next_main_player = self.current
                self.iter += 1
        
    # 初始化卡牌组
    def o_init_card_pile(self):
        r'''
            需要更新一个参数
            1. self.card_pile 使用的牌库
        '''
        self.card_pile = MCardPile(0)
        raise NotImplementedError()

    def o_clear_players_cards(self):
        for i in range(self.max_player):
            self.players[i].clear_cards()

    # round指的是一轮游戏
    def _init_round(self):
        self.hand_card_can_be_well = [None for _ in range(self.max_player)]
        self.current = self.main_player                         # 最开始的玩家为主玩家
        self.current_player = self.players[self.current]        # 当前玩家
        self.next = (self.current + 1) % self.max_player        # 轮流打牌
        self.draw_before_round = True                           # 回合开始前要抽牌
        self.active_card = None                                 # 当前激活的卡 (被打出，被抽到)
        self.last_drawing_card = None                           # 最后抽到的卡
        self.gang_in_turn = False                               # 本回合是否有杠
        self.next_player_is_gang = False                        # 下一个玩家是否杠了自己的牌
        self.standing = [0 for _ in range(self.max_player)]     # 是否立直
        self.once = [0 for _ in range(self.max_player)]         # 是否一发
        self.must_play_in = None                                # 最后打牌时必须从中打的牌
        self.o_init_round_basic_props()                         # 初始化属性
        self.o_init_card_pile()                                 # 初始化卡牌组
        self._init_player_hand_cards_for_round()                # 初始手牌
        self._init_player_hand_card_analysis()                  # 初始化手牌分析（用于加速）
        self._log_init_round_info()                             # 日志化
    
    # 回合初始化函数 [1] 初始化基础属性
    def o_init_round_basic_props(self):
        r'''
            定义额外的内部属性, 用于扩展游戏
        '''
        self.logger.error("MGame can't init basic props")
        raise NotImplementedError()

    # 回合初始化函数 [2] 初始化手牌
    def _init_player_hand_cards_for_round(self):
        for i in range(self.max_player):
            while len(self.players[i].hand_cards) + 3 * len(self.players[i].table_cards) < 13 and not self.card_pile.exhaushed():
                self.players[i].draw_a_card(self.card_pile)

    # 回合初始化函数 [3] 初始化手牌分析算法
    def _init_player_hand_card_analysis(self):
        for i in range(self.max_player):
            self.hand_card_can_be_well[i] = HandCardCanBeWell(self.players[i].hand_cards, False)

    # 回合初始化函数 [4] log方法
    def _log_init_round_info(self):
        self.logger.trace("round {} is already".format(self.iter + 1))
        for i in range(4):
            self.logger.trace("Player {} get cards {}".format(self.players[i].name, CardList.to_str(self.players[i].hand_cards)))
        self.logger.trace("masked card: {}".format(CardList.to_str(self.card_pile.get_masked_cards())))
        self.logger.trace("  card pile: {}".format(CardList.to_str(self.card_pile.get_unmasked_cards())))
        self.logger.importent("Player {} is main".format(self.players[self.main_player].name))

    # 钩函数
    def _on_someone_is_win(self, winner_list, items, sources):
        scores = self.rule.o_calculate_score(self, winner_list, items, sources)
        self.add_scores(scores)
        return scores

    # 将获胜玩家的信息记录在
    def o_record_winner_list(self, winner_list, items, source):
        r'''
            需要更新一个参数
            1. self.win_info 一个记录了胜利者的信息的字典
        '''
        for i in range(len(winner_list)):
            if self.win_info.get(self.iter) is None:
                self.win_info[self.iter] = []
            self.win_info[self.iter].append([winner_list[i], items[i], source])

    # 记录胜利者的信息
    def _process_winner_list(self, winner_list, items, source):
        self.o_record_winner_list(winner_list, items, source)
        for i in range(len(winner_list)):
            self.logger.trace("Player {} win {}".format(self.players[winner_list[i]].name, WinableType.to_str(items[i])))
            self.logger.trace("Source is player {}".format(self.players[source].name))
        self._on_someone_is_win(winner_list, items, source)
        return self.rule.o_judge_round_is_over(self)

    # 开始运行一个回合
    def _run_round(self):
        end_game = False
        while True:
            end = False
            for score in self.scores:
                if score < 0:
                    end_game = True
            if end_game: 
                self.logger.trace("Some player has no score, game will end")
                self.logger.trace("Scores: {}".format(",".join(["{}".format(score) for score in self.scores])))
                break
            result = self._can_current_player_start_his_time()          
            if result == 0: break
            if result == 2: 
                self.change_round()
                continue
            self._player_start_his_round()                              # 玩家开始他的回合

            # 如果需要抽牌则抽一张牌
            if self.draw_before_round:                      
                end = self._player_draw_a_card(self.current)        # must play in 可以在此设置
                if end == 1: break
                if end == 2: 
                    self.change_round()
                    continue  
                       
            # 结束一发
            self.draw_before_round = True
            if not self.standing[self.current]:
                self.must_play_in = None 
            self.once[self.current] = False
                
            if not self.standing[self.current]:

                # 获取在当前回合能够进行的操作列表，查看是够能进行加杠或者暗杠
                if Op.Gang4 in self.rule.ops_in_turn:
                    end = self._game_and_player_gang4(self.current)
                    if end == 1: break
                    if end == 2: 
                        self.change_round()
                        continue

                # 询问是否立直
                if Op.Stand in self.rule.ops_in_turn:
                    self._standable_for_current_player()

            # 询问打牌
            if Op.Play in self.rule.ops_in_turn:
                end = self._playable_for_current_player()
                if end == 1: break
                if end == 2: 
                    self.change_round()
                    continue
            self.change_round()

        if end_game: return True
        self.o_end_round()
        return False
        
    # 回合运行函数 [1] 玩家是否能开启他的回合
    def _can_current_player_start_his_time(self):
        if self.card_pile.exhaushed() and self.draw_before_round:
            self.logger.trace("Card pile is exhaushed")
            return False
        return True

    # 回合运行函数 [2] 玩家开始他的回合
    def _player_start_his_round(self):
        self.current_player = self.players[self.current]
        assert(isinstance(self.current_player, Player))
        self.logger.trace("### Player {} start his round ###".format(self.current_player.name))
        self.gang_in_turn = self.next_player_is_gang
        self.next_player_is_gang = False

    # 回合运行函数 [3] 玩家从牌库抽一张牌
    def _player_draw_a_card(self, player_index):
        player = self.players[player_index]
        assert(isinstance(player, Player))
        card = player.draw_a_card(self.card_pile)
        self.must_play_in = [card]
        return self._on_player_draw_a_card(player_index, card, False, self.hand_card_can_be_well[player_index])
    
    # 回合运行函数 [4] 暗杠 加杠 
    def _game_and_player_gang4(self, player_index):
        player = self.players[player_index]
        assert(isinstance(player, Player))
        
        while True:
            gang4able, gang4s = self.rule.gang4able(player, self.card_pile)
            if not gang4able:
                break
            gang4, gang = player.o_make_decision_to_gang4(self, gang4s)
            if not gang4:
                self.logger.trace("Player {} has gang4 choice {}, but not gang".format(player.name, CardList.array2_to_str(gang4s)))
                break
            self.logger.trace("Player {} has gang4 choice {}, he choose index {}".format(player.name, CardList.array2_to_str(gang4s), gang))
            gang = gang4s[gang]
            if len(gang) == 1: 
                end = self._on_player_play_a_card(player_index, gang[0], False, True)
                if end:
                    CardList.remove_if_find(player.hand_cards, gang[0])
                    return end
            new_card = player.gang4(gang, self.card_pile)
            end = self._on_gang4(player_index, gang) 
            if end: return end 
            end = self._on_player_draw_a_card(player_index, new_card, True)
            if end: return end

    # 回合运行函数 [4] 立直
    def _standable_for_current_player(self):
        stanable, standable_cards = self.rule.standable(self.current_player.hand_cards)
        if stanable:
            stand = self.current_player.o_make_decision_to_stand(self, standable_cards)
            self.standing[self.current] = stand 
            if stand:
                self.standing[self.current] = 1 if len(self.current_player.card_river) != 0 else 2
                self.must_play_in = standable_cards
                self.once[self.current] = True
            self.logger.trace("Player {} choice {}".format(self.current_player.name, "stand" if stand else "not stand"))

    # 回合运行函数 [5] 出牌
    def _playable_for_current_player(self):
        played_card = None
        card_source = NONE
        if self.must_play_in is None:
            index = self.current_player.o_make_decision_to_play(self, self.current_player.hand_cards)
            card = self.current_player.hand_cards[index]
            card_source = 0
        else:
            if len(self.must_play_in) == 1:
                card = self.must_play_in[0]
            else:
                index = self.current_player.o_make_decision_to_play(self, self.must_play_in)
                card = self.must_play_in[index]
            card_source = 1
        self.logger.trace("Player {} choice play card {} from {}".format(self.current_player.name, card.get_name(), ["hand", "must"][card_source]))
        played_card = self.current_player.play_a_card(card)
        return self._on_player_play_a_card(self.current, played_card, self.gang_in_turn, False)

    def _on_gang4(self, player_index, gang):
        self.logger.trace("Player hand cards (after gang4): {}".format(CardList.to_str(self.players[player_index].hand_cards)))
        self.gang_in_turn = True
        for i in range(self.max_player):
            self.once[i] = False
        return False

    def o_can_be_check_ckp_when_someone_play_card(self, player_index):
        return True

    def _ckpable_for_all_user(self, player_index, played_card):
        index, op, op_cards, ckpp_index = -1, Op.Empty, None, NONE
        for i in range(self.max_player - 1):
            if self.standing[(i + player_index + 1) % self.max_player] or not self.o_can_be_check_ckp_when_someone_play_card((i + player_index + 1) % self.max_player):
                continue
            player = self.players[(i + player_index + 1) % self.max_player]
            assert(isinstance(player, Player))
            if i == 0 and Op.Eat in self.rule.ops_for_card:
                etable, eat = self.rule.eatable(player, played_card)
            else:
                etable, eat = False, []

            if Op.Bump in self.rule.ops_for_card:
                bumpable, bump = self.rule.bumpable(player, played_card)
            if not self.card_pile.exhaushed() and Op.Gang1 in self.rule.ops_for_card:
                gang1able, gang = self.rule.gang1able(player, played_card)
            else:
                gang1able, gang = False, []
            ops = []
            if etable:
                ops.append(Op.Eat)
            if bumpable:
                ops.append(Op.Bump)
            if gang1able:
                ops.append(Op.Gang1)
            if len(ops) > 0:
                pop, pindex, cards = player.o_make_decision_to_cpg(self, played_card, ops, eat, bump, gang)
                if pop != Op.Empty:
                    op, index, op_cards, ckpp_index = pop, pindex, cards, (i + player_index + 1) % self.max_player
                    if pop in [Op.Bump, Op.Gang1]:
                        break
        if ckpp_index != NONE:
            ckpp = self.players[ckpp_index]
            assert(isinstance(ckpp, Player))
        if op == Op.Eat:
            ckpp.eat(op_cards, played_card)
            self.logger.trace("Player {} {} card {}".format(ckpp.name, Op.str(op), played_card.get_name()))
            self.logger.trace(ckpp.showable_info_by_self())
        elif op == Op.Bump:
            ckpp.bump(op_cards, played_card)
            self.logger.trace("Player {} {} card {}".format(ckpp.name, Op.str(op), played_card.get_name()))
            self.logger.trace(ckpp.showable_info_by_self())
        elif op == Op.Gang1:
            new_card = ckpp.gang1(op_cards, played_card, self.card_pile)
            self.logger.trace("Player {} {} card {}".format(ckpp.name, Op.str(op), played_card.get_name()))
            self.logger.trace("Draw card {}".format(new_card.get_name()))
            self.logger.trace(ckpp.showable_info_by_self())
        if op != Op.Empty:
            self.logger.trace("Player hand cards: {}".format(CardList.to_str(ckpp.hand_cards)))
        return self.o_on_cpg(op, ckpp_index, player_index, played_card)

    def o_on_cpg(self, op, player_index, source_index, card):
        if op == Op.Gang1:
            self.gang_in_turn = True
            self.next_player_is_gang = True
        
        if op != Op.Empty:
            self.next = player_index
            self.draw_before_round = False
            for i in range(self.max_player):
                self.once[i] = False

        return 0

    def _on_player_cpg_card(self, op, player_index):
        if op != Op.Empty:
            self.next = player_index
            self.draw_before_round = False
        self.next_player_is_gang = op == Op.Gang1

    def o_can_be_check_win_when_someone_play_card(self, player_index):
        return True

    def _winable_for_all_user(self, player_index, bloom, add_gang):
        winable_list, win_items, packages = [], [], []
        for i in range(1, self.max_player):
            if not self.o_can_be_check_win_when_someone_play_card((player_index + i) % self.max_player):
                continue
            index = (player_index + i) % self.max_player
            package = self._make_package(False, add_gang, bloom, index)
            win, items, detector = self.rule.o_winable(self.players[index], self, package, None)
            if win: 
                winable_list.append(index)
                win_items.append(items)
                packages.append(package)
        if len(winable_list) != 0:
            win_list = []
            for i in range(len(winable_list)):
                win = self.players[winable_list[i]].o_make_decision_to_win(self, win_items[i], packages[i])
                if win: 
                    win_list.append(winable_list[i])
                    end = self._process_winner_list(win_list, win_items, player_index)
                    return end
        return 0

    # 钩函数，一旦有人打出牌
    def _on_player_play_a_card(self, player_index, card, bloom, add_gang):
        self.active_card = card

        # 轮询是否能够获胜
        end = self._winable_for_all_user(player_index, bloom, add_gang)
        if end: return end

        # 轮询是否吃碰杠，加杠不允许此操作
        if not add_gang:
            end = self._ckpable_for_all_user(player_index, card)
            if end: return end

        # 更新hand for well
        self.hand_card_can_be_well[player_index] = HandCardCanBeWell(self.players[player_index].hand_cards, False)
        return 0 

    # 钩函数, 一旦有人抽牌就要判断是否结束游戏
    def _on_player_draw_a_card(self, player_index, card, bloom, hand_can_be_well=None):
        player = self.players[player_index]
        assert(isinstance(player, Player))
        self.logger.trace("Player {} draw a card {}".format(player.name, card.get_name()))
        self.logger.trace(player.showable_info_by_self())
        self.active_card = card
        self.last_drawing_card = card
        package = self._make_package(True, False, bloom, player_index)
        winable, win_types, detector = self.rule.o_winable(self.players[player_index], self, package, hand_can_be_well)
        if winable:
            result = player.o_make_decision_to_win(self, win_types, package)
            if result:
                end = self._process_winner_list([player_index], [win_types], player_index)
                if end: 
                    return end
        return False
    
    def o_flow_round(self):
        pass

    def o_end_round(self):
        self.logger.importent("Scores: {}".format(",".join(["{}".format(score) for score in self.scores])))
        self.logger.trace("---------------- End {}/{} ----------------".format(self.iter + 1, self.max_iter))
        self.o_flow_round()
        self.o_decision_next_main_player()
        self.o_clear_players_cards()                            # 清理玩家牌
        self.log_win_list()
    
    def log_win_list(self):
        for key in self.win_info.keys():
            for i in range(len(self.win_info[key])):
                info = self.win_info[key]
                name = self.players[info[i][0]].name
                self.logger.importent("{}: Player {} win {}".format(key, name, WinableType.to_str(info[i][1])))

    def o_end_game(self):
        self.logger.trace("================ End Game ================")

    # 创建winable的参数包
    def _make_package(self, by_self, grab_gang, bloom, player_index):
        package = default_MD_package()
        package["by_self"] = by_self
        package["stand"] = self.standing[player_index]
        package["self_wind"] = NONE
        package["ground_wind"] = NONE
        package["grab_gang"] = grab_gang
        package["bloom"] = bloom
        package["bottom"] = self.card_pile.exhaushed()
        package["once"] = self.once[player_index]
        package["w_stand"] = self.standing[player_index] == 2
        package["clear_river"] = len(self.players[player_index].card_river) == 0
        package["main_player"] = player_index == self.main_player
        return package

    # 寻找一名玩家
    def _find_player(self, player):
        for index in range(len(self.players)):
            if self.players[index].name == player.name:
                return index
        return -1

    # 移除一名玩家
    def remove_player(self, players):
        if isinstance(players, Player):
            index = self._find_player(players)
            if index == -1:
                self.logger.warning("Player {} not found".format(players.name))
            else:
                self.players.remove(self.players[index])
                self.logger.trace("Player {} is removed".format(players.name))
        else:
            for player in players:
                assert(isinstance(player, Player))
                index = self._find_player(player)
                if index == -1:
                    self.logger.warning("Player {} not found".format(player.name))
                else:
                    self.players.remove(self.players[index])
                    self.logger.trace("Player {} is removed".format(player.name))

    # 添加一名玩家
    def add_player(self, players):
        if isinstance(players, Player):
            if len(self.players) >= self.max_player:
                self.logger.warning("Too many player add to game")
            else:
                if self._find_player(players) != -1:
                    self.logger.trace("Player {} is already in game".format(players.name))
                else:
                    self.players.append(players)
                    self.logger.trace("Player {} add in game, now {}".format(players.name, len(self.players)))
        else:
            if len(self.players) + len(players) > self.max_player:
                self.logger.warning("Too many player add to game")
            else:
                for player in players:
                    assert(isinstance(player, Player))
                    if self._find_player(player) != -1:
                        self.logger.trace("Player {} is already in game".format(player.name))
                    else:
                        self.players.append(player)
                        self.logger.trace("Player {} add in game, now {}".format(player.name, len(self.players)))