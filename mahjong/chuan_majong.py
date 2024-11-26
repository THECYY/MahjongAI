from .game import *

class ChuanRule(Rule):
    ops_for_card = [Op.Bump, Op.Gang1]
    ops_in_turn = [Op.Play, Op.Gang4]

    activate_winable_types = [
        WinableType.Hand_Well,  
        WinableType.Grab_Gang,
        WinableType.Bloom,
        WinableType.Torch_Fish,
        WinableType.Torch_Moon,  

        WinableType.All_Bump,
        
        WinableType.Seven_Pairs,
        WinableType.Purity_Color,
        WinableType.Gold_Hook,
        WinableType.Purity_Unitary_Nine,

        WinableType.Dragon_Seven_Pair,
        WinableType.Four_Gang,
        WinableType.Major_Bump,
        WinableType.Sky,
        WinableType.Land,
    ]

    times_map = {
        WinableType.Hand_Well: 0,    

        WinableType.All_Bump: 1,
        WinableType.Grab_Gang: 1,
        WinableType.Bloom: 1,
        WinableType.Torch_Fish: 1,
        WinableType.Torch_Moon: 1, 
        
        WinableType.Seven_Pairs: 2,
        WinableType.Purity_Color: 2,
        WinableType.Gold_Hook: 2,
        WinableType.Purity_Unitary_Nine: 2,

        WinableType.Dragon_Seven_Pair: 4,
        WinableType.Four_Gang: 5,
        WinableType.Major_Bump: 4,
        WinableType.Sky: 5,
        WinableType.Land: 5,
    }

    restrain = {
        WinableType.Major_Bump: [WinableType.All_Bump],
        WinableType.Four_Gang: [WinableType.Gold_Hook, WinableType.All_Bump],
        WinableType.Dragon_Seven_Pair: [WinableType.Seven_Pairs],
        WinableType.Gold_Hook: [WinableType.All_Bump]
    }

    def has_lack(self, game: MGame, player: Player):
        index = game._find_player(player)
        lack = game.lack[index]
        for card in player.hand_cards:
            if MCType.is_type(card.mc, lack):
                return True
        for group in player.table_cards:
            for card in group.cards:
                if MCType.is_type(card.mc, lack):
                    return True
        return False

    def o_winable(self, player, game, package, hand_can_be_well=None):
        
        # 保证去掉任意一张牌都听牌
        player_index = game._find_player(player)
        args = {
            "lack": game.lack[player_index],
            "table_cards": game.players[player_index].table_cards
        }
        if not self.acceptable(player, game, package, hand_can_be_well, args):
            return False, [], None

        # 检测牌型
        results, detector = self.get_winable_map(player.hand_cards, player.table_cards, game.active_card, package, None, hand_can_be_well)
        return len(results) != 0, results, detector

    def o_calculate_score(self, game, winner_list, items, source_player_index):
        won_list = game.won
        sums = [0 for i in range(game.max_player)]
        for i in range(len(winner_list)):

            # 计算倍数
            types = items[i]
            table, hand = game.players[winner_list[i]].get_gang_num()
            score = self.get_score(game, types, table + hand)
            by_self = winner_list[i] == source_player_index
            if not by_self:     # 不自摸
                sums[source_player_index] -= score
                sums[winner_list[i]] += score
            else:
                sum = 0
                for k in range(len(won_list)):
                    if not won_list[k] and k != winner_list[i]:
                        sums[k] -= 2 * score
                        sum +=1
                sums[winner_list[i]] += sum * 2 * score
        return sums

    def get_score(self, game, items, gang_num):
        exps = gang_num
        for type in items:
            exps += self.times_map[type]
        score = pow(2, exps) * game.basic_score
        return score

    def o_judge_round_is_over(self, game: MGame):
        winners = game.win_info.get(game.iter)
        count = 0 if winners is None else len(winners)
        if game.max_player - count <= 1:
            return 1
        else:
            return 2

    def listen(self, hand_cards, hand_can_be_well: HandCardCanBeWell = None, args=None):
        for card in hand_cards:
            if MCType.is_type(card.mc, args["lack"]):
                return False, []
        for group in args["table_cards"]:
            for card in group.cards:
                if MCType.is_type(card.mc, args["lack"]):
                    return False, []
        return super().listen(hand_cards, hand_can_be_well)

class ChuanConfig(Config):
    max_player = 4
    max_iter = 99999999
    start_scores = [1000, 1000, 1000, 1000]
    basic_score = 5
    change_three = False

class ChuanMajong(MGame):

    def __init__(self, config: ChuanConfig, name="test"):
        super().__init__(config, name)

    def _can_current_player_start_his_time(self):
        can = super()._can_current_player_start_his_time() 
        if not can:
            return 0
        elif self.won[self.current]:
            return 2

    def o_init_round_basic_props(self):
        self.won = [False for _ in range(self.max_player)]
        self.basic_score = self.config.basic_score
        self.lack = [None for _ in range(self.max_player)]
        self.change_three = self.config.change_three
        self.gang_record = []       
    
    def _init_round(self):
        super()._init_round()
        if self.change_three:
            self._change_three_cards()
        self._all_player_choose_lack()
        
    # 额外记录目前获胜的所有玩家，他们不需要给分数
    def o_record_winner_list(self, winner_list, items, source):
        for i in range(len(winner_list)):
            if self.win_info.get(self.iter) is None:
                self.win_info[self.iter] = []
            self.win_info[self.iter].append([winner_list[i], items[i], source, copy.deepcopy(self.won)])

    # 定缺
    def _all_player_choose_lack(self):
        for i in range(self.max_player):
            player = self.players[i]
            assert(isinstance(player, Player))
            self.lack[i] = player.o_make_decision_to_lack(self)
            self.logger.trace("Player {} lack {}".format(self.players[i].name, MCType.str(self.lack[i])))

    # 赢的人无法响应后续打牌
    def o_can_be_check_ckp_when_someone_play_card(self, player_index):
        return not self.won[player_index]
    def o_can_be_check_win_when_someone_play_card(self, player_index):
        return not self.won[player_index]

    def _on_someone_is_win(self, winner_list, items, sources):
        scroes = super()._on_someone_is_win(winner_list, items, sources)
        for winner in winner_list:
            self.won[winner] = 1
        return scroes

    # 换三张
    def _change_three_cards(self):
        changing_cards = []
        for i in range(self.max_player):
            player = self.players[i]
            assert(isinstance(player, Player))
            cards = [[], [], []]
            for card in player.hand_cards:
                if card.mc in MC.I:
                    cards[0].append(card)
                elif card.mc in MC.O:
                    cards[1].append(card)
                else:
                    cards[2].append(card)
            r = []
            for k in range(3):
                if len(cards[k]) < 3:
                    r.append(cards[k])
            for remove in r:
                cards.remove(remove)
            
            index, i1, i2, i3 = player.o_make_decision_to_change_three(cards)
            changing_cards.append(
                [cards[index][i1], cards[index][i2], cards[index][i3]]
            )
            self.logger.trace("Player {} change {}".format(self.players[i].name, CardList.to_str(changing_cards[i])))

            for card in changing_cards[i]:
                player.hand_cards.remove(card)

        for i in range(self.max_player):
            player = self.players[i]
            assert(isinstance(player, Player))
            for k in range(3):
                player.add_a_card(changing_cards[(i + 1) % self.max_player][k])

    def o_on_cpg(self, op, player_index, source_index, card):
        scroes = [0 for _ in range(self.max_player)]
        end = False
        if op == Op.Gang1:
            sum = 0
            for i in range(self.max_player):
                if i != player_index and not self.won[i]:
                    scroes[i] = -self.basic_score
                    sum += scroes[i]
            scroes[source_index] = -self.basic_score * 2
            sum += -self.basic_score
            scroes[player_index] = -sum
            self.logger.importent("Player {} gang1".format(self.players[player_index].name))
            end = self.add_scores(scroes)
            self.gang_record.append([player_index, source_index, Op.Gang1, [not item for item in self.won]])

        end = end or super().o_on_cpg(op, player_index, source_index, card)
        return end

    # 谁赢谁坐庄
    def o_decision_next_main_player(self):
        winner_list = self.win_info.get(self.iter)
        self.next_main_player = self.main_player if winner_list is None else winner_list[0][0]
        self.iter += 1
    
    def o_init_card_pile(self):
        self.card_pile = MCardPile(0)
        self.card_pile.init_by_types([MCType.I, MCType.O, MCType.W])
        self.card_pile.shuffle()

    def o_init_rule(self):
        self.rule = ChuanRule()

    def _on_gang4(self, player_index, gang):
        scroes = [0 for _ in range(self.max_player)]
        scale = 1 if len(gang) == 1 else 2
        sum = 0
        for i in range(self.max_player):
            if i != player_index and not self.won[i]:
                scroes[i] = -self.basic_score * scale if not self.won[i] else 0
                sum += scroes[i]
        scroes[player_index] = -sum
        self.logger.importent("Player {} gang{}".format(self.players[player_index].name, 1 if len(gang) == 1 else 4))
        end = self.add_scores(scroes)
        self.gang_record.append([player_index, player_index, Op.Gang1 if len(gang) == 1 else Op.Gang4, [not item for item in self.won]])
        return end or super()._on_gang4(player_index, gang)

    def get_scroes(self, player_index, listening):
        lowest = 99999999
        its = None
        lsmc = None
        source_player = self.players[player_index]
        hand_can_be_well = HandCardCanBeWell(source_player.hand_cards, False)
        for mc in listening:
            card = Card(mc, False)
            player = copy.deepcopy(source_player)
            assert(isinstance(player, Player))
            player.add_a_card(card)
            self.players[player_index] = player
            self.active_card = card
            package = self._make_package(False, False, False, player_index)
            winable, items, detector = self.rule.o_winable(player, self, package, hand_can_be_well)
            for item in [WinableType.Torch_Fish, WinableType.Torch_Moon, WinableType.Bloom]:
                if items.keys().__contains__(item):
                    items.pop(item)
            assert(winable == True)
            table, hand = player.get_gang_num()
            score = self.rule.get_score(self, items, table + hand)
            if lowest > score:
                lowest = score
                its = items 
                lsmc = mc
        self.players[player_index] = source_player
        return lowest, its, lsmc

    def o_end_round(self):
        nls, ls, scroes = [], [], []
        for i in range(self.max_player):
            if self.won[i]:
                continue
            player = self.players[i]
            assert(isinstance(player, Player))
            args = {
                "lack": self.lack[i],
                "table_cards": self.players[i].table_cards
            }
            listenable, listening = self.rule.listen(player.hand_cards, None, args)   # 仅仅是基础的listen, 定缺没有
            if listenable:
                ls.append(i)
                score, items, lsmc = self.get_scroes(i, listening)
                scroes.append(score)
                self.logger.importent("Player {} listening {}, score {}, win types {}".format(player.name, MC.str(lsmc), score, WinableType.to_str(items)))
            else:
                nls.append(i)
                self.logger.importent("Player {} not listen".format(player.name))
        
        if len(nls) != 0:
            ss = [0 for _ in range(self.max_player)]
            for i in nls:
                for index in range(len(ls)):
                    ss[i] -= scroes[index]
                    ss[ls[index]] += scroes[index]
            self.add_scores(ss)
            self.not_listen_score = ss

            self.logger.importent("Back gang score")
            ss = [0 for _ in range(self.max_player)]
            for gang_info in self.gang_record:
                if gang_info[0] not in nls:
                    continue
                if gang_info[2] == Op.Gang4:
                    sum = 0
                    for index in ls:
                        ss[index] += self.basic_score * 2
                        sum += self.basic_score * 2
                    ss[gang_info[0]] -= sum
                else:
                    sum = 0
                    for index in ls:
                        if index == gang_info[1]:
                            ss[index] += self.basic_score * 2
                            sum += self.basic_score * 2
                        else:
                            ss[index] += self.basic_score
                            sum += self.basic_score
                    ss[gang_info[0]] -= sum
            self.add_scores(ss)
        self.not_listen_score = [0 for _ in range(self.max_player)]
        
        super().o_end_round()