'''
===================================
    Norm utils
===================================
'''
import random
import copy
import argparse
def repeate_str(str, length):
    return "".join([str for _ in range(length)])

def expand_str(str, length, expand_char=" ", mode=0):  # 0 左填充 1 右填充 2 均衡填充
    l = len(str)
    if l >= length:
        return str
    else:
        if mode <= 1:
            expand = repeate_str(expand_char, length - l)
            if mode == 0:
                return expand + str
            else:
                return str + expand
        else:
            l_expand = repeate_str(expand_char, (length - l) // 2)
            r_expand = repeate_str(expand_char, length - (length - l) // 2)
            return "{}{}{}".format(l_expand, str, r_expand)
            
def int_to_fix_len_str(num, length, expand_char=" "):
    num_str = f"{num}"
    if len(num_str) > length:
        return num_str[-length:]
    elif len(num_str) == length:
        return num_str
    else:
        return expand_str(num_str, length, expand_char, 0)

def str_to_int(str):
    try:
        i = int(str)
    except:
        i = -1
    return i

def string_splits(s:str):
    return s.split(",")

def get_cfg():
    prase = argparse.ArgumentParser("AssemblyPredict")

    # dataset
    #prase.add_argument("--player_types", type=string_splits, default="chuan_heu,chuan_pp,chuan_sp,chuan_ai")
    #prase.add_argument("--player_names", type=string_splits, default="lyx,wtf,lm,cyy")
    prase.add_argument("--game_type", type=str, default="chuan", choices=["sts4", "chuan"])
    prase.add_argument("--only_win", type=int, default=1)

    # ai
    prase.add_argument("--ai_name", type=str, default="Miao_01")

    # model
    prase.add_argument("--h_dim", type=int, default=216)
    prase.add_argument("--transformer_layers", type=int, default=4)
    prase.add_argument("--t_dim", type=int, default=108)
    prase.add_argument("--r_dim", type=int, default=216)
    prase.add_argument("--c_dim", type=int, default=216)

    # train
    prase.add_argument("--check_point", type=int, default=-1)
    prase.add_argument("--flow_repeat", type=int, default=10) # 每一个流数据使用的次数
    prase.add_argument("--flow_size", type=int, default=8) # 每一个流数据进行的局数大小

    args = prase.parse_args()
    return args

'''
===================================
    File utils
===================================
'''
import os 
import shutil
import pickle as pkl
import json

def _make_sure_dir_patches(patches):
    checking_dir = ""
    for i in range(len(patches)):
        checking_dir = checking_dir + "/" + patches[i] if i != 0 else patches[i]
        if not os.path.exists(checking_dir):
            os.mkdir(checking_dir)

def make_sure_dir(dir: str):
    split_str = dir.split("/")
    _make_sure_dir_patches(split_str)
    
def make_sure_dir_of_file(path):
    split_str = path.split("/")
    _make_sure_dir_patches(split_str[:-1])

def exist_dir(dir):
    return os.path.exists(dir)

def write_json_data(data, path):
    json_data = json.dumps(data)
    with open(path, "w") as f:
        f.write(json_data)

def load_json_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_pkl_data(data, path):
    with open(path, "wb") as f:
        pkl.dump(data, f)
    
def load_pkl_data(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data

'''
===================================
    Time utils
===================================
'''
from datetime import datetime
def system_time_str(date_link="-", time_link=":", inner_link = " "):
    now = datetime.now()
    time = "{}{}{}{}{}{}{}{}{}{}{}".format(
        int_to_fix_len_str(now.year, 2, "0"),
        date_link,
        int_to_fix_len_str(now.month, 2, "0"),
        date_link,
        int_to_fix_len_str(now.day, 2, "0"),
        inner_link,
        int_to_fix_len_str(now.hour, 2, "0"),
        time_link,
        int_to_fix_len_str(now.minute, 2, "0"),
        time_link,
        int_to_fix_len_str(now.second, 2, "0")
    )
    return time


'''
===================================
    Torch utils
===================================
'''
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
def local_deivce():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_activate_fn(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "lrelu":
        return nn.LeakyReLU()
    raise Exception("No activate fuction")

def mlp(layer_num, activate, in_dim, out_dim, forward_dim, bias=False, norm=True):
    assert(layer_num >= 1)
    dims = [forward_dim for _ in range(layer_num - 1)]
    dims.insert(0, in_dim)
    dims.append(out_dim)
    layers = nn.Sequential()
    for i in range(layer_num - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias))
        layers.append(get_activate_fn(activate))
        if norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
    layers.append(nn.Linear(dims[layer_num - 1], dims[layer_num], bias))
    return layers
    
def padding_to_len(data, len):
    now_len = data.shape[-1]
    padding_mask = torch.ones_like(data, dtype=torch.int, device=data.device)
    if now_len == len:
        return data, padding_mask == 1
    shape = data.shape[:-1] + (len - now_len, )
    new_data = torch.zeros(shape, dtype=data.dtype, device=data.device)
    return torch.concatenate([data, new_data], dim=-1), torch.concatenate([padding_mask, new_data], dim=-1) == 1

def torch_cards_to_uniform_mc_list(cards, uniform):
    mcs = torch.tensor([card.mc for card in cards]).long()
    mcs, padding_mask = padding_to_len(mcs, uniform)
    return mcs, padding_mask

def torch_table_cards_to_uniform_list(table_cards, uniform):
    if len(table_cards) == 0:
        return torch.zeros(uniform, dtype=torch.int32), torch.zeros([uniform, 3], dtype=torch.int32), torch.zeros(uniform, dtype=torch.int32) == 1
    types = torch.tensor([group.op for group in table_cards], dtype=torch.int32)
    types_tensor, padding_mask = padding_to_len(types, uniform)
    mcs = []
    for group in table_cards:  # n, 3
        mcs.append([group.cards[0].mc, group.cards[1].mc, group.cards[2].mc])
    for _ in range(uniform - len(table_cards)):
        mcs.append([0, 0, 0])
    return types_tensor, torch.tensor(mcs, dtype=torch.int32), padding_mask
        

'''
===================================
    Alg utils
===================================
'''
def binary_insert(array, item, key=lambda x: x):
    array_len = len(array)
    if array_len == 0:
        array.append(item)
        return
    left, right = 0, array_len - 1
    to_large = key(array[right]) >= key(array[left])
    while True:
        middle = (left + right) // 2
        large = key(item) >= key(array[middle])
        if to_large ^ large:
            if left == right:
                array.insert(left, item)
                break
            right = middle
        else:
            if left == middle:
                large_right = key(item) > key(array[right])
                if not large_right ^ to_large:
                    array.insert(right + 1, item)
                else:
                    array.insert(left + 1, item)
                break
            left = middle
    
        
'''
===================================
    Game utils
===================================
'''
def game_shotcum_to_play_card_data(game):
    current = game.current

    # 获取当前玩家的所有信息
    hand_cards = game.players[current].hand_cards
    hand_cards_tensor, hand_cards_padding_mask = torch_cards_to_uniform_mc_list(hand_cards, 14)
    card_river_tensors = []
    card_river_padding_masks = []
    table_card_type_tensors = []
    table_card_mc_tensors = []
    table_card_padding_masks = []

    won = []
    for i in range(game.max_player):
        player_index = (current + i) % game.max_player
        card_river = game.players[player_index].card_river
        card_river_tensor, card_river_padding_mask = torch_cards_to_uniform_mc_list(card_river, 27)
        card_river_tensors.append(card_river_tensor)
        card_river_padding_masks.append(card_river_padding_mask)

        table_card = game.players[player_index].table_cards
        types, mcs, padding_mask = torch_table_cards_to_uniform_list(table_card, 4)
        table_card_type_tensors.append(types)
        table_card_mc_tensors.append(mcs)
        table_card_padding_masks.append(padding_mask)
        won.append(game.won[player_index])
    
    return {
        "hand": hand_cards_tensor,
        "hand_mask": hand_cards_padding_mask,

        "table_types": table_card_type_tensors[0],
        "table_mcs": table_card_mc_tensors[0],
        "table_mask": table_card_padding_masks[0],

        "other_table_types": torch.stack(table_card_type_tensors[1:], dim=0),
        "other_table_mcs": torch.stack(table_card_mc_tensors[1:], dim=0),
        "other_table_mask": torch.stack(table_card_padding_masks[1:], dim=0),

        "river": card_river_tensors[0],
        "river_mask": card_river_padding_masks[0],
        "other_river": torch.stack(card_river_tensors[1:], dim=0),
        "other_river_mask": torch.stack(card_river_padding_masks[1:], dim=0),

        "won": torch.tensor(won[1:], dtype=torch.int32)
    }
