from py_utils import *
from system.logger import Logger
from PIL import Image
from mahjong.card import *
from model.module.self_atten import *

def logger_test():
    logger = Logger()
    logger.trace("tracing")
    logger.warning("nonono")
    logger.error("sdfhsidufh")
    logger.print_to_console()
    logger.log_to_file("./log/log.txt")
    logger.trace("new")
    logger.print_to_console()
    logger.log_to_file("./log/log.txt")

def insert():
    array = [2, 4, 5, 6, 7, 4, 5, 2, 1, 9, 0]
    arr = [9, 4, 1]
    for item in array:
        binary_insert(arr, item)
        print(arr)
#  logger_test()

def dpi():
    PATHS = [
        "C:\\Users\\16036\\Desktop\\图片1.png",
        "C:\\Users\\16036\\Desktop\\图片2.png"
    ]
    SAVE_PATHS = [
        "C:\\Users\\16036\\Desktop\\xiaozhu1.png",
        "C:\\Users\\16036\\Desktop\\xiaozhu2.png"
    ]
    for i in range(2):
        image = Image.open(PATHS[i])
        image.save(SAVE_PATHS[i], dpi=(400, 400))

def class_func_map():
    class A:
        def __init__(self, name):
            self.name = name
        def print_name(self):
            print(self.name)
    
    map = {
        0: A.print_name
    }
    a = A("2")
    map[0](a)

def ref_func():

    class A:
        a = 1

    def inner_ref(a):
        a = A()

    b = None
    inner_ref(b)
    print(b)


def indices():
    print([1, 2, 3, 4][:0])

def deep():
    cards = [Card(MC.I1, False), Card(MC.I2, False)]
    card = cards[0]
    cs = copy.deepcopy(cards)
    cs.remove(card)
    print(cs)


def lenb0():
    print([] > 0)


def card_remove():
    cards = CardList.from_mc([
        MC.I1, MC.I5, MC.I5, MC.I5, MC.O3, MC.O6
    ])

    r_cards = CardList.from_mc([
        MC.I5, MC.I5
    ])

    CardList.remove_if_find(cards, r_cards)
    print(CardList.to_str(cards))

def atten():
    atten = SelfAttention(4, 256, 0.1)
    embedding = torch.randint(0, 10, [2, 6, 256]).float().transpose(0, 1)
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ]).bool()
    emb = atten(embedding, mask)

def embedding():
    emb = nn.Embedding(3, 2)
    a = torch.tensor([
        0, 1, 2
    ]).long()
    b = emb(a)
    print(b)

def nan():
    a = torch.tensor([2, 3, 4]) / 0
    a = torch.softmax(a, dim=-1)
    a = torch.argmax(a)
    print(a)
