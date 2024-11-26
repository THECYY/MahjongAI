from py_utils import *
from mglobal import *

class MajongFlowTrainer:

    def __init__(self, proj_name, app_name, cfg):
        print("Traning majongAI {}".format(cfg.ai_name))
        self.check_point = cfg.check_point
        self.cfg = cfg
        self.device = local_deivce()
        self.app_name = app_name
        self.proj_name = proj_name
        self.round = 0                  # 一个train round包含数个游戏round(默认是8)
        self.repeat = 0
        self.init_model()
        if self.check_point != -1:
            self.round = self.check_point * 1000    # 每1000train_round作为一个check_point
            self.load_model()
        self.model = self.model.to(self.device).train()
        self.parameters = list(self.model.parameters())
        self.init_optimizer()

    def init_model(self):
        self.model = None
        print("No model")

    # 创建优化器
    def init_optimizer(self, lr=5e-4, weight_decay=1e-6, eps=1e-8, betas=(0.95, 0.999)):
        betas = (0.9, 0.999) if betas == NONE else betas
        eps = 1e-8 if eps == NONE else eps
        weight_decay = 0.01 if NONE else weight_decay
        lr = 0.001 if lr == NONE else lr
        self.optim = torch.optim.AdamW(self.parameters, betas=betas, lr=lr, weight_decay=weight_decay, eps=eps)

    def get_model_path(self, *args):
        cfg = self.cfg
        memory = "memory_{}".format(int_to_fix_len_str(self.round // 1000, 8, "0"))
        return "ai_factory/{}/{}/{}.pth".format(cfg.ai_name, self.app_name, memory)

    def load_model(self, *args):
        print("Loading round {}".format(self.round // 1000))
        model_path = self.get_model_path(*args)
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, data):
        return self.model(data)
    
    def calculate_loss(self, data, output):
        return NONE

    def init_flow_dataset(self):
        self.flow_dataset = None
        self.flow_dataloader = None

    def train_a_size(self):
        self.model.train()
        pbar = tqdm(self.flow_dataloader)
        pbar.set_description("Memory {}_{}".format(self.round, self.repeat))
        for data in pbar:
            output = self.forward(data)
            loss, loss_dict = self.calculate_loss(data, output)
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters, max_norm=5.0)
            self.optim.step()
            pbar.set_postfix(loss_dict)
        pbar.close()
        self.repeat = (self.repeat + 1) % self.cfg.flow_repeat

    def train_a_flow(self):
        self.init_flow_dataset()
        for _ in range(self.cfg.flow_repeat):
            self.train_a_size()
        self.round += 1

    def save_model(self):
        model_path = self.get_model_path()
        make_sure_dir_of_file(model_path)
        torch.save(self.model.state_dict(), model_path)

    def eval(self):
        pass

    def train(self):
        while True:
            self.train_a_flow()
            if self.round % 1000 == 0:
                self.save_model()
                self.eval_ai()