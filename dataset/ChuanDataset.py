from chuan_data_generator import *
from torch.utils.data import Dataset, DataLoader

class ChuanDataset(Dataset):

    def __init__(self, cfg, dir):
        super().__init__()
        self.data_dir = dir
        if not os.path.exists(self.data_dir):
            generate_chuan_data(cfg)
        datas = load_chuan_data(dir)
        indices = [0 for _ in range(14)]
        for round_data in datas:
            for data in round_data:
                indices[int(data["index"])] += 1
        max_count = 0
        for index in indices:
            if max_count < index:
                max_count = index
        print("Index label: ", indices)
        self.device = local_deivce()
        print("Load game round {}".format(len(datas)))
        self.datas = []
        for round_data in datas:
            for data in round_data:
                self.datas.append(data)
        self.loss_scale = [max_count / index if index !=0 else 0 for index in indices]
        print("Total {} data".format(len(self.datas)))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        #hand_len = data["hand_mask"].sum()
        #indices = torch.randperm(hand_len)
        #if hand_len != 14:
        #    indices = torch.concatenate([indices, torch.arange(14 - hand_len) + hand_len], dim=-1)
        #hand = data["hand"][indices].to(self.device)
        #for i in range(hand_len):
        #    if int(indices[i]) == int(data["index"]):
        #        index = torch.tensor([i], dtype=torch.int32).to(self.device)
        #        break
        hand = data["hand"].to(self.device)
        index = torch.tensor(data["index"]).to(self.device)

        return hand, data["hand_mask"].to(self.device), data["table_types"].to(self.device), data["table_mcs"].to(self.device), \
            data["table_mask"].to(self.device), data["river"].to(self.device), data["river_mask"].to(self.device), data["other_table_types"].to(self.device), \
                data["other_table_mcs"].to(self.device), data["other_table_mask"].to(self.device), \
                    data["other_river"].to(self.device), data["other_river_mask"].to(self.device), data["won"].to(self.device), \
                        torch.tensor(data["score"]).to(self.device), index

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size, False)
