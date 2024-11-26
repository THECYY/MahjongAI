from chuan_data_generator import *
from torch.utils.data import Dataset, DataLoader

class ChuanDataset(Dataset):

    def __init__(self, cfg, dir):
        super().__init__()
        self.data_dir = dir
        if not os.path.exists(self.data_dir):
            generate_chuan_data(cfg)
        datas = load_chuan_data(dir)
        self.device = local_deivce()
        print("Load game round {}".format(len(datas)))
        self.datas = []
        for round_data in datas:
            for data in round_data:
                self.datas.append(data)
        print("Total {} data".format(len(self.datas)))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        return data["hand"].to(self.device), data["hand_mask"].to(self.device), data["table_types"].to(self.device), data["table_mcs"].to(self.device), \
            data["table_mask"].to(self.device), data["river"].to(self.device), data["river_mask"].to(self.device), data["other_table_types"].to(self.device), \
                data["other_table_mcs"].to(self.device), data["other_table_mask"].to(self.device), \
                    data["other_river"].to(self.device), data["other_river_mask"].to(self.device), data["won"].to(self.device), \
                        torch.tensor(data["score"]).to(self.device), torch.tensor(data["index"]).to(self.device)

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size, True)
