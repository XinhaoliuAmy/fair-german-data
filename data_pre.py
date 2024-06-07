import pandas as pd
import torch
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        row = self.data.iloc[index]
        sensitive = row['sex']
        target = row['good/bad']
        return sensitive, target

    def __len__(self):
        return len(self.data)


# 1. 加载数据
csv_file_path = './data/German_data/German_data.csv'
data = pd.read_csv(csv_file_path)

# 2. 数据准备
# 如果需要对数据进行处理，请在这里添加逻辑

# 3. 数据转换
dataset = CustomDataset(data)


# 4. 保存数据
save_dir = './data/German_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch.save(dataset, os.path.join(save_dir, 'custom_dataset.pt'))
