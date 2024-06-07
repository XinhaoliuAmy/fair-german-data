import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np


class ToTensor:
    def __call__(self, sample):
        # 确保 sample 是数值类型
        if isinstance(sample, (np.ndarray, pd.Series)):
            # 如果 sample 是整个数据点，则提取特征部分
            if isinstance(sample, pd.Series):
                sample = sample.values
            # 转换为数值类型，如果有非数值类型将会报错
            sample = sample.astype(np.float32)
            return torch.tensor(sample, dtype=torch.float32)
        else:
            raise TypeError("Sample should be a numpy array or pandas Series.")


class GermanDataset(Dataset):
    def __init__(self, num_class, sex_probability, root='data', train=True, valid=False, transform=ToTensor(), target_transform=None):
        super(GermanDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.num_class = num_class
        assert not (train and valid)
        self.prepare_german(is_train=train, is_valid=valid,
                            sex_probability=sex_probability)
        print(f'Preparing German Train={train} Set')
        if train:
            self.data_label_tuples = torch.load(
                os.path.join(self.root, 'train.pt'))
        elif not train and not valid:  # test
            self.data_label_tuples = torch.load(
                os.path.join(self.root, 'test.pt'))
        else:  # valid set
            self.data_label_tuples = torch.load(
                os.path.join(self.root, 'valid.pt'))

    def __getitem__(self, index):
        item = self.data_label_tuples[index]
        item_data = item[0]
        sensitive = item[1]
        target = item[2]

        # 将敏感特征添加到输入特征中
        if isinstance(item_data, torch.Tensor):
            item_data = torch.cat((item_data, torch.tensor(
                [sensitive], dtype=torch.float32)), dim=0)
        elif self.transform:
            item_data = self.transform(item_data)
            item_data = torch.cat((item_data, torch.tensor(
                [sensitive], dtype=torch.float32)), dim=0)

        if self.target_transform:
            target = self.target_transform(target)

        # 确保标签在正确的范围内
        target = int(target)
        assert 0 <= target < self.num_class, f"Label {target} out of bounds."

        return item_data, sensitive, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_german(self, is_train=True, is_valid=False, sex_probability=[0.5, 0.5]):
        # load dataset
        german_data = pd.read_csv(
            os.path.join(self.root, 'German_data.csv'))

        # Split features and target
        features = german_data.drop(columns=['Good/Bad'])
        target = german_data['Good/Bad']

        train_set = []
        valid_set = []
        labels = []
        fairs = []

        for idx, row in features.iterrows():
            if idx % 100 == 0:
                print(f'Converting {idx}/{len(target)}')
            feature_row = row.to_dict()
            feature_tensor = torch.tensor(
                list(feature_row.values()), dtype=torch.float)
            binary_label = 1 if target[idx] == 1 else 0

            if np.random.uniform() < 0.05:
                binary_label = binary_label ^ 1

            if binary_label == 1:  # good
                if np.random.uniform() < sex_probability[0]:
                    sex = 0
                else:
                    sex = 1
            else:  # bad
                if np.random.uniform() < sex_probability[1]:
                    sex = 0
                else:
                    sex = 1

            labels.append(binary_label)
            fairs.append(sex)

            if idx > 500:
                valid_set.append((feature_tensor, sex, binary_label))
            else:
                train_set.append((feature_tensor, sex, binary_label))

        fairs = np.array(fairs)
        labels = np.array(labels)
        female_pos = sum((fairs == 0) & (labels == 1))
        male_pos = sum((fairs == 1) & (labels == 1))
        female_neg = sum((fairs == 0) & (labels == 0))
        male_neg = sum((fairs == 1) & (labels == 0))

        print(f'\nProportion male: {fairs.sum() / len(fairs)}')
        print(f'Proportion class 1 {labels.sum() / len(labels)}')
        print(
            f'female_pos {female_pos / len(fairs)} \nfemale_neg {female_neg / len(fairs)} \nmale_pos {male_pos / len(fairs)} \nmale_neg {male_neg / len(fairs)}\n')

        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if is_train:
            torch.save(train_set, os.path.join(self.root, 'train.pt'))
            torch.save(valid_set, os.path.join(self.root, 'valid.pt'))
        else:  # test
            # This is just a placeholder. Adjust as needed.
            test_set = train_set
            torch.save(test_set, os.path.join(self.root, 'test.pt'))

        if is_train:
            self.data_label_tuples = train_set
        else:
            self.data_label_tuples = valid_set


if __name__ == '__main__':
    train_set = GermanDataset(train=True, sex_probability=[
                              0.2875536480686695, 0.67], num_class=2)
    valid_set = GermanDataset(train=False, valid=True, sex_probability=[
        0.2875536480686695, 0.67], num_class=2)
    test_set = GermanDataset(train=False, valid=False,
                             sex_probability=[0.5, 0.5], num_class=2)
    train_set = torch.load('data/train.pt')
    print(len(train_set))
    # Sample data from train set
    sample_data, sensitive, label = train_set[0]
    print("Sample data:", sample_data)
    print("Sensitive feature:", sensitive)
    print("Label:", label)
