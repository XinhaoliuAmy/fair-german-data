import torch
import os
import copy
import numpy as np
import pandas as pd
import shutil
from scipy import stats
import collections
import torch.nn as nn
import torch.optim as optim
import torchvision
# import torchvision.transforms as transforms
from collections import OrderedDict
import pynvml
import types
# from prep_data import ci_mnist
# from prep_data import prep_celeba
# from prep_data import svhn
from prep_data import german_data

import model


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=lambda x: x, ind=False):
        self.data = x
        self.labels = y
        self.transform = transform
        if ind:
            self.indices = np.arange(len(x))
        else:
            self.indices = None

    def __getitem__(self, index):
        x = self.transform(
            self.data[index]) if self.transform else self.data[index]
        y = self.labels[index]
        if self.indices is not None:
            return x, y, self.indices[index]
        else:
            return x, y

    def __len__(self):
        return self.data.shape[0]


def get_parameters(net, numpy=False, squeeze=True, trainable_only=True):
    trainable = []
    non_trainable = []
    trainable_name = [name for (name, param) in net.named_parameters()]
    state = net.state_dict()
    for i, name in enumerate(state.keys()):
        if name in trainable_name:
            trainable.append(state[name])
        else:
            non_trainable.append(state[name])

    if squeeze:
        trainable = torch.cat([i.reshape([-1]) for i in trainable])
        # print(non_trainable)
        if len(non_trainable) > 0:
            non_trainable = torch.cat([i.reshape([-1]) for i in non_trainable])
        if numpy:
            trainable = trainable.cpu().numpy()
            if len(non_trainable) > 0:
                non_trainable = non_trainable.cpu().numpy()

    if trainable_only:
        parameter = trainable
    else:
        parameter = trainable + non_trainable

    return parameter


def set_parameters(net, parameters, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                   verbose=False):
    net.load_state_dict(to_state_dict(net, parameters, device, verbose))
    return net


def to_state_dict(net, parameters, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                  verbose=False):
    state_dict = OrderedDict()
    trainable_name = [name for (name, param) in net.named_parameters()]
    if len(trainable_name) < len(parameters):
        if verbose:
            print("Setting trainable and non-trainable parameters")
        i, j = 0, 0
        for name in net.state_dict().keys():
            if name in trainable_name:
                if isinstance(parameters[i], torch.Tensor):
                    state_dict[name] = parameters[i].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[i]).to(device)
                i += 1
            else:
                if isinstance(parameters[len(trainable_name) + j], torch.Tensor):
                    state_dict[name] = parameters[len(
                        trainable_name) + j].to(device)
                else:
                    state_dict[name] = torch.Tensor(
                        parameters[len(trainable_name) + j]).to(device)
                j += 1
    else:
        if verbose:
            print("Setting trainable parameters only")
        i = 0
        for name in net.state_dict().keys():
            if name in trainable_name:
                if isinstance(parameters[i], torch.Tensor):
                    state_dict[name] = parameters[i].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[i]).to(device)
                i += 1
            else:
                state_dict[name] = net.state_dict()[name]
    return state_dict


def record_to_csv(data, file, headers=None):
    data = [str(x) for x in data]
    if os.path.isfile(file):
        # append data
        with open(file, 'a') as fo:
            fo.write(','.join(data)+'\n')
    else:
        # create file, write headers, write data
        assert len(data) == len(headers)
        with open(file, 'a+') as fo:
            fo.write(','.join(headers)+'\n'+','.join(data)+'\n')


def load_dataset(dataset, train, sex_probability, valid=False, **kwargs):
    if dataset == 'German_data':
        transform = german_data.ToTensor()
        if train:
            dataset = german_data.GermanDataset(num_class=2,
                                                train=True, valid=False, sex_probability=sex_probability, transform=transform)
        elif not train and not valid:
            dataset = german_data.GermanDataset(num_class=2,  # test set
                                                train=False, valid=False, sex_probability=[0.5, 0.5], transform=transform)
        else:  # valid
            dataset = german_data.GermanDataset(num_class=2,
                                                train=False, valid=True, sex_probability=sex_probability, transform=transform)
        # data = german_data.GermanDataset(
        #     train=train, valid=valid, transform=transform)
        return dataset
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")


def num_parameters(net):
    return sum(p.numel() for p in net.parameters())


def find_last_chekpoint(dir_name):
    list_checkpoints = [0]
    for d in os.listdir(dir_name):
        if d.startswith("model"):
            try:
                ckpt = int(d.split('.')[0][6:])
            except:
                raise ValueError(
                    'Unexpected error happened at loading previous checkpoints')
            list_checkpoints.append(ckpt)
    return max(list_checkpoints)

# self.dataset, self.cla_net, self.cla_lr, self.c_num_batch,


def get_optimizer(dataset, cla_net, cla_lr, num_batch, dec_lr=None, privacy_engine=None, gamma=0.1, optimizer="sgd", weight_decay=None):
    print(dataset)
    if dataset == 'German_data':
        if optimizer == "ADAM":
            optimizer = optim.Adam(cla_net.parameters(),
                                   lr=cla_lr, weight_decay=1e-5)
            scheduler = None
        elif optimizer == "sgd":
            optimizer = optim.SGD(
                cla_net.parameters(), lr=cla_lr, momentum=0.9, weight_decay=weight_decay or 5e-4)
            if dec_lr is None:
                dec_lr = [30, 60, 90]  # 默认值，如果未提供
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=dec_lr, gamma=gamma)
        else:
            print(dataset)
            raise ValueError(
                f"Optimizer {optimizer} not supported for German dataset")
    else:
        raise ValueError(f"Dataset {dataset} not supported in this function")

    if privacy_engine is not None:
        privacy_engine.attach(optimizer)

    return optimizer, scheduler


def get_initial_model(model, save_path=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    if isinstance(model, str):
        try:
            architecture = eval(f"model.{model}")
        except:
            architecture = eval(f"torchvision.models.{model}")
        net = architecture().to(device)
    else:
        net = model().to(device)

    if save_path is not None:
        state = {'net': net.state_dict()}
        torch.save(state, os.path.join(save_path, f"initial_model.pt"))

    return net


def unsqueeze(architecture, parameter):
    unsqueezed = []
    net = architecture()
    reference = get_parameters(net, squeeze=False)
    for layer in reference:
        layer_shape = layer.shape
        layer_size = layer.reshape(-1).shape[0]
        unsqueezed.append(parameter[:layer_size].reshape(layer_shape))
        parameter = parameter[layer_size:]
    return unsqueezed


def add_states(state1, state2, a, b):
    return [a * i + b * j for i, j in zip(state1, state2)]


def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def get_model(model, architecture, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    state = torch.load(model)
    net = architecture()
    net.load_state_dict(state['net'])
    net.to(device)
    return net


def unnormalize(data, dataset=None, mean=None, std=None, rgb_last=False):
    if mean is None or std is None:
        normalize = [tm for tm in dataset.transform.transforms if isinstance(
            tm, transforms.transforms.Normalize)][0]
        mean, std = normalize.mean, normalize.std
    mean, std = mean_std_to_array(mean, std, rgb_last=rgb_last)
    if isinstance(data, torch.Tensor):
        mean, std = torch.from_numpy(
            mean).float(), torch.from_numpy(std).float()
    return data * std + mean


def renormalize(data, dataset=None, mean=None, std=None, rgb_last=False):
    if mean is None or std is None:
        normalize = [tm for tm in dataset.transform.transforms if isinstance(
            tm, transforms.transforms.Normalize)][0]
        mean, std = normalize.mean, normalize.std
    mean, std = mean_std_to_array(mean, std, rgb_last=rgb_last)
    if isinstance(data, torch.Tensor):
        mean, std = torch.from_numpy(
            mean).float(), torch.from_numpy(std).float()
    return (data - mean) / std


def get_save_dir(save_name, save_dir=None, save_dir_lab=None):
    # Check if save_dir exists
    if save_dir and os.path.exists(save_dir):
        print(save_dir)
        return os.path.join(save_dir, save_name)
    # Check if save_dir_lab exists
    elif save_dir_lab and os.path.exists(save_dir_lab):
        return os.path.join(save_dir_lab, save_name)
    # Default directory
    else:
        return os.path.join("models", save_name)


def get_last_ckpt(save_dir, keyword):
    saved_points = [int(model_path[len(keyword):]) for model_path in os.listdir(save_dir)
                    if keyword in model_path]
    return max(saved_points) if len(saved_points) > 0 else -1


def get_last_gen(save_dir, keyword):
    saved_gens = [int(path.split('_')[1]) for path in os.listdir(save_dir)
                  if keyword in path]
    return max(saved_gens) if len(saved_gens) > 0 else -1


def get_last_seed(save_dir, keyword):
    has_key = []
    for model_path in os.listdir(save_dir):
        if keyword in model_path:
            try:
                has_key.append(int(model_path[len(keyword):]))
            except:
                pass
    return max(has_key) if len(has_key) > 0 else -1


def random_pos(downscale=2):
    x = np.random.normal(0, 0.5)
    while x > 1 or x < - 1:
        x = np.random.normal(0, 0.5)
    if x < 0:
        x = x / downscale / 2 + 1
    else:
        x = x / downscale / 2
    return x
