# Debug counter -- 14 ):

import numpy as np
import os
import math
import random
import types
import torch
import piq
import copy
import shutil
import time
import inspect
import torchvision.transforms as transforms
import torchvision.utils as torch_utils
import torch.nn.functional as F
from collections import OrderedDict
import warnings
import utils
import model
from models import *
import generator_utils as gen_utils
import matplotlib.pyplot as plt
import fairness
from collections import OrderedDict

torch.autograd.set_detect_anomaly(True)


class train_fn():
    def __init__(self, gen_lr=1e-3, gen_batch_size=256, cla_lr=.001, cla_batch_size=64,
                 dataset='German_data', generator=model.TabularGenerator, classifier=model.TabularClassifier,
                 exp_id=None, model_dir=None, save_freq=None, eval_freq=None, trainset=None,
                 save_name=None, num_class=2, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 seed=0, g_optimizer="ADAM", weight_decay=1e-4, g_epochs=30,
                 c_optimizer="sgd", c_epochs=30, variational_beta=1, pos_class_thresh=5,
                 overwrite=0, capacity=64, latent_dims=58, sample_data=1, sex_probability=[.5, .5], synthetic_perc=1,
                 use_reparation=False, rep_budget=0, gamma=.95, roll_ckpts=0, input_dim=58, hidden_dim=128, output_dim=58, output_dim_classifer=1, input_dim_classifer=57):  # here the good/bad should be done in the annoator
        inputs = inspect.signature(train_fn).parameters
        for item in inputs:
            setattr(self, item, eval(item))

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # setup save directories
        if save_name is None:
            save_name = f"MCBench/ckpt_{self.dataset}_{exp_id}"
            self.save_dir = utils.get_save_dir(save_name)

        if save_freq is not None:
            self.save_dir = utils.get_save_dir(save_name)

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"mkdir {self.save_dir}")
            else:
                if len(os.listdir(self.save_dir)) > 0:
                    print(
                        f"Checkpointing directory is not empty {self.save_dir}")
                    if overwrite:
                        shutil.rmtree(self.save_dir)
                        os.makedirs(self.save_dir)
                        print(f"overwrite {self.save_dir}")
                        assert len(os.listdir(self.save_dir)) == 0
        else:
            self.save_dir = None

        # NOTE uncomment below for saving plots for every seed too.
        if seed == 0:  # Only saving latent reps for seed 0
            self.fig_dir = f"./figs/{self.dataset}/{self.dataset}_{exp_id}"
            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir)
                print(f"makedirs {self.fig_dir}")
        self.result_dir = f"./results/{self.dataset}/{self.dataset}_{exp_id}"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            print(f"makedirs {self.result_dir}")

        # load datasets
        if trainset is None:
            self.trainset = utils.load_dataset(
                self.dataset, train=True, sex_probability=self.sex_probability)
            self.validset = utils.load_dataset(
                self.dataset, train=False, valid=True, sex_probability=self.sex_probability)
        else:
            self.trainset = trainset
        self.testset = utils.load_dataset(
            self.dataset, sex_probability=[0.5, 0.5], train=False)

        train_size = self.trainset.__len__()

        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=int(self.gen_batch_size),
                                                        shuffle=True, pin_memory=True, num_workers=0, timeout=0)
        self.valid_loader = torch.utils.data.DataLoader(self.validset, batch_size=int(self.cla_batch_size),
                                                        shuffle=True, pin_memory=True, num_workers=0, timeout=0)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=int(self.gen_batch_size),
                                                       shuffle=True, pin_memory=True, num_workers=0, timeout=0)

    def init_gen_opt(self, generator, dataset, gen_lr, g_optimizer, weight_decay):
        # 初始化模型，设置 GPU
        self.gen_net = generator(
            self.input_dim, self.hidden_dim, self.output_dim, good_bad_dim=1, sensitive_dim=1, feature_dim=56)
        if torch.cuda.device_count() > 1:
            self.gen_net = torch.nn.DataParallel(self.gen_net)
            print(f"number of GPUS：{self.gen_net.device_ids}")
        try:
            self.g_num_batch = self.trainset.__len__() / self.batch_size
        except:
            self.g_num_batch = None
        self.gen_net.to(self.device)
        # 设置调度器、优化器和损失函数
        self.gen_optimizer, self.gen_scheduler = utils.get_optimizer(dataset, self.gen_net, gen_lr, self.g_num_batch,
                                                                     optimizer=g_optimizer, weight_decay=weight_decay, gamma=self.gamma)
        # 根据数据集选择合适的损失函数
        if dataset == 'German_data':
            self.gen_criterion = gen_utils.custom_generator_loss
        else:  # 默认损失函数
            self.gen_criterion = gen_utils.default_loss_function

    def update(self):
        self.gen_optimizer.step()
        self.gen_optimizer.zero_grad()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def compute_loss(self, data, good_bad_dim, sensitive_dim):
        inputs = data[0].to(self.device).float()

        # 从生成器模型获取输出数据
        image_batch_recon = self.gen_net(inputs)

        # 计算损失
        loss = self.gen_criterion(
            image_batch_recon, inputs, good_bad_dim, sensitive_dim)

        return loss

    def train_step(self, data, good_bad_dim=1, sensitive_dim=1):
        loss = self.compute_loss(data, good_bad_dim, sensitive_dim)
        loss.backward()
        self.update()
        return loss.item()

    def c_update(self):
        self.cla_optimizer.step()
        self.cla_optimizer.zero_grad()
        if self.cla_scheduler is not None:
            self.cla_scheduler.step()

    def c_compute_loss(self, data, label_idx, **kwargs):
        # 'labels' are color for ano_fair
        inputs, labels = data[0].to(
            self.device), data[label_idx].to(self.device)
        outputs = self.cla_net(inputs)
        # print(outputs)
        # count = torch.sum(labels > 1).item() # check if outof bounds
        # print(count)
        outputs = outputs.squeeze(dim=1)
        labels = labels.float()
        loss = self.cla_criterion(outputs, labels)
        # print(loss)
        return loss

    def c_train_step(self, data, label_idx):
        loss = self.c_compute_loss(data, label_idx)
        loss.backward()
        self.c_update()
        return loss.item()

    def save(self, gen_number=None, epoch=None, save_path=None, overwrite=False, is_generator=True, is_label_annotator=False, is_fair_annotator=False):
        # NOTE, comment if you want to save intermediate checkpoints.
        # if is_generator and epoch != self.g_epochs:
        #     return
        # if not is_generator and epoch != self.c_epochs:
        #     return
        if epoch == 0:
            return
        prev_path = None
        assert epoch is not None or save_path is not None
        if is_generator and save_path == None:
            save_path = os.path.join(
                self.save_dir, f"gen_{gen_number}_epoch_{epoch}")
            prev_path = os.path.join(
                self.save_dir, f"gen_{gen_number-1}_epoch_{epoch}")
            net = self.gen_net
            optimizer = self.gen_optimizer
            scheduler = self.gen_scheduler
        elif not is_generator and save_path == None:  # saving classifier
            if not is_label_annotator and not is_fair_annotator:
                save_path = os.path.join(
                    self.save_dir, f"cla_{gen_number}_epoch_{epoch}")
                prev_path = os.path.join(
                    self.save_dir, f"cla_{gen_number-1}_epoch_{epoch}")
            elif is_label_annotator:
                save_path = os.path.join(
                    self.save_dir, f"ano_label_epoch_{epoch}")
            else:  # is_fair_annotator:
                save_path = os.path.join(
                    self.save_dir, f"ano_fair_epoch_{epoch}")
            net = self.cla_net
            optimizer = self.cla_optimizer
            scheduler = self.cla_scheduler
        else:
            pass

        if "data_parallel" in str(type(net)):
            print("parallel save")
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        if os.path.exists(save_path) and overwrite:
            os.remove(save_path)

        # Especially for celeba, remove previous checkpoint to save space
        if self.roll_ckpts == 1 and prev_path != None:
            try:
                os.remove(prev_path)
                print(f'Removed {prev_path}')
            except:
                print(f"Would've removed {prev_path}")
        if not os.path.exists(save_path):
            state = {'net': net_state_dict,
                     'optimizer': optimizer.state_dict()}
            if scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            torch.save(state, save_path)

    def load(self, path, is_generator):
        states = torch.load(path)
        if is_generator:
            print(f'Loading generator from {path}')
            try:
                self.gen_net.load_state_dict(states['net'])
            except RuntimeError as e:
                print(f'Error loading generator state_dict directly: {e}')
                print('Attempting to load generator state_dict with "module." prefix...')
                new_dict = {}
                for (key, val) in states['net'].items():
                    newkey = "module." + \
                        key if not key.startswith("module.") else key
                    new_dict[newkey] = val
                try:
                    self.gen_net.load_state_dict(new_dict)
                except RuntimeError as e:
                    print(
                        f'Error loading generator state_dict with "module." prefix: {e}')
            self.gen_optimizer.load_state_dict(states['optimizer'])
            if self.gen_scheduler is not None:
                self.gen_scheduler.load_state_dict(states['scheduler'])
                print(
                    f"current learning rate: {self.gen_scheduler.get_last_lr()}")
        else:  # loading classifier
            print(f'Loading classifier from {path}')
            try:
                self.cla_net.load_state_dict(states['net'])
            except RuntimeError as e:
                print(f'Error loading classifier state_dict directly: {e}')
                print(
                    'Attempting to load classifier state_dict with "module." prefix...')
                new_dict = {}
                for (key, val) in states['net'].items():
                    newkey = "module." + \
                        key if not key.startswith("module.") else key
                    new_dict[newkey] = val
                try:
                    self.cla_net.load_state_dict(new_dict)
                except RuntimeError as e:
                    print(
                        f'Error loading classifier state_dict with "module." prefix: {e}')
            self.cla_optimizer.load_state_dict(states['optimizer'])
            if self.cla_scheduler is not None:
                self.cla_scheduler.load_state_dict(states['scheduler'])
                print(
                    f"current learning rate: {self.cla_scheduler.get_last_lr()}")

    def init_classifier(self):
        # init model, setup GPUs
        self.cla_net = self.classifier(
            hidden_dim=self.hidden_dim, output_dim_classifer=self.output_dim_classifer, input_dim_classifer=self.input_dim_classifer)
        if torch.cuda.device_count() > 1:
            self.cla_net = torch.nn.DataParallel(self.cla_net)
            print(f"using gpus: {self.cla_net.device_ids}")
        try:
            self.c_num_batch = self.trainset.__len__() / self.cla_batch_size
        except:
            self.c_num_batch = None
        self.cla_net.to(self.device)
        # setup sched, optimizer, and loss
        self.cla_optimizer, self.cla_scheduler = utils.get_optimizer(
            self.dataset, self.cla_net, self.cla_lr, self.c_num_batch, optimizer=self.c_optimizer)
        self.cla_criterion = torch.nn.BCELoss()

    def reparation_batch(self, sample_from, label_from, group_from, generation, epoch, batch_size):
        budget = batch_size + self.rep_budget
        # sample randomly
        with torch.no_grad():
            image_batch = self.sample_decoder(sample_from, batch_size=budget)
            lab_outputs = label_from(image_batch)
            _, labels = torch.max(lab_outputs, 1)
            labels = labels.cpu().numpy()
            grp_outputs = group_from(image_batch)
            _, groups = torch.max(grp_outputs, 1)
            groups = groups.cpu().numpy()
        # categorize by label and sensitive attribute
        c0_g0 = np.intersect1d(np.where(labels == 0), np.where(groups == 0))
        c0_g1 = np.intersect1d(np.where(labels == 0), np.where(groups == 1))
        c1_g0 = np.intersect1d(np.where(labels == 1), np.where(groups == 0))
        c1_g1 = np.intersect1d(np.where(labels == 1), np.where(groups == 1))
        cat_idxs = [c0_g0, c0_g1, c1_g0, c1_g1]
        sc = np.array([len(cat) for cat in cat_idxs])  # sampled counts
        # get number desired to get from each group
        # for complete balance (c0g0, c0g1, c1g0, c1g1)
        fair_ideal = np.array([.25] * 4)
        fair_counts = np.floor(fair_ideal * batch_size)
        idx = []
        # flooring might underestimate batch size
        to_resample = batch_size - np.sum(fair_counts)
        for i in range(len(fair_counts)):
            number = int(fair_counts[i])
            if number > len(cat_idxs[i]):
                to_resample += number - len(cat_idxs[i])
                idx += cat_idxs[i].tolist()
            else:
                idx += cat_idxs[i].tolist()[:number]
        images = image_batch[idx]
        labels = labels[idx]
        to_resample = int(to_resample)
        # If we didn't meet our criteria, sample randomly until we have enough to fill out the batch.
        # print(f"Need to resample {to_resample} samples :(")
        if to_resample > 0:
            with torch.no_grad():
                extra_imgs = self.sample_decoder(
                    sample_from, batch_size=to_resample)
                outputs = label_from(extra_imgs)
                _, extra_labels = torch.max(outputs, 1)
                extra_labels = extra_labels.cpu().numpy()
            images = torch.cat((images, extra_imgs), dim=0)
            labels = np.concatenate((labels, extra_labels), axis=0)
        labels = torch.tensor(labels)
        labels = labels.to(self.device)
        # shuffle batch
        idx = torch.randperm(images.shape[0])
        images = images[idx].view(images.size())
        labels = labels[idx].view(labels.size())
        image_batch = [images, labels]
        assert len(image_batch[0]) == batch_size

        # save some stats on the batch
        file = os.path.join(
            self.result_dir, f"ar_{self.use_reparation}_batches.csv")
        cnames = ['generation', 'epoch', 'resampled',
                  'c0g0', 'c0g1', 'c1g0', 'c1g1']
        data = [generation, epoch, to_resample, sc[0], sc[1], sc[2], sc[3]]
        utils.record_to_csv(data, file, headers=cnames)
        return image_batch

    def sample_decoder(self, sample_from, batch_size):
        latent = torch.randn(batch_size, self.latent_dims, device=self.device)
        try:
            try:  # if isinstance(gen_net, torch.nn.DataParallel):
                images = sample_from.decoder(latent)
            except:
                images = sample_from.decoder(latent)
        except:
            try:  # if isinstance(gen_net, torch.nn.DataParallel):
                images = sample_from.decoder(latent)
            except:  # else:
                images = sample_from.decoder(latent)
        return images

    def train_classifier(self, gen=None, og_rate=0, sample_from=None, label_from=None, group_from=None, is_ano_lab=False,
                         is_ano_fair=False, **kwargs):
        """ Can train classifiers of three varieties: 
                ano_label: annotates labels. Trained from og data 
                ano_fair: annotates sensitive attr. Trained from og data
                gen_cla: classifier trained from (generator, ano_label)
        Params:
            gen: (int) For sampling from generators. First generator (trained on OG data) is number 0. 
                        -1 means training without gen sampling
            og_rate: (float) Proportion of data from original distribution
            sample_from: generator net
            label_from: label annotator or (sequential case) previous classifier
            is_ano_lab/fair: signifies if training one of the annotators.
        """
        init_epoch = 0
        self.init_classifier()
        # resolve save/load keyword
        if is_ano_lab:
            assert not is_ano_fair
            keyword = "ano_label_epoch_"
            save_name = 'ano_label'
            print(f"Training label annotator")
            label_idx = 2
        elif is_ano_fair:
            keyword = "ano_fair_epoch_"
            save_name = 'ano_fair'
            print(f"Training sensitive attribute annotator")
            label_idx = 1
        else:  # cla_gen
            assert gen is not None and sample_from is not None and label_from is not None
            keyword = f"cla_{gen}_epoch_"
            save_name = 'cla'
            print(f"Training classifier for generation {gen}")
            label_idx = 1
        # check for existing classifier
        print(self.save_dir)
        if self.save_dir is not None:
            last_ckpt = utils.get_last_ckpt(self.save_dir, keyword)
            if last_ckpt == self.c_epochs:
                # will skip to eval
                print(f"Classifier exists, ckpt at {keyword}{last_ckpt}")
                init_epoch = last_ckpt
            if last_ckpt > 0:
                self.load(os.path.join(self.save_dir,
                          f"{keyword}{last_ckpt}"), is_generator=False)
                init_epoch = last_ckpt

        self.cla_net.train()
        if sample_from is not None:
            sample_from.eval()
        if label_from is not None:
            label_from.eval()
            # print(label_from != None)

        num_params = sum(p.numel()
                         for p in self.cla_net.parameters() if p.requires_grad)
        print('Number of parameters: %d' % num_params)
        train_loss_avg = []

        cur_epoch = init_epoch
        for epoch in range(init_epoch, self.c_epochs):
            train_loss_avg.append(0)
            num_batches = 0
            for idx, image_batch in enumerate(self.train_loader, 0):
                if not is_ano_fair and not is_ano_lab:  # cla_gen
                    if self.use_reparation in ['cla', 'both']:
                        print(self.rep_budget > 0)
                        assert (label_from != None) and (
                            sample_from != None) and (self.rep_budget > 0)
                        image_batch = self.reparation_batch(
                            sample_from, label_from, group_from, gen, epoch, self.cla_batch_size)
                    else:
                        with torch.no_grad():
                            image_batch = self.sample_decoder(
                                sample_from, batch_size=image_batch[0].shape[0])
                            outputs = label_from(image_batch)
                            max_logits, labels = torch.max(outputs, 1)
                            image_batch = [image_batch, labels]
                # else image_batch is for ano_lab/fair
                loss_val = self.c_train_step(image_batch, label_idx)
                train_loss_avg[-1] += loss_val
                num_batches += 1

            train_loss_avg[-1] /= num_batches

            if self.save_freq is not None and self.save_freq > 0 and epoch % self.save_freq == 0:
                self.save(gen_number=gen, epoch=epoch, is_generator=False,
                          is_fair_annotator=is_ano_fair, is_label_annotator=is_ano_lab)

            if self.eval_freq is not None and epoch % self.eval_freq == 0 and self.eval_freq > 0:
                print('Classifier epoch [%d / %d] average train loss: %f' %
                      (epoch+1, self.c_epochs, train_loss_avg[-1]))
                self.validate(save_name, gen, cur_epoch,
                              is_ano_fair=is_ano_fair, use_valid=True)
            cur_epoch += 1
        print(f'Done training at epoch {cur_epoch}!')

        # save final model
        if self.save_freq > 0:
            self.save(gen_number=gen, epoch=self.c_epochs, is_generator=False,
                      is_fair_annotator=is_ano_fair, is_label_annotator=is_ano_lab)
        # eval final model and plot reconstructions
        if len(train_loss_avg) > 0:
            print('Final epoch average reconstruction error: %f' %
                  (train_loss_avg[-1]))
        self.validate(save_name, gen, cur_epoch,
                      is_ano_fair=is_ano_fair, use_valid=True)
        self.validate(save_name, gen, cur_epoch,
                      is_ano_fair=is_ano_fair, use_valid=False)
        return self.cla_net

    def train_generator(self, gen_number, sample_generator=False, sample_from=None, label_from=None, group_from=None, good_bad_dim=1, sensitive_dim=1, feature_dim=56, **kwargs):
        if gen_number > 0:
            num_to_sample = int(self.gen_batch_size * self.synthetic_perc)
            num_natural = self.gen_batch_size - num_to_sample
        else:
            num_to_sample = 0
            num_natural = self.gen_batch_size
        print(f"\nsampling {num_to_sample}, original {num_natural}\n")
        # check for existing generator
        init_epoch = 0
        # initialize fresh model every time this function is called.
        self.init_gen_opt(self.generator, self.dataset,
                          self.gen_lr, self.g_optimizer, self.weight_decay)

        if self.save_dir is not None:
            keyword = f"gen_{gen_number}_epoch_"
            last_ckpt = utils.get_last_ckpt(self.save_dir, keyword)
            if last_ckpt == self.g_epochs:
                # will skip train loop but do eval
                print(
                    f"Generation already trained: ckpt at {keyword}{last_ckpt}")
                init_epoch = last_ckpt
            if last_ckpt > 0:
                self.load(os.path.join(self.save_dir,
                          f"{keyword}{last_ckpt}"), is_generator=True)
                init_epoch = last_ckpt

        self.gen_net.train()
        if sample_from is not None:
            sample_from.eval()

        num_params = sum(p.numel()
                         for p in self.gen_net.parameters() if p.requires_grad)
        print('Number of parameters: %d' % num_params)
        train_loss_avg = []

        print(f'Training gen {gen_number}...')
        for epoch in range(init_epoch, self.g_epochs):
            train_loss_avg.append(0)
            num_batches = 0

            for idx, image_batch in enumerate(self.train_loader, 0):
                if sample_generator and sample_from is not None:
                    if self.use_reparation in ['gen', 'both']:
                        assert (label_from != None) and (
                            sample_from != None) and (self.rep_budget > 0)
                        image_batch = self.reparation_batch(
                            sample_from, label_from, group_from, gen_number, epoch, self.gen_batch_size)
                    else:
                        with torch.no_grad():
                            syn_image_batch = self.sample_decoder(
                                sample_from, batch_size=num_to_sample)
                            dummy_label = torch.zeros(len(image_batch))
                            if num_natural > 0:
                                nat_in = image_batch[0].to(self.device)[
                                    :num_natural]
                                inputs = torch.cat(
                                    (syn_image_batch, nat_in), dim=0)
                                # shuffle natural and synthetic images together so there's no untoward effects.
                                idx = torch.randperm(inputs.shape[0])
                                inputs = inputs[idx].view(inputs.size())
                            else:
                                inputs = syn_image_batch
                            image_batch = [inputs, dummy_label]

                loss_val = self.train_step(
                    image_batch, good_bad_dim, sensitive_dim)
                train_loss_avg[-1] += loss_val
                num_batches += 1

            train_loss_avg[-1] /= num_batches

            if self.save_freq is not None and self.save_freq > 0 and epoch % self.save_freq == 0:
                self.save(gen_number=gen_number, epoch=epoch,
                          is_generator=True, save_path=None)
            if self.eval_freq is not None and epoch % self.eval_freq == 0 and self.eval_freq > 0:
                print('Epoch [%d / %d] average reconstruction error: %f' %
                      (epoch, self.g_epochs, train_loss_avg[-1]))
                headers = ['generation', 'epoch', 'gen_loss']
                data = [gen_number, epoch, train_loss_avg[-1]]
                file = os.path.join(self.result_dir, f"gen_loss.csv")
                utils.record_to_csv(data, file, headers=headers)
                if self.seed == 0:
                    self.visualize_outputs(gen_number, epoch)
        print('Done training!')
        # save final model
        if self.save_freq > 0:
            self.save(gen_number=gen_number,
                      epoch=self.g_epochs, is_generator=True)
        # eval final model and plot reconstructions
        if len(train_loss_avg) > 0:
            print('Final epoch average reconstruction error: %f' %
                  (train_loss_avg[-1]))
        if self.seed == 0:
            self.visualize_outputs(gen_number, self.g_epochs)

        return self.gen_net

    def predict(self, inputs):
        outputs = self.cla_net(inputs)
        if isinstance(outputs, tuple) and len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs.shape) > 2:
            outputs = outputs.squeeze()
        elif not isinstance(outputs, torch.Tensor):
            outputs = outputs.logits
        return outputs

    def validate(self, save_name, gen, epoch, is_ano_fair=False, use_valid=True):
        # results_dir = './results/dataset_exp_id' + save_name 'cla' 'ano_lab' or 'ano_fair'
        assert save_name in ['cla', 'ano_label', 'ano_fair']
        # for classifiers only
        self.cla_net.eval()
        correct = 0
        total = 0
        # for fair vs classifier task labels
        if is_ano_fair:
            label_idx = 1
        else:
            label_idx = 2
        # choose validation or test set
        if use_valid:
            dloader = self.valid_loader
        else:
            dloader = self.test_loader

        labels_list = []
        preds_list = []
        sensitive_list = []
        with torch.no_grad():
            for data in dloader:
                inputs, labels = data[0].to(
                    self.device), data[label_idx].to(self.device)
                outputs = self.predict(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                preds_list += predicted.tolist()
                labels_list += data[2].tolist()
                sensitive_list += data[1].tolist()
        if use_valid:
            print(f'Validation Accuracy: {100 * correct / total} %')
        else:
            print(f'Test Accuracy: {100 * correct / total} %')
        if is_ano_fair:
            # sens is true fairs, preds is predicted fairs. For fairness analysis, group by sensitives.
            agg_labels, agg_data, overall_labels, overall_data = fairness.eval_classifier(
                sensitive_list, preds_list, sensitive_list)
        else:
            agg_labels, agg_data, overall_labels, overall_data = fairness.eval_classifier(
                labels_list, preds_list, sensitive_list)

        # get file names
        if use_valid:
            valid = "valid"
        else:
            valid = "test"

        # NOTE: only writing to result csvs if the model is fully trained
        if epoch != self.c_epochs:
            return
        print("Saving csv data")

        overall_file = os.path.join(
            self.result_dir, f"{save_name}_{valid}_overall.csv")
        # overall_data = [str(x) for x in overall_data.tolist()]
        agg_file = os.path.join(
            self.result_dir, f"{save_name}_{valid}_aggregated.csv")
        # agg_data = [str(x) for x in agg_data.tolist()]

        utils.record_to_csv(overall_data.tolist(
        ) + [gen], overall_file, headers=overall_labels + ['generation'])
        utils.record_to_csv(
            agg_data.tolist() + [gen], agg_file, headers=agg_labels + ['generation'])
        return correct / total

    def visualize_outputs(self, gen_number, epoch):
        for images, sensitive, labels in self.test_loader:
            break

        with torch.no_grad():
            images = images.to(self.device)
            images, _, _ = self.gen_net(images)
            images = images.cpu()
            images = gen_utils.to_img(images)
            np_imagegrid = torch_utils.make_grid(images[1:50], 10, 5).numpy()
            recon_grid = np.transpose(np_imagegrid, (1, 2, 0))
            plt.imshow(recon_grid)
            save_path_re = os.path.join(
                self.fig_dir, f"gen_{gen_number}_epoch_{epoch}_recon.pdf")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            if self.seed == 0:
                plt.savefig(save_path_re, dpi=400)
            plt.cla()
            plt.clf()
            plt.close()
        print(f"Visual saved {save_path_re}")

    def get_mu_vars(self, init_model, gen_number):
        img_recon = self.sample_decoder(self.gen_net, batch_size=100).cpu()
        save_path_rec = os.path.join(
            self.fig_dir, f"gen_{gen_number}_latent.pdf")
        gen_utils.save_image(torch_utils.make_grid(
            img_recon.data[:100], 10, 5), save_path_rec)

        mus, vars = init_model.encoder(img_recon.to(self.device))
        mus = mus.cpu().detach().numpy()
        vars = vars.cpu().detach().numpy()
        return mus, vars

    def generated_population_stats(self, gen, gen_net, ano_fair_net, ano_label_net, batch_size=200):
        pred_threshold = .5  # pos class prediction threshold

        # 将模型设置为评估模式
        gen_net.eval()
        ano_fair_net.eval()
        ano_label_net.eval()

        # 从生成器模型中采样数据
        with torch.no_grad():
            images = self.sample_decoder(gen_net, batch_size=batch_size)

        # 使用标签分类模型获取预测的标签
        outputs = ano_label_net(images)
        _, labels = torch.max(outputs, 1)
        labels = labels.cpu().numpy()
        # 获取敏感标签
        _, sensitives = torch.max(outputs, 1)
        sensitives = sensitives.cpu().numpy()

        # 使用敏感属性分类模型获取敏感属性的预测
        outputs = ano_fair_net(images)
        outputs = outputs.squeeze(dim=1)
        # print(f"ano_fair_net output shape: {outputs.shape}")  # 调试输出维度
        # print(outputs)
        # print(outputs.squeeze(dim=1))
        # 确保输出有两个类
        # if outputs.shape[1] != 2:
        #     raise ValueError(
        #         f"ano_fair_net output should have 2 classes, but got shape {outputs.shape}")

        # outputs = F.softmax(outputs, dim=0)
        # print(outputs)
        # # 假设敏感属性输出为二分类 [0, 1]
        positive = outputs[outputs > 0]
        num_positive = positive.shape[0]
        negative = outputs[outputs == 0]
        num_negative = negative.shape[0]
        pos_prob = num_positive / outputs.shape[0]
        neg_prob = num_negative / outputs.shape[0]

        # # 计算标签平衡
        label_balance = np.sum(labels) / labels.shape[0]

        # # 计算敏感属性平衡
        color_balance = np.sum(sensitives) / sensitives.shape[0]

        # # 打印标签平衡和敏感属性平衡的百分比
        print(
            f"Estimated label balance {label_balance*100}% 1s, color balance {color_balance*100}% for 'sex'.")

        # # 定义保存文件的路径和表头
        headers = ['generation', 'label_bal', 'color_bal']
        headers_conf = ['proba_0', 'proba_1', 'generation']
        save_to = os.path.join(self.result_dir, f"gen_pop_stats.csv")
        sens_proba_save_to = os.path.join(
            self.result_dir, "confidence_sens.csv")

        # # 将生成的数据和置信度数据保存到 CSV 文件中
        data = [gen, label_balance, color_balance]
        data_conf = [neg_prob, pos_prob, gen]
        utils.record_to_csv(data, save_to, headers=headers)
        utils.record_to_csv(data_conf, sens_proba_save_to,
                            headers=headers_conf)

        return label_balance, color_balance
