import torch
from copy import deepcopy
import utils
import model
import argparse
import train
import os
from plotting_scripts import plotting

parser = argparse.ArgumentParser()
parser.add_argument('--number-generations', type=int,
                    default=5, help='number gens testing')
parser.add_argument('--gen-batch-size', type=int, default=256)
parser.add_argument('--cla-batch-size', type=int, default=256)
parser.add_argument('--gen-lr', type=float, default=0.001)
parser.add_argument('--cla-lr', type=float, default=0.001)
parser.add_argument('--gen-epochs', type=int, default=100)
parser.add_argument('--cla-epochs', type=int, default=100)
parser.add_argument('--gen-optimizer', type=str, default="ADAM")
parser.add_argument('--cla-optimizer', type=str, default="sgd")
parser.add_argument('--dataset', type=str, default="German_data")
parser.add_argument('--gen-model', type=str, default="TabularGenerator")
parser.add_argument('--cla-model', type=str, default="TabularClassifier")
parser.add_argument('--save-freq', type=int, default=5,
                    help='frequency of saving checkpoints in epochs')
parser.add_argument('--eval-freq', type=int, default=5,
                    help='frequency for model evaluation in epochs')
parser.add_argument('--gamma', type=float, default=.95,
                    help='for learning rate schedule')
# using slurm task ids as seeds
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--overwrite', type=int, default=0,
                    help='if set to 1 and save_dir non-empty, then will empty the save dir')
parser.add_argument('--id', type=str, default='debugging')
parser.add_argument('--gp-n', type=float, default=0.2875536480686695,
                    help='Proportion of preferable class belonging to advantaged group')
parser.add_argument('--gp-p', type=float, default=0.67,
                    help='Proportion of non-preferable class belonging to advantaged group')
parser.add_argument('--pos-class-thresh', type=int, default=5,
                    help='Lowest MNIST number considered part of class 1 (for label imbalance)')
parser.add_argument('--synthetic-perc', type=float, default=1,
                    help='Proprtion of data to sample from generator when training next generator.')
parser.add_argument('--use-reparation', type=str, default='cla',
                    help='If using reparation, set to cla, gen, or both.')
parser.add_argument('--rep-budget', type=int, default=1,
                    help='Number extra samples to take to meet reparation ideal batch')
parser.add_argument('--latent-dims', type=int, default=58,
                    help='Latent dims used by generators')
parser.add_argument('--roll-ckpts', type=int, default=0,
                    help='Set to 1 if need to overwrite generations due to size (celeba)')
arg = parser.parse_args()

try:
    classifier = eval(f"model.{arg.cla_model}")
    generator = eval(f"model.{arg.gen_model}")
except:
    classifier = eval(f"torchvision.models.{arg.cla_model}")
    generator = eval(f"torchvision.models.{arg.gen_model}")
# print(generator)#<function vae at 0x7fecf9e09700>
# print(classifier)#<class 'model.SimpleCNN'>
train_fn = train.train_fn(arg.gen_lr,
                          arg.gen_batch_size,
                          arg.cla_lr,
                          arg.cla_batch_size,
                          arg.dataset,
                          generator,
                          classifier,
                          exp_id=arg.id,
                          save_freq=arg.save_freq,
                          eval_freq=arg.eval_freq,
                          g_optimizer=arg.gen_optimizer,
                          g_epochs=arg.gen_epochs,
                          c_optimizer=arg.cla_optimizer,
                          c_epochs=arg.cla_epochs,
                          seed=arg.seed,
                          overwrite=arg.overwrite,
                          sex_probability=[
                              arg.gp_n, arg.gp_p],
                          pos_class_thresh=arg.pos_class_thresh,
                          synthetic_perc=arg.synthetic_perc,
                          use_reparation=arg.use_reparation,
                          rep_budget=arg.rep_budget,
                          latent_dims=arg.latent_dims,
                          gamma=arg.gamma,
                          roll_ckpts=arg.roll_ckpts,
                          input_dim=58,
                          hidden_dim=128,
                          output_dim=58,
                          #   good_bad_dim=1,
                          #   sensitive_dim=1,
                          output_dim_classifer=1,
                          input_dim_classifer=58)
# print(train_fn)  # <train.train_fn object at 0x7fbe4c390ac0>
# 在这里train_fn已经成立
# train annotators
# 根据之前的fairfeedbackloop，我们只用这一个应该也得到平衡的结果：green_neg = red_neg
# 定义标签注释器
print("定义标签注释器")
ano_lab_net = deepcopy(train_fn.train_classifier(is_ano_lab=True))
# 分类器
print("定义分类器")
prev_cla_net = deepcopy(train_fn.train_classifier(
    is_ano_lab=True))  # this one should not be none
# print(prev_cla_net == None)
# 敏感注释器
print("定义敏感注释器")
ano_fair_net = deepcopy(train_fn.train_classifier(is_ano_fair=True))

# Start at most recent generation
start_gen = 0
keyword = f"gen"
last_ckpt = utils.get_last_gen(train_fn.save_dir, keyword)
if last_ckpt >= 0:
    start_gen = last_ckpt
print(f"\nStarting at genration {start_gen}\n")

muss = []
for gen in range(start_gen, arg.number_generations):
    if gen == 0:  # first generator trained on original data
        prev_gen_net = train_fn.train_generator(
            0, sample_generator=False, sample_from=None, label_from=ano_lab_net, group_from=ano_fair_net)
        init_gen = deepcopy(prev_gen_net)
    elif gen == start_gen:
        # generator already exists, so load it
        prev_gen_net = train_fn.train_generator(
            start_gen, False, None, None, None)
    else:
        prev_gen_net = train_fn.train_generator(
            gen, sample_generator=True, sample_from=prev_gen_net, label_from=prev_cla_net, group_from=ano_fair_net)

    # measure generated population
    train_fn.generated_population_stats(
        gen, prev_gen_net, ano_fair_net, ano_lab_net)

#     # train classifier on prev_gen_net
#     # SEQUENTIAL CLASSIFIER  # if sequential, train classifier from labels of prior classifier
#     print(prev_cla_net == None)
#     prev_cla_net = train_fn.train_classifier(
#         gen, og_rate=0, sample_from=prev_gen_net, label_from=prev_cla_net, group_from=ano_fair_net)  # here we need to change the asseration of rep_budget to make sure rep_budget > 0
#     # so here the sequential classifier od not work
#     # NONSEQ CLASSIFIER
#     # prev_cla_net = train_fn.train_classifier(gen, og_rate=0, sample_from=prev_gen_net, label_from=ano_lab_net, group_from=ano_fair_net)
