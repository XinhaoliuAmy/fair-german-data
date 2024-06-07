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
                    default=1, help='number gens testing')
parser.add_argument('--gen-batch-size', type=int, default=256)
parser.add_argument('--cla-batch-size', type=int, default=256)
parser.add_argument('--gen-lr', type=float, default=1e-3)
parser.add_argument('--cla-lr', type=float, default=0.1)
parser.add_argument('--gen-epochs', type=int, default=30)
parser.add_argument('--cla-epochs', type=int, default=30)
parser.add_argument('--gen-optimizer', type=str, default="ADAM")
parser.add_argument('--cla-optimizer', type=str, default="sgd")
parser.add_argument('--dataset', type=str, default="ColoredMNIST")
parser.add_argument('--gen-model', type=str, default="vae")
parser.add_argument('--cla-model', type=str, default="SimpleCNN")
parser.add_argument('--save-freq', type=int, default=5,
                    help='frequency of saving checkpoints in epochs')
parser.add_argument('--eval-freq', type=int, default=5,
                    help='frequency for model evaluation in epochs')
parser.add_argument('--gamma', type=float, default=None,
                    help='for learning rate schedule')
# using slurm task ids as seeds
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--overwrite', type=int, default=0,
                    help='if set to 1 and save_dir non-empty, then will empty the save dir')
parser.add_argument('--id', type=str, default='debugging')
parser.add_argument('--gp-n', type=float, default=.3,
                    help='Proportion of class 0 (negative) belonging to advantaged group')
parser.add_argument('--gp-p', type=float, default=.7,
                    help='Proportion of class 1 (positive) belonging to advantaged group')
parser.add_argument('--pos-class-thresh', type=int, default=5,
                    help='Lowest MNIST number considered part of class 1 (for label imbalance)')
parser.add_argument('--synthetic-perc', type=float, default=1,
                    help='Proprtion of data to sample from generator when training next generator.')
parser.add_argument('--use-reparation', type=str, default='cla',
                    help='If using reparation, set to cla, gen, or both. Or anything else if no rep.')
parser.add_argument('--rep-budget', type=int, default=2,
                    help='Number extra samples to take to meet reparation ideal batch')
parser.add_argument('--latent-dims', type=int, default=20,
                    help='Latent dims used by generators')
parser.add_argument('--roll-ckpts', type=int, default=0,
                    help='True if overwrite generations due to size (celeba)')
arg = parser.parse_args()

try:
    classifier = eval(f"model.{arg.cla_model}")
    generator = eval(f"model.{arg.gen_model}")
except:
    classifier = eval(f"torchvision.models.{arg.cla_model}")
    generator = eval(f"torchvision.models.{arg.gen_model}")

print(generator)
print(classifier)

train_fn = train.train_fn(arg.gen_lr, arg.gen_batch_size, arg.cla_lr, arg.cla_batch_size,
                          arg.dataset, generator, classifier,
                          exp_id=arg.id, save_freq=arg.save_freq, eval_freq=arg.eval_freq,
                          g_optimizer=arg.gen_optimizer, g_epochs=arg.gen_epochs,
                          c_optimizer=arg.cla_optimizer, c_epochs=arg.cla_epochs,
                          seed=arg.seed, overwrite=arg.overwrite, green_probas=[
                              arg.gp_n, arg.gp_p],
                          pos_class_thresh=arg.pos_class_thresh, synthetic_perc=arg.synthetic_perc,
                          use_reparation=arg.use_reparation, rep_budget=arg.rep_budget, latent_dims=arg.latent_dims,
                          roll_ckpts=arg.roll_ckpts)

# train annotators
ano_lab_net = deepcopy(train_fn.train_classifier(is_ano_lab=True))
prev_cla_net = deepcopy(ano_lab_net)
ano_fair_net = deepcopy(train_fn.train_classifier(is_ano_fair=True))
gen_net = train_fn.train_generator(0, sample_generator=False, sample_from=None)

start_gen = 0
keyword = f"cla"
last_ckpt = utils.get_last_gen(train_fn.save_dir, keyword)
if last_ckpt >= 0:
    start_gen = last_ckpt
print(f"\nStarting at genration {start_gen}\n")

# just looking at how labels change over time, using generator to sample from
for gen in range(start_gen, arg.number_generations):
    # train classifier on prev_gen_net
    train_fn.generated_population_stats(
        gen, gen_net, ano_fair_net, prev_cla_net)
    prev_cla_net = train_fn.train_classifier(
        gen, og_rate=0, sample_from=gen_net, label_from=prev_cla_net, group_from=ano_fair_net)

train_fn.generated_population_stats(
    arg.number_generations, gen_net, ano_fair_net, prev_cla_net)
