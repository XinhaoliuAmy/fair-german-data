# FAIRNESS FEEDBACK LOOPS: TRAINING ON SYNTHETIC DATA AMPLIFIES BIAS.

This is the official code repository for this paper. 

## Running:
We use Python 3.9 and the libraries listed in `requirements.txt`, install with `pip -r requirements.txt`.

Set experimental parameters and run the SeqGenSeqClass or SeqGenNonSeqClass settings with `mc_train.py` (training with model collapse).
Similarly, use `nomc_train.py` for the SeqClass setting (no model collapse). 
Regardless, the brunt of the code is carried out in `train.py`, which holds a class of all required parameters, architectures, etc, to conduct SeqGenSeqClass, our most complicated setting. Of most relevance will likely be the `train_generator`, `train_classifier`, and `reparation_batch` functions. Models for both generators and classifiers are in `models.py` file (all together), or separately in the `models` directory. Many other algorithmic choices are found in the `utils.py` file. 

### Datasets:
All in `prep_data` directory.
1. Default parameters in `_train.py` files correspond to ColoredMNIST, which is created by `ci_mnist.py`.
2. ColoredSVHN is refered to as SVHN in this repo. Created by `SVHN.py`
3. CelebA created in `prep_celeba.py`. Note that conducting experiments for this setting is expensive.
4. FairFace is created in `fairface.py`, again experimenting with this dataset is expensive.
Please see the Appendix D for details on the datasets, models, and compute.

### Structure:
There are several plotting scripts in the `plotting_script` directory.
1. The `plotting.py` can generate figures illustrating MIDS for one experiment. 
2. `comparisons.py` plots the figures used to comapare MIDS and reparations. 
3. `plotting_ablation.py` creates figures for our abblation studies.
4. `overal_model_stats.py` can recall the average and stdev performances for A_L and A_S

### Note:
The names of the settings in the code differ from the paper:
* MC. Model collapse, SeqGen (usually SeqGenSeqClass).
* ARCLA. Classifier-side AR for SegGenSeqClass. In ColoredMNIST, this is just 'AR'.
* ARGEN. Generator-side AR for SeqGenSeqClass.
* MCNOSEQ. Model collapse with nonsequential classifiers, SegGenNonSeqClass.
* NOMC. No model collapse, SeqClass.
* ARNOMC. AR + NOMC, classifier-side AR for SeqClass

Additionally, references to sensitive groups usually use red and green to denote the minoritized and majoritized groups, as in ColoredMNIST and ColoredSVHN.

An arxiv version of this work may be found here: https://arxiv.org/abs/2403.07857

