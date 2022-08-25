
from __future__ import absolute_import, division, print_function

import os

from torchvision import models
import torch
import numpy as np
import random

from trainer import Trainer
from options import GaussianSmoothingOptions


opts = GaussianSmoothingOptions().parse(toTerminal=True)

data_path = {"300W-LP": "./data/300w_lp/", 
             "BIWI": "./data/biwi/"}

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    opts.train_dataset_path = os.path.join(data_path[opts.train_dataset], "train")
    opts.val_dataset_path = os.path.join(data_path[opts.val_dataset], "val")

    # record params
    with open('train_opt_log.txt','w') as file:
        for k in sorted (vars(opts).keys()):
           file.write("'%s':'%s', \n" % (k, vars(opts)[k]))

    setup_seed(30)

    trainer = Trainer(opts)

    trainer.train()
