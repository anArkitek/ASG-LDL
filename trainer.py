from functools import reduce
import os
import sys
import time
import argparse

import torch
from torch.nn.functional import normalize
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter

import torchvision

from networks.resnet import ResnetEncoder
from the300w_lp_dataset import The300WLPDataset
from utils.sys_utils import isRotationMatrix, normalizeQuat, normalizeVec, quat_from_file, the300w_lp_R2Euler
from utils.torch_utils import KentDistribution, PointsGenerator, ErrorMeter
from losses import LossCalculator

class Trainer:
    def __init__(self, opts: argparse.Namespace) -> None:
        self.opts = opts
        self.loss_calculator = LossCalculator(self.opts)
        if self.opts.rot_type in ["euler", "lie"]  and not self.opts.do_smooth:
            self.out_features = 3
        elif self.opts.rot_type == "quat" and not self.opts.do_smooth:
            self.out_features = 4
        elif self.opts.rot_type == "rot_mat" and not self.opts.do_smooth:
            self.out_features = 9
        elif self.opts.rot_type == "rot_mat" and self.opts.do_smooth:
            self.out_features = self.opts.num_pts * 3
            gs = PointsGenerator(self.opts.num_pts)
            self.gs_pts = torch.tensor(gs.generate_pts(), dtype=torch.float32).to(self.opts.device)
            self.gs_pts_T = self.gs_pts.permute(1, 0).contiguous()

        self.indentifier = self.create_model_indentifier()
        print(self.indentifier)

        # BACKBONE
        if self.opts.backbone == "resnet50":
            backbone = ResnetEncoder(opts=self.opts, num_layers=50, out_features=self.out_features, pretrained=True)
        elif self.opts.backbone == "resnet18":
            backbone = ResnetEncoder(opts=self.opts, num_layers=18, out_features=self.out_features, pretrained=True)
        else:
            sys.exit("Not supported backbone.")
        self.models = {}
        self.models["backbone"] = backbone
        self.models["backbone"].to(self.opts.device)

        # OPTIMIZER
        self.parameters_to_train = self.models["backbone"].parameters()
        self.optimizer = Adam(self.parameters_to_train, self.opts.learning_rate)
        self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=self.opts.lr_gamma)

        if self.opts.snapshot_path is not None:
            self.load_model()

        # DATASET
        train_dataset = The300WLPDataset(self.opts, is_train=True)
        val_dataset = The300WLPDataset(self.opts, is_train=False)
        self.train_loader = dataloader.DataLoader(dataset=train_dataset, 
                                                  batch_size=self.opts.batch_size, 
                                                  shuffle=True, 
                                                  num_workers=self.opts.num_workers, 
                                                  pin_memory=True)
        self.val_loader = dataloader.DataLoader(dataset=val_dataset,
                                                batch_size=1,
                                                num_workers=self.opts.num_workers,
                                                pin_memory=True)

        # ------------------------------ steps ------------------------------ #
        self.train_iter_nums = len(train_dataset) // self.opts.batch_size * self.opts.num_epochs
        self.val_iter_nums = len(train_dataset) // self.opts.batch_size
        
        # ------------------------------ tensorboard ------------------------------ #
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(log_dir=os.path.join(self.opts.tensorboard_path, 
                                                                    self.indentifier))

        # ------------------------------ loss ------------------------------ #
        self.mse = torch.nn.MSELoss(reduction="mean").to(self.opts.device)
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(self.opts.device)


    def train(self):
        self.start_time = time.time()
        self.min_mae = sys.float_info.max

        for self.curr_epoch in range(self.opts.num_epochs):
            self.run_epoch()
            self.error_meter = ErrorMeter(self.opts)
            val_loss = self.val()
            curr_mae = self.error_meter.compute_mae()
            self.error_meter.print_errors()
            if curr_mae < self.min_mae:
                self.save_model()
                self.min_mae = curr_mae

    def run_epoch(self):
        print("-" * 30, "Start Training", "-" * 30)
        self.set_train()
        for self.curr_batch_idx, [[imgs, labels], [img_names, label_names, eulers]] in enumerate(self.train_loader):

            imgs = imgs.to(self.opts.device)
            labels = labels.to(self.opts.device)
            out = self.models["backbone"](imgs)
            train_losses = self.compute_loss(out, labels)
            self.write2Tensorboard(mode="train", losses=train_losses)
            if (self.curr_batch_idx + 1) % 200 == 0:
                self.write2Terminal(mode="train", losses=train_losses)

            # self.writeASGParams(eulers, out[:, :6])

            self.optimizer.zero_grad()
            train_losses["total_loss"].backward()
            self.optimizer.step()

        self.lr_scheduler.step()
    

    def compute_loss(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Size([batch_size, M])
            gt (torch.Tensor): Size([batch_size, M])
        Returns:
            Dict of losses
        """
        losses = {}
        total_loss = 0.
        if self.opts.rot_type in ["euler", "lie"] and not self.opts.do_smooth:
            losses["mse"] = self.mse(pred, gt)
        elif self.opts.rot_type == "quat" and not self.opts.do_smooth:
            pred = normalizeQuat(quat=pred)
            losses["mse"] = self.mse(pred, gt)
        elif self.opts.rot_type == "rot_mat" and not self.opts.do_smooth:
            losses = self.loss_calculator.rot_mat_loss(pred, gt)    
        elif self.opts.rot_type == "rot_mat" and self.opts.do_smooth:
            losses = self.loss_calculator.rot_mat_loss_with_asg(pred, gt)

        for loss_name, loss_val in losses.items():
            total_loss += loss_val
        losses["total_loss"] = total_loss
        return losses


    def val(self):

        self.set_eval()
        total_val_losses = {}
        with torch.no_grad():
            for batch_idx, [[imgs, labels], [img_indices, label_indices, eulers]] in enumerate(self.val_loader):
                imgs = imgs.to(self.opts.device)
                labels = labels.to(self.opts.device)
                out = self.models["backbone"](imgs)
                
                val_losses = self.compute_loss(out, labels)
                self.error_meter.update_errors(pred=out, target=labels)
                self.write2Tensorboard(mode="val", losses=val_losses)

                for key, val in val_losses.items():
                    if key in total_val_losses.keys():
                        total_val_losses[key] += val / len(self.val_loader)
                    else:
                        total_val_losses[key] = val / len(self.val_loader)
        self.write2Terminal(mode="val", losses=total_val_losses)
        return total_val_losses["total_loss"].cpu().item()


    def create_model_indentifier(self):
        model_dir = "{}_{}_smooth:{}".format(self.opts.backbone, self.opts.rot_type, self.opts.do_smooth)
        if self.opts.rot_type == "rot_mat":
            if not self.opts.do_smooth:
                weight_info_dir = "ortho_loss_weight:{:.2f}".format(self.opts.ortho_loss_weight)
            else:
                weight_info_dir = "numPts:{:d}_softLossWeight:{:.3f}_shapeRegWeight:{:.3f}_orthoLossWeight:{:.3f}".format(
                    self.opts.num_pts, self.opts.soft_loss_weight, self.opts.shape_regular_weight, self.opts.ortho_loss_weight
                )
            identifier = os.path.join(model_dir, weight_info_dir)
        else:
            identifier = model_dir
        return identifier


    def save_model(self):
        save_dir = os.path.join(self.opts.prefix, self.opts.save_path, self.indentifier, "epoch_{:d}".format(self.curr_epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("saving model to {}".format(save_dir))

        for model_name, model in self.models.items():
            model_save_path = os.path.join(save_dir, "{}.pth".format(model_name))
            model2save = model.state_dict()
            torch.save(model2save, model_save_path)

        optim_save_path = os.path.join(save_dir, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), optim_save_path)

        with open(os.path.join(save_dir, "train_opt_log.txt"), "a") as f:
            for k in sorted (vars(self.opts).keys()):
              f.write("'%s':'%s', \n" % (k, vars(self.opts)[k]))


    def load_model(self):
        with open(os.path.join(self.opts.snapshot_path, "train_opt_log.txt"), "r") as f:
            lines = f.readlines()
            self.opt_dict = {}
            for line in lines:
                key = line.split(":")[0].strip("'")
                val = line[len(key) + 3: -3].strip("'")
                self.opt_dict[key] = val

        self.opts.rot_type = self.opt_dict["rot_type"]
        self.opts.do_smooth = {"True": True, "False": False}[self.opt_dict["do_smooth"]]
        self.opts.backbone = self.opt_dict["backbone"]
        self.opts.device = self.opt_dict["device"]
        self.opts.img_size = int(self.opt_dict["img_size"])
        self.opts.num_pts = int(self.opt_dict["num_pts"])
        self.opts.max_kappa = float(self.opt_dict["max_kappa"])
        
        # load_ model
        if self.opts.rot_type in ["euler", "lie"] and not self.opts.do_smooth:
            self.out_features = 3
        elif self.opts.rot_type == "quat" and not self.opts.do_smooth:
            self.out_features = 4
        elif self.opts.rot_type == "rot_mat" and not self.opts.do_smooth:
            self.out_features = 9
        elif self.opts.rot_type == "rot_mat" and self.opts.do_smooth:
            self.out_features = self.opts.num_pts * 3
            gs = PointsGenerator(self.opts.num_pts)
            self.gs_pts = gs.generate_pts()

        if self.opts.backbone == "resnet50":
            backbone = ResnetEncoder(opts=self.opts, num_layers=50, out_features=self.out_features, pretrained=True)
        elif self.opts.backbone == "resnet18":
            backbone = ResnetEncoder(opts=self.opts, num_layers=18, out_features=self.out_features, pretrained=True)
        
        self.models = {}
        self.models["backbone"] = backbone
        self.models["backbone"].to(self.opts.device)
        
        assert os.path.isdir(self.opts.snapshot_path), "Cannot find folder {}".format(self.opts.snapshot_path)
        print("loading model from folder {}".format(self.opts.snapshot_path))

        for model_name in self.models:
            print("Loading {} weights...".format(model_name))
            path = os.path.join(self.opts.snapshot_path, "{}.pth".format(model_name))
            model_dict = self.models[model_name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[model_name].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.opts.snapshot_path, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


    def set_train(self):
        for m in self.models.values():
            m.train()
        print("Training status has been set.")


    def set_eval(self):
        for m in self.models.values():
            m.eval()
        print("Eval status has been set.")


    def write2Tensorboard(self, mode, losses):
        for loss_name, loss_val in losses.items():
             self.writers[mode].add_scalar("{}/{}".format(mode, loss_name), loss_val, self.curr_epoch)
             

    def write2Terminal(self, mode, losses):
        if mode == "train":
            msg = "Epoch: {}/{} | Iter: {}/{} | ".format(
                self.curr_epoch, self.opts.num_epochs, 
                self.curr_batch_idx, self.train_iter_nums)
            for key, val in losses.items():
                msg += "{}: {:.8f} | ".format(key, val.cpu().item())
                
        else:
            msg = "val_loss || "
            for key, val in losses.items():
                msg += "{}: {:.8f} | ".format(key, val.cpu().item())

        print(msg)

    def writeASGParams(self, eulers, params):
        with open("300W-LP-ASG-PARAMS.txt", "a+") as f:
            for i in range(params.shape[0]):
                f.write(str(eulers[i, 0].item()) + "," + str(eulers[i, 1].item()) + "," + str(eulers[i, 2].item()) + "," + \
                        str(params[i, 0].item()) + "," + str(params[i, 1].item()) + "," + str(params[i, 2].item()) + "," + \
                        str(params[i, 3].item()) + "," + str(params[i, 4].item()) + "," + str(params[i, 5].item()) + "\n")