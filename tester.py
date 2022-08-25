import os
import argparse
import numpy as np

import torch
from torch.utils.data import dataset, dataloader

from the300w_lp_dataset import The300WLPDataset
from networks.resnet import ResnetEncoder
from utils.sys_utils import normalizeQuat, the300w_lp_quat2euler, normalizeVec, isRotationMatrix, the300w_lp_R2Euler, the300w_lp_axisAngle2R
from utils.torch_utils import PointsGenerator

class Tester:
    def __init__(self, opts) -> None:
        self.opts = opts
        self.test_data_path_dict = {"AFLW2000": "./data/aflw2000", 
                                    "BIWI": "./data/biwi"}
        self.test_data_path = self.test_data_path_dict[self.opts.val_dataset]
        
        # load options
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
        self.opts.val_dataset_path = self.test_data_path
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
        
        self.load_model(self.opts.snapshot_path)

        # load dataset
        val_dataset = The300WLPDataset(opts=self.opts, is_train=False)
        self.test_loader = dataloader.DataLoader(dataset=val_dataset,
                                                 batch_size=1)


    def test(self):

        self.euler_error_dict = {"pitch": 0., "yaw": 0., "roll": 0.}
        self.quat_error_dict = {"q1": 0., "q2": 0., "q3": 0., "q4": 0.}
        self.vec_error_dict = {"l_vec": 0., "d_vec": 0., "f_vec": 0.}

        self.set_eval()
        error_record_dict = {}
        error_record_dict.update({"pitch_error_deg": 0., "yaw_error_deg": 0., "roll_error_deg": 0., "cnt": 0})
        error_record_dict.update({"l_error_deg": 0., "d_error_deg": 0., "f_error_deg": 0.})

        with torch.no_grad():
            for batch_idx, [[imgs, labels], [img_name, label_name, eulers]] in enumerate(self.test_loader):
                imgs = imgs.to(self.opts.device)
                labels = labels.numpy()
                out = self.models["backbone"](imgs).cpu().numpy()

                if self.opts.rot_type == "euler":
                    out = out
                    pred_pitch_deg, pred_yaw_deg, pred_roll_deg = out[:, 0], out[:, 1], out[:, 2]
                    gt_pitch_deg, gt_yaw_deg, gt_roll_deg = labels[:, 0], labels[:, 1], labels[:, 2]
                    l_vec_error_deg = 0
                    d_vec_error_deg = 0
                    f_vec_error_deg = 0

                elif self.opts.rot_type == "lie":
                    pred_angle_rad = np.linalg.norm(out[0])
                    pred_axis = out[0] / pred_angle_rad
                    pred_R = the300w_lp_axisAngle2R(pred_axis, pred_angle_rad, degrees=False)
                    gt_angle_rad = np.linalg.norm(labels[0])
                    gt_axis = labels[0] / gt_angle_rad
                    gt_R = the300w_lp_axisAngle2R(gt_axis, gt_angle_rad, degrees=False)
                    pred_pitch_deg, pred_yaw_deg, pred_roll_deg = the300w_lp_R2Euler(pred_R, degrees=True)
                    gt_pitch_deg, gt_yaw_deg, gt_roll_deg = the300w_lp_R2Euler(gt_R, degrees=True)
                    pred_l_vec, pred_d_vec, pred_f_vec = pred_R[:, 0], pred_R[:, 1], pred_R[:, 2]
                    gt_l_vec, gt_d_vec, gt_f_vec = gt_R[:, 0], gt_R[:, 1], gt_R[:, 2]
                    
                    l_vec_error_deg = np.arccos(np.clip(np.sum(pred_l_vec * gt_l_vec), -1, 1)) * 180. / np.pi
                    d_vec_error_deg = np.arccos(np.clip(np.sum(pred_d_vec * gt_d_vec), -1, 1)) * 180. / np.pi
                    f_vec_error_deg = np.arccos(np.clip(np.sum(pred_f_vec * gt_f_vec), -1, 1)) * 180. / np.pi
                elif self.opts.rot_type == "quat":
                    pred_quat = normalizeQuat(out)
                    pred_pitch_deg, pred_yaw_deg, pred_roll_deg = the300w_lp_quat2euler(pred_quat)
                    gt_pitch_deg, gt_yaw_deg, gt_roll_deg = the300w_lp_quat2euler(labels)
                elif self.opts.rot_type == "rot_mat":
                    if not self.opts.do_smooth:
                        pred_l_vec = normalizeVec(vec=out[:, :3])
                        pred_d_vec = normalizeVec(vec=out[:, 3:6])
                        pred_f_vec = normalizeVec(vec=out[:, 6:])
                    else:
                        out = out[:, 6:]
                        pred_l_vec = normalizeVec(np.matmul(out[:, : self.opts.num_pts], self.gs_pts))
                        pred_d_vec = normalizeVec(np.matmul(out[:, self.opts.num_pts : self.opts.num_pts * 2], self.gs_pts))
                        pred_f_vec = normalizeVec(np.matmul(out[:, self.opts.num_pts * 2: ], self.gs_pts))

                    l_vec_error_deg = np.arccos(np.clip(np.sum(pred_l_vec * labels[:, :3], axis=1), -1, 1)) * 180. / np.pi
                    d_vec_error_deg = np.arccos(np.clip(np.sum(pred_d_vec * labels[:, 3:6], axis=1), -1, 1)) * 180. / np.pi
                    f_vec_error_deg = np.arccos(np.clip(np.sum(pred_f_vec * labels[:, 6:], axis=1), -1, 1)) * 180. / np.pi
                    # error_record_dict["l_error_deg"] += l_vec_error_deg[0]
                    # error_record_dict["d_error_deg"] += d_vec_error_deg[0]
                    # error_record_dict["f_error_deg"] += f_vec_error_deg[0]
                    #---------------------------------------------------------------#
                    pred_R = np.array([pred_l_vec[0], pred_d_vec[0], pred_f_vec[0]]).T
                    U, Sig, V_T = np.linalg.svd(pred_R)
                    R_hat = np.matmul(U, V_T)
                    assert isRotationMatrix(R_hat)
                    pred_pitch_deg, pred_yaw_deg, pred_roll_deg = the300w_lp_R2Euler(R_hat, degrees=True)
                    gt_R = labels.reshape(3, 3).T
                    assert isRotationMatrix(gt_R)
                    gt_pitch_deg, gt_yaw_deg, gt_roll_deg = the300w_lp_R2Euler(gt_R, degrees=True)

                pitch_error = np.abs(pred_pitch_deg - gt_pitch_deg)
                yaw_error = np.abs(pred_yaw_deg - gt_yaw_deg)
                roll_error = np.abs(pred_roll_deg - gt_roll_deg)
                error_record_dict["l_error_deg"] += l_vec_error_deg
                error_record_dict["d_error_deg"] += d_vec_error_deg
                error_record_dict["f_error_deg"] += f_vec_error_deg
                error_record_dict["pitch_error_deg"] += pitch_error
                error_record_dict["yaw_error_deg"] += yaw_error
                error_record_dict["roll_error_deg"] += roll_error
                error_record_dict["cnt"] += 1
        
        # if self.opts.rot_type == "rot_mat":
        mean_l_vec_error_deg = error_record_dict["l_error_deg"] / error_record_dict["cnt"]
        mean_d_vec_error_deg = error_record_dict["d_error_deg"] / error_record_dict["cnt"]
        mean_f_vec_error_deg = error_record_dict["f_error_deg"] / error_record_dict["cnt"]
        maev_deg = (mean_l_vec_error_deg + mean_d_vec_error_deg + mean_f_vec_error_deg) / 3.
        print("mean_left_vector_error_deg: {}".format(mean_l_vec_error_deg))
        print("mean_down_vector_error_deg: {}".format(mean_d_vec_error_deg))
        print("mean_font_vector_error_deg: {}".format(mean_f_vec_error_deg))
        print("maev: {}".format(maev_deg))

        mean_pitch_error_deg = error_record_dict["pitch_error_deg"] / error_record_dict["cnt"]
        mean_yaw_error_deg = error_record_dict["yaw_error_deg"] / error_record_dict["cnt"]
        mean_roll_error_deg = error_record_dict["roll_error_deg"] / error_record_dict["cnt"]
        mae_deg = (mean_pitch_error_deg + mean_yaw_error_deg + mean_roll_error_deg) / 3.
        print("mean_pitch_error_deg: {}".format(mean_pitch_error_deg))
        print("mean_yaw_error_deg: {}".format(mean_yaw_error_deg))
        print("mean_roll_error_deg: {}".format(mean_roll_error_deg))
        print("mae: {}".format(mae_deg))
   
    def load_model(self, model_path):
        
        assert os.path.isdir(model_path), "Cannot find folder {}".format(model_path)
        print("loading model from folder {}".format(model_path))

        for model_name in self.models:
            print("Loading {} weights...".format(model_name))
            path = os.path.join(model_path, "{}.pth".format(model_name))
            model_dict = self.models[model_name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[model_name].load_state_dict(model_dict)


    def set_eval(self):
        for m in self.models.values():
            m.eval()
        print("Eval status has been set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Smoothing Options Tester")
    parser.add_argument("--snapshot_path", help="path to pre-trained model directory")
    parser.add_argument("--val_dataset", type=str, choices=["AFLW2000", "BIWI"])
    opts = parser.parse_args()
    
    tester = Tester(opts)

    tester.test()