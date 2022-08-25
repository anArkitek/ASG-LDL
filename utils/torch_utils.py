from math import degrees
from os import error
import torch
from torch import cos, sin, matmul, softmax
import numpy as np
from utils.sys_utils import isRotationMatrix, the300w_lp_R2Euler, normalizeVec, the300w_lp_R2axisAngle, the300w_lp_axisAngle2R, the300w_lp_Euler2R




class PointsGenerator:
    def __init__(self, num_pts):
        self.num_pts = num_pts
    
    def generate_pts(self):
        # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
        ratio = (1 + 5 ** 0.5)/2
        i = np.arange(0, self.num_pts)
        theta = 2 * np.pi * i / ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / self.num_pts)
        xs, ys, zs = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        GS_pts = np.concatenate((xs.reshape((-1,1)), ys.reshape((-1,1)), zs.reshape((-1,1))), axis=1)

        return GS_pts


class KentDistribution:
    def __init__(self, kappa, ellip, alpha, beta, gamma, gt_R) -> None:
        """ Initialize a kent distribution on given points: https://en.wikipedia.org/wiki/Kent_distribution
        Args:
            kappa (torch.Tensor): parameter of concentration; shape: Size(batch_size, 1)
            ellip (torch.Tensor): paramter of ellipse; shape: Size(batch_size, 1)
            alpha (torch.Tensor): roll: rotation along z axis in radian; shape: Size(batch_size, 1)
            beta (torch.Tensor): pitch: rotation along y axis in radian; shape: Size(batch_size, 1)
            gamma (torch.Tensor): yaw: rotation along x axis in radian; shape: Size(batch_size, 1)
            gt_R (torch.Tensor): the 3x3 orthogonal matrix formed by [gt_l, gt_d, gt_f]; shape: (batch_size, 3, 3)
        Returns:
            kent_probs (torch.Tensor): Probability distribution on sampled points; shape: (batch_size, num_pts)
        """
        # R = torch.tensor([[R_00, R_01, R_02],
        #                   [R_10, R_11, R_12],
        #                   [R_20, R_21, R_22]])
        # R_00.shape: Size(batch_size, 1)
        self.kappa = kappa.reshape(-1, 1).contiguous()
        self.ellip = ellip.reshape(-1, 1).contiguous()
        alpha = alpha
        beta = beta
        gamma = gamma
        R_00 = cos(alpha) * cos(beta)
        R_10 = sin(alpha) * cos(beta)
        R_20 = -sin(beta)
        R_01 = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)
        R_11 = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma)
        R_21 = cos(beta) * sin(gamma)
        R_02 = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)
        R_12 = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)
        R_22 = cos(beta) * cos(gamma)
        # RR_00.shape: Size(batch_size, 1)
        self.RR_00 = (R_00 * gt_R[:, 0, 0] + R_01 * gt_R[:, 1, 0] + R_02 * gt_R[:, 2, 0]).reshape(-1, 1).contiguous()
        self.RR_10 = (R_10 * gt_R[:, 0, 0] + R_11 * gt_R[:, 1, 0] + R_12 * gt_R[:, 2, 0]).reshape(-1, 1).contiguous()
        self.RR_20 = (R_20 * gt_R[:, 0, 0] + R_21 * gt_R[:, 1, 0] + R_22 * gt_R[:, 2, 0]).reshape(-1, 1).contiguous()
        self.RR_01 = (R_00 * gt_R[:, 0, 1] + R_01 * gt_R[:, 1, 1] + R_02 * gt_R[:, 2, 1]).reshape(-1, 1).contiguous()
        self.RR_11 = (R_10 * gt_R[:, 0, 1] + R_11 * gt_R[:, 1, 1] + R_12 * gt_R[:, 2, 1]).reshape(-1, 1).contiguous()
        self.RR_21 = (R_20 * gt_R[:, 0, 1] + R_21 * gt_R[:, 1, 1] + R_22 * gt_R[:, 2, 1]).reshape(-1, 1).contiguous()
        self.RR_02 = (R_00 * gt_R[:, 0, 2] + R_01 * gt_R[:, 1, 2] + R_02 * gt_R[:, 2, 2]).reshape(-1, 1).contiguous()
        self.RR_12 = (R_10 * gt_R[:, 0, 2] + R_11 * gt_R[:, 1, 2] + R_12 * gt_R[:, 2, 2]).reshape(-1, 1).contiguous()
        self.RR_22 = (R_20 * gt_R[:, 0, 2] + R_21 * gt_R[:, 1, 2] + R_22 * gt_R[:, 2, 2]).reshape(-1, 1).contiguous()

    def generate_probs(self, gs_pts, col_idx):
        # d0.shape(batch_size, num_pts) || pts[:, 0].T shape: (1, num_pts)
        d0 = matmul(self.RR_00, gs_pts[:, 0].reshape(1, -1)) + matmul(self.RR_10, gs_pts[:, 1].reshape(1, -1)) + matmul(self.RR_20, gs_pts[:, 2].reshape(1, -1))    
        d1 = matmul(self.RR_01, gs_pts[:, 0].reshape(1, -1)) + matmul(self.RR_11, gs_pts[:, 1].reshape(1, -1)) + matmul(self.RR_21, gs_pts[:, 2].reshape(1, -1))
        d2 = matmul(self.RR_02, gs_pts[:, 0].reshape(1, -1)) + matmul(self.RR_12, gs_pts[:, 1].reshape(1, -1)) + matmul(self.RR_22, gs_pts[:, 2].reshape(1, -1))

        if col_idx == 0:
            logits = self.kappa * d0 + self.ellip * (d1 ** 2 - d2 ** 2)
        elif col_idx == 1:
            logits = self.kappa * d1 + self.ellip * (d2 ** 2 - d0 ** 2)
        elif col_idx == 2:
            logits = self.kappa * d2 + self.ellip * (d0 ** 2 - d1 ** 2)
        kent_probs = softmax(logits, dim=1)

        # pred_probs.shape(batch_size, N_pts)
        return kent_probs


class ASGDistribution:

    def __init__(self, lambd, mu, l_vecs, d_vecs, f_vecs) -> None:
        """Initialize an Anisotropic Gaussian Distribution
        Args:
            lambd: bandwidth for x-axis; Size([batch_size, 1])
            mu:    bandwidth for y-axis; Size([batch_size, 1])
            l_vecs, d_vecs, f_vecs: ground truth of left, down, front vectors. each of Size([batch_size, 3])
        """
        self.lambd = lambd.reshape(-1, 1).contiguous()
        self.mu = mu.reshape(-1, 1).contiguous()
        self.l_vecs = l_vecs
        self.d_vecs = d_vecs
        self.f_vecs = f_vecs

    def generate_probs(self, gs_pts_T, col_idx):
        """ assign probability values to points

        Args:
            gs_pts_T (torch.Tensor): Size([3, num_pts])
            col_idx (int): 0/1/2
        Returns:
            ASG_probs (torch.Tensor): Probability distribution on sampled points; shape: (batch_size, num_pts)
        """
        if col_idx == 0:
            exp_term = torch.exp(- self.lambd * torch.matmul(self.d_vecs, gs_pts_T) ** 2
                                 - self.mu * torch.matmul(self.f_vecs, gs_pts_T) ** 2)  # Size([batch_size, num_pts])
            smooth_term = torch.matmul(self.l_vecs, gs_pts_T)
            smooth_term[smooth_term < 0.] = 0. # Size([batch_size, num_pts])
            probs = torch.softmax(smooth_term * exp_term, dim=1)  # Size([batch_size, num_pts])
        elif col_idx == 1:
            exp_term = torch.exp(- self.lambd * torch.matmul(self.f_vecs, gs_pts_T) ** 2 
                                 - self.mu * torch.matmul(self.l_vecs, gs_pts_T) ** 2)
            smooth_term = torch.matmul(self.d_vecs, gs_pts_T)
            smooth_term[smooth_term < 0.] = 0.
            probs = torch.softmax(smooth_term * exp_term, dim=1)
        elif col_idx == 2:
            exp_term = torch.exp(- self.lambd * torch.matmul(self.l_vecs, gs_pts_T) ** 2
                                 - self.mu * torch.matmul(self.d_vecs, gs_pts_T) ** 2)
            smooth_term = torch.matmul(self.f_vecs, gs_pts_T)
            smooth_term[smooth_term < 0.] = 0.
            probs = torch.softmax(smooth_term * exp_term, dim=1)

        return probs


class ErrorMeter:

    def __init__(self, opts) -> None:
        """Record and compute Euler angle errors
        """
        self.opts = opts
        self.error_record_dict = {}
        self.error_record_dict.update({"pitch_error_deg": 0., "yaw_error_deg": 0., "roll_error_deg": 0., "cnt": 0})
        self.error_record_dict.update({"l_error_deg": 0., "d_error_deg": 0., "f_error_deg": 0.})
        if self.opts.do_smooth:
            generator = PointsGenerator(self.opts.num_pts)
            self.gs_pts = generator.generate_pts()

    def _update(self, key, val):
        self.error_record_dict[key] += val

    def print_errors(self):
        mean_l_vec_error_deg = self.error_record_dict["l_error_deg"] / self.error_record_dict["cnt"]
        mean_d_vec_error_deg = self.error_record_dict["d_error_deg"] / self.error_record_dict["cnt"]
        mean_f_vec_error_deg = self.error_record_dict["f_error_deg"] / self.error_record_dict["cnt"]
        maev_deg = self.compute_maev()
        print("mean_left_vector_error_deg: {}".format(mean_l_vec_error_deg))
        print("mean_down_vector_error_deg: {}".format(mean_d_vec_error_deg))
        print("mean_font_vector_error_deg: {}".format(mean_f_vec_error_deg))
        print("maev: {}".format(maev_deg))

        mean_pitch_error_deg = self.error_record_dict["pitch_error_deg"] / self.error_record_dict["cnt"]
        mean_yaw_error_deg = self.error_record_dict["yaw_error_deg"] / self.error_record_dict["cnt"]
        mean_roll_error_deg = self.error_record_dict["roll_error_deg"] / self.error_record_dict["cnt"]
        mae_deg = self.compute_mae()
        print("mean_pitch_error_deg: {}".format(mean_pitch_error_deg))
        print("mean_yaw_error_deg: {}".format(mean_yaw_error_deg))
        print("mean_roll_error_deg: {}".format(mean_roll_error_deg))
        print("mae: {}".format(mae_deg))

    def compute_maev(self):
        return (self.error_record_dict["l_error_deg"] + 
                self.error_record_dict["d_error_deg"] + 
                self.error_record_dict["f_error_deg"]) / (3. * self.error_record_dict["cnt"])
    
    def compute_mae(self):
        return (self.error_record_dict["pitch_error_deg"] + 
                self.error_record_dict["yaw_error_deg"] + 
                self.error_record_dict["roll_error_deg"]) / (3. * self.error_record_dict["cnt"])

    def update_errors(self, pred, target):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        if self.opts.rot_type in ["lie", "euler"]:
            for i in range(pred.shape[0]):
                if self.opts.rot_type == "lie":
                    pred_angle_rad = np.linalg.norm(pred[i])
                    pred_axis = pred[i] / pred_angle_rad
                    pred_R = the300w_lp_axisAngle2R(pred_axis, pred_angle_rad, degrees=False)
                    gt_angle_rad = np.linalg.norm(target[i])
                    gt_axis = target[i] / gt_angle_rad
                    gt_R = the300w_lp_axisAngle2R(gt_axis, gt_angle_rad, degrees=False)
                elif self.opts.rot_type == "euler":
                    pred_R = the300w_lp_Euler2R(pred[i, 0], pred[i, 1], pred[i, 2], degrees=True)
                    gt_R = the300w_lp_Euler2R(target[i, 0], target[i, 1], target[i, 2], degrees=True)

                pred_pitch_deg, pred_yaw_deg, pred_roll_deg = the300w_lp_R2Euler(pred_R, degrees=True)
                gt_pitch_deg, gt_yaw_deg, gt_roll_deg = the300w_lp_R2Euler(gt_R, degrees=True)
                pred_l_vec, pred_d_vec, pred_f_vec = pred_R[:, 0], pred_R[:, 1], pred_R[:, 2]
                gt_l_vec, gt_d_vec, gt_f_vec = gt_R[:, 0], gt_R[:, 1], gt_R[:, 2]
                self._update("l_error_deg", np.arccos(np.clip(np.sum(pred_l_vec[i] * gt_l_vec), -1, 1)) * 180. / np.pi)
                self._update("d_error_deg", np.arccos(np.clip(np.sum(pred_d_vec[i] * gt_d_vec), -1, 1)) * 180. / np.pi)
                self._update("f_error_deg", np.arccos(np.clip(np.sum(pred_f_vec[i] * gt_f_vec), -1, 1)) * 180. / np.pi)
                self._update("pitch_error_deg", np.abs(pred_pitch_deg - gt_pitch_deg))
                self._update("yaw_error_deg", np.abs(pred_yaw_deg - gt_yaw_deg))
                self._update("roll_error_deg", np.abs(pred_roll_deg - gt_roll_deg))
                self._update("cnt", 1)
                return
        # ------------------------------------------------------------------ #
        if self.opts.rot_type == "rot_mat" and not self.opts.do_smooth:
            pred_l_vec = normalizeVec(pred[:,  : 3])
            pred_d_vec = normalizeVec(pred[:, 3: 6])
            pred_f_vec = normalizeVec(pred[:, 6:  ])
        elif self.opts.rot_type == "rot_mat" and self.opts.do_smooth:
            pred_probs = pred[:, 6: ]
            pred_l_probs = pred_probs[:,                      : self.opts.num_pts    ]
            pred_d_probs = pred_probs[:,     self.opts.num_pts: self.opts.num_pts * 2]
            pred_f_probs = pred_probs[:, self.opts.num_pts * 2:                      ]
            pred_l_vec = normalizeVec(np.matmul(pred_l_probs, self.gs_pts))
            pred_d_vec = normalizeVec(np.matmul(pred_d_probs, self.gs_pts))
            pred_f_vec = normalizeVec(np.matmul(pred_f_probs, self.gs_pts))

        #---------------------------------------------------------------#
        for i in range(pred.shape[0]):
            pred_R = np.array([pred_l_vec[i], pred_d_vec[i], pred_f_vec[i]]).T
            U, Sig, V_T = np.linalg.svd(pred_R)
            R_hat = np.matmul(U, V_T)
            assert isRotationMatrix(R_hat)
            pred_pitch_deg, pred_yaw_deg, pred_roll_deg = the300w_lp_R2Euler(R_hat, degrees=True)
            gt_R = target[i].reshape(3, 3).T
            assert isRotationMatrix(gt_R)
            gt_l_vec = target[i,  :3]
            gt_d_vec = target[i, 3:6]
            gt_f_vec = target[i, 6: ]
            gt_pitch_deg, gt_yaw_deg, gt_roll_deg = the300w_lp_R2Euler(gt_R, degrees=True)

            # print(gs_pts)
            # print("pred_l_vec: ", np.sum(pred_l_vec[i] ** 2))
            # print("l_error_deg: ", np.arccos(np.sum(pred_l_vec[i] * gt_l_vec)))
            self._update("l_error_deg", np.arccos(np.clip(np.sum(pred_l_vec[i] * gt_l_vec), -1, 1)) * 180. / np.pi)
            self._update("d_error_deg", np.arccos(np.clip(np.sum(pred_d_vec[i] * gt_d_vec), -1, 1)) * 180. / np.pi)
            self._update("f_error_deg", np.arccos(np.clip(np.sum(pred_f_vec[i] * gt_f_vec), -1, 1)) * 180. / np.pi)
            self._update("pitch_error_deg", np.abs(pred_pitch_deg - gt_pitch_deg))
            self._update("yaw_error_deg", np.abs(pred_yaw_deg - gt_yaw_deg))
            self._update("roll_error_deg", np.abs(pred_roll_deg - gt_roll_deg))
            self._update("cnt", 1)