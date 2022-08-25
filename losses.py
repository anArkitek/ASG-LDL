from numpy.lib.shape_base import expand_dims
import torch
from torch.nn import MSELoss

from utils.sys_utils import normalizeVec
from utils.torch_utils import PointsGenerator, KentDistribution, ErrorMeter, ASGDistribution


def ortho_loss(vec_1, vec_2, vec_3):
    ortho_12 = torch.sum(vec_1 * vec_2, dim=1) ** 2
    ortho_23 = torch.sum(vec_2 * vec_3, dim=1) ** 2
    orhto_13 = torch.sum(vec_1 * vec_3, dim=1) ** 2
    return torch.mean(ortho_12 + ortho_23 + orhto_13)


class LossCalculator:

    def __init__(self, opts) -> None:
        self.opts = opts
        self.mse = torch.nn.MSELoss(reduction="mean").to(self.opts.device)
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(self.opts.device)

        if self.opts.rot_type == "rot_mat" and self.opts.do_smooth:
            self.gs = PointsGenerator(self.opts.num_pts)
            print(opts.device)
            self.gs_pts = torch.tensor(self.gs.generate_pts(), dtype=torch.float32).to(self.opts.device)
            self.gs_pts_T = self.gs_pts.permute(1, 0)


    def rot_mat_loss(self, pred, target):
        losses = {}
        losses["mse"] = self.mse(pred, target)
        return losses


    def rot_mat_loss_with_kent(self, pred, target):
        """ Compute the loss of rotation matrix with Kent distribution

        Args:
            pred (torch.Tensor): Size([batch_size, out_features])
            target (torch.Tensor): Size([batch_size, 5 * 3])
            pts(torch.Tensor): Size([num_pts, 3])
        """
        # alpha, beta, gamma, kappa, ellip = pred[:, :3], pred[:, 3:6], pred[:, 6:9], pred[:, 9:12], pred[:, 12:15]
        # pred_kd = pred[:, 15:]
        # pred_l_kd = pred_kd[:, :self.opts.num_pts]
        # pred_d_kd = pred_kd[:, self.opts.num_pts: self.opts.num_pts * 2]
        # pred_f_kd = pred_kd[:, self.opts.num_pts * 2: ]
        # pred_l_vec = normalizeVec(torch.matmul(pred_l_kd, self.gs_pts))
        # pred_d_vec = normalizeVec(torch.matmul(pred_d_kd, self.gs_pts))
        # pred_f_vec = normalizeVec(torch.matmul(pred_f_kd, self.gs_pts))
        # losses["l_loss"] = self.mse(pred_l_vec, gt[:, :3])
        # losses["d_loss"] = self.mse(pred_d_vec, gt[:, 3:6])
        # losses["f_loss"] = self.mse(pred_f_vec, gt[:, 6:])
        # losses["ortho_loss"] = ortho_loss(pred_l_vec, pred_d_vec, pred_f_vec) * self.opts.ortho_loss_weight

        # gt_R = gt.reshape(gt.shape[0], 3, 3).permute(0, 2, 1).contiguous()
        # l_kd = KentDistribution(kappa[:, 0], ellip[:, 0], alpha[:, 0], beta[:, 0], gamma[:, 0], gt_R)
        # d_kd = KentDistribution(kappa[:, 1], ellip[:, 1], alpha[:, 1], beta[:, 1], gamma[:, 1], gt_R)
        # f_kd = KentDistribution(kappa[:, 2], ellip[:, 2], alpha[:, 2], beta[:, 2], gamma[:, 2], gt_R)
        # pred_l_kd_target = l_kd.generate_probs(self.gs_pts, col_idx=0).detach()
        # pred_d_kd_target = d_kd.generate_probs(self.gs_pts, col_idx=1).detach()
        # pred_f_kd_target = f_kd.generate_probs(self.gs_pts, col_idx=2).detach()
        # losses["l_kd_loss"] = self.kl_div(pred_l_kd.log(), pred_l_kd_target) * self.opts.soft_loss_weight
        # losses["d_kd_loss"] = self.kl_div(pred_d_kd.log(), pred_d_kd_target) * self.opts.soft_loss_weight
        # losses["f_kd_loss"] = self.kl_div(pred_f_kd.log(), pred_f_kd_target) * self.opts.soft_loss_weight

        # losses["spread_regular_loss"] = torch.mean(torch.exp(- kappa)) * self.opts.shape_regular_weight
        # losses["rot_regular_loss"] = torch.mean(alpha ** 2 + beta ** 2 + gamma ** 2) * self.opts.rot_regular_weight
        alpha, beta, gamma, kappa, ellip = pred[:, :3], pred[:, 3:6], pred[:, 6:9], pred[:, 9:12], pred[:, 12:15]
        Is = torch.eye(3).repeat(pred.shape[0], 1, 1).to(self.opts.device)
        l_kenter = KentDistribution(kappa[:, 0], ellip[:, 0], alpha[:, 0], beta[:, 0], gamma[:, 0], Is)
        d_kenter = KentDistribution(kappa[:, 1], ellip[:, 1], alpha[:, 1], beta[:, 1], gamma[:, 1], Is)
        f_kenter = KentDistribution(kappa[:, 2], ellip[:, 2], alpha[:, 2], beta[:, 2], gamma[:, 2], Is)

        pred_l_vec_probs = l_kenter.generate_probs(self.gs_pts, 0)
        pred_d_vec_probs = d_kenter.generate_probs(self.gs_pts, 1)
        pred_f_vec_probs = f_kenter.generate_probs(self.gs_pts, 2)

        pred_l_vec = normalizeVec(torch.matmul(pred_l_vec_probs, self.gs_pts))
        pred_d_vec = normalizeVec(torch.matmul(pred_d_vec_probs, self.gs_pts))
        pred_f_vec = normalizeVec(torch.matmul(pred_f_vec_probs, self.gs_pts))

        # pred_R_T = torch.cat((pred_l_vec, pred_d_vec, pred_f_vec), dim=1).reshape(-1, 3, 3).contiguous()
        # V, Sigma, U_T = torch.svd(pred_R_T)
        # R_hat_T = torch.matmul(V, U_T)
        # R_hat_T_vec = R_hat_T.reshape(-1, 9).contiguous()
        # losses["mse"] = self.mse(R_hat_T_vec, target)
        
        target_l_vec = target[:,  :3]
        target_d_vec = target[:, 3:6]
        target_f_vec = target[:, 6: ]
        
        losses = {}
        losses["l_loss"] = self.mse(pred_l_vec, target_l_vec)
        losses["d_loss"] = self.mse(pred_d_vec, target_d_vec)
        losses["f_loss"] = self.mse(pred_f_vec, target_f_vec)
        losses["shape_regular_loss"] = torch.mean(torch.exp(- kappa)) * self.opts.shape_regular_weight
       # losses["rot_regular_loss"] = torch.mean(alpha ** 2 + beta ** 2 + gamma ** 2) * self.opts.rot_regular_weight

        return losses

    def rot_mat_loss_with_asg(self, pred, target):
        """ Compute the loss of rotation matrix with Anisotropic Spherical Gaussian

        Args:
            pred (torch.Tensor): Size([batch_size, out_features])
            target (torch.Tensor): Size([batch_size, 5 * 3])
            pts(torch.Tensor): Size([num_pts, 3])
        Returns: 
            losses (Dict):
        """
        lambd, mu = pred[:, :3], pred[:, 3:6]
        pred_probs = pred[:, 6: ]
        pred_l_probs = pred_probs[:, : self.opts.num_pts]
        pred_d_probs = pred_probs[:, self.opts.num_pts: self.opts.num_pts * 2]
        pred_f_probs = pred_probs[:, self.opts.num_pts * 2: ]
        pred_l_vec = normalizeVec(torch.matmul(pred_l_probs, self.gs_pts))
        pred_d_vec = normalizeVec(torch.matmul(pred_d_probs, self.gs_pts))
        pred_f_vec = normalizeVec(torch.matmul(pred_f_probs, self.gs_pts))

        target_l_vec = target[:,  :3]
        target_d_vec = target[:, 3:6]
        target_f_vec = target[:, 6: ]
        l_asg = ASGDistribution(lambd[:, 0], mu[:, 0], target_l_vec, target_d_vec, target_f_vec)
        d_asg = ASGDistribution(lambd[:, 1], mu[:, 1], target_l_vec, target_d_vec, target_f_vec)
        f_asg = ASGDistribution(lambd[:, 2], mu[:, 2], target_l_vec, target_d_vec, target_f_vec)
        target_l_probs = l_asg.generate_probs(self.gs_pts_T, col_idx=0)
        target_d_probs = d_asg.generate_probs(self.gs_pts_T, col_idx=1)
        target_f_probs = f_asg.generate_probs(self.gs_pts_T, col_idx=2)

        
        losses = {}
        losses["l_reg_loss"] = self.mse(pred_l_vec, target_l_vec)
        losses["d_reg_loss"] = self.mse(pred_d_vec, target_d_vec)
        losses["f_reg_loss"] = self.mse(pred_f_vec, target_f_vec)
        losses["l_soft_loss"] = self.kl_div(pred_l_probs.log(), target_l_probs) * self.opts.soft_loss_weight
        losses["d_soft_loss"] = self.kl_div(pred_d_probs.log(), target_d_probs) * self.opts.soft_loss_weight
        losses["f_soft_loss"] = self.kl_div(pred_f_probs.log(), target_f_probs) * self.opts.soft_loss_weight
        losses["shape_loss"] = torch.sum( 1 / lambd ** 2 + 1 / mu ** 2) * self.opts.shape_regular_weight
        losses["ortho_loss"] = ortho_loss(pred_l_vec, pred_d_vec, pred_f_vec) * self.opts.ortho_loss_weight
        return losses