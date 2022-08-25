import pprint
import argparse


class GaussianSmoothingOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Gaussian Smoothing Options")

        # PATHS
        self.parser.add_argument("--train_dataset", type=str, choices=["300W-LP", "BIWI"])
        self.parser.add_argument("--val_dataset", type=str, choices=["300W-LP", "BIWI"])
        self.parser.add_argument("--save_path", default="ASG_saved_model/", help="path to intermediate result directory")
        self.parser.add_argument("--tensorboard_path", default="./saved_tensorboard/", help="path to saved tensorboard")
        self.parser.add_argument("--snapshot_path", help="path to pre-trained model directory")

        # MODEL
        self.parser.add_argument("--backbone", default="resnet18", choices=["resnet18"])
        self.parser.add_argument("--rot_type", default="lie", choices=["rot_mat", "quat", "lie", "euler"])
        self.parser.add_argument("--do_smooth", action='store_true', help="if set, do labeling smoothing")
        self.parser.add_argument("--ortho_loss_weight", default=0.0, type=float, help="ortho loss coef")
        self.parser.add_argument("--soft_loss_weight", default=1.0, type=float, help="distribution loss coef")
        self.parser.add_argument("--num_pts", default=600, type=int, help="number of sampled points on unit sphere")
        self.parser.add_argument("--max_kappa", default=10000.0, type=float, help="max concentration value for kent distribution")
        self.parser.add_argument("--shape_regular_weight", default=0.0, type=float, help="weight for Kent kappa and ellip regularization")
        self.parser.add_argument("--rot_regular_weight", default=0.0, type=float, help="weight for Kent rotation regularization")

        # TRAINING
        self.parser.add_argument("--batch_size", default=64, type=int, help="batch size for training")
        self.parser.add_argument("--device", default="cuda:0", choices=["cpu", "cuda:0"])
        self.parser.add_argument("--img_size", default=224, type=int) 
        self.parser.add_argument("--num_epochs", default=50, type=int)
        self.parser.add_argument("--num_workers", default=8, type=int)
        self.parser.add_argument("--learning_rate", default=1e-4, type=float)
        self.parser.add_argument("--lr_gamma", default=0.9, help="decay rate of the scheduler")
        self.parser.add_argument("--prefix", type=str, help="saved model prefix")


    def parse(self, toTerminal=True) -> argparse.Namespace:
        self.options = self.parser.parse_args()
        if toTerminal:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(vars(self.options))
        return self.options
