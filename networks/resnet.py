import torch
from torch import nn
from torch.nn import Sigmoid, Tanh, Softmax, ReLU
from torchvision import models

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, opts, num_layers, out_features, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.opts = opts
        self.num_ch_enc = [64, 64, 128, 256, 512]
        torch.pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)
        self.expansion = 1
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
            self.expansion *= 4
        self.fc0 = nn.Linear(512 * self.expansion, out_features)
        self.softmax = Softmax(dim=-1)
        if self.opts.rot_type == "rot_mat" and self.opts.do_smooth:
            self.fc2 = nn.Linear(512 * self.expansion, 6)
            
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        x = self.encoder.avgpool(x) # x.Size([batch_size, 512*expansion, 1, 1])
        x = torch.flatten(x, 1)
        if self.opts.rot_type in ["euler", "quat", "lie"]:
            out = self.fc0(x)
        if self.opts.rot_type == "rot_mat" and not self.opts.do_smooth:
            out = self.fc0(x)
        if self.opts.rot_type == "rot_mat" and self.opts.do_smooth:
            out = self.fc0(x)
            l_dstrb = self.softmax(out[:, :self.opts.num_pts])
            d_dstrb = self.softmax(out[:, self.opts.num_pts: self.opts.num_pts * 2])
            f_dstrb = self.softmax(out[:, self.opts.num_pts * 2: ])

            ASG_paras = self.fc2(x)

            # kent_rotate = torch.tanh(self.fc22(torch.relu(self.fc21(x)))) * torch.pi # range (-pi, pi)
            # kappa  = torch.sigmoid(self.fc32(torch.relu(self.fc31(x)))) # range(0, max_kappa)
            # beta = torch.tanh(self.fc42(torch.relu(self.fc41(x)))) * 0.5 * kappa  # beta range (-0.5*kappa, 0.5*kappa)
            out = torch.cat((ASG_paras, l_dstrb, d_dstrb, f_dstrb), dim=1)
        
        return out
