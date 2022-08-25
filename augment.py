import cv2
import random
import numpy as np
import torch
import torchvision

def random_crop(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]
    out = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)
    return out

def random_crop_black(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def random_crop_white(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0+255
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out


def random_cutout(img, patches, size):
    h = img.shape[0]
    w = img.shape[1]

    for i in range(patches):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)

        img[y1: y2, x1: x2, :] = 0

    return img


def random_gamma(image, gammas=[0.5,0.7,1.0,1.5,2.0]):
    add_gamma = np.random.random()
    if add_gamma < 0.66:
        return image

    gamma = gammas[np.random.randint(5)]
    invGamma = 1.0 / gamma

    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    image=cv2.LUT(image, table)

    return image


def random_zoom(image, low=0.8, high=1.2):
    h = image.shape[0]
    w = image.shape[1]
    h_scale, w_scale = np.random.uniform(low=low, high=high, size=2)
    dx = (w - w * w_scale) / 2
    dy = (h - h * h_scale) / 2
    M = np.array([[w_scale, 0., dx],
                  [0., h_scale, dy]])
    image = cv2.warpAffine(src=image, M=M, dsize=(h, w),  borderMode=cv2.BORDER_REPLICATE)
    return image

def random_noise(img):
    add_noise = random.randint(0, 4)
    if add_noise == 4:
        noise_types = ["gauss", "s&p", "poisson"]
        noise_idx = random.randint(0, 2)
        noise_type = noise_types[noise_idx]

        # img = np.array(img)

        if noise_type == "gauss":
            row, col, ch = img.shape
            mean = 0
            # var = 0.1
            # sigma = var ** 0.5
            sigma = 0.31622776601683794
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            img = img + gauss

        elif noise_type == "s&p":
            row, col, ch = img.shape
            s_vs_p = 0.5
            amount = 0.004
            # Salt mode
            num_salt = np.ceil(amount * (img.size / ch) * s_vs_p)
            coords = [
                np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[0:2]
            ]
            img[tuple(coords)] = 255
            # Pepper mode
            num_pepper = np.ceil(amount * (img.size / ch) * (1.0 - s_vs_p))
            coords = [
                np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[0:2]
            ]
            img[tuple(coords)] = 0

        elif noise_type == "poisson":
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            img = np.random.poisson(img * vals) / float(vals)

        # img = Image.fromarray(img.astype("uint8"))
        img = img.astype("uint8")

    return img


def augment_data(image):
    rand_r = np.random.random()
    if rand_r < 0.25:
        dn = np.random.randint(60, size=1)[0] + 1
        image = random_crop(image,dn)
    elif rand_r >= 0.25 and rand_r < 0.5:
        dn = np.random.randint(60,size=1)[0] + 1
        image = random_crop_black(image,dn)
    elif rand_r >= 0.5 and rand_r < 0.75:
        dn = np.random.randint(60,size=1)[0] + 1
        image = random_crop_white(image,dn)
    else:
        image = random_cutout(image, 2, 30)

    if np.random.random() > 0.3:
        image = random_zoom(image)

    image = random_gamma(image)
    # image = random_noise(image)
    brightness = (0.8, 1.2)
    contrast = (0.8, 1.2)
    saturation = (0.8, 1.2)
    hue = (-0.1, 0.1)
    image = np.transpose(image, axes=(2, 0, 1)) # (H, W, C) -> (C, H, W)
    image = torch.from_numpy(image) / 255. # to range [0, 1]
    jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
    image = jitter(image)
    # image.shape: (3, 224, 224)

    # image = image.numpy()
    # image = np.transpose(image, axes=(1, 2, 0))
    return image