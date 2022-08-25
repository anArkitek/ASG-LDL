# -*- coding:utf-8 -*-
"""
    utils script
"""
import os
import cv2
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import angle
import torch
import numpy as np
import matplotlib
#matplotlib.use("Qt4Agg")
import math
import matplotlib.pyplot as plt
from math import cos, sin
from mpl_toolkits.mplot3d.axes3d import Axes3D
#from rotation import Rotation as R
from scipy.linalg import norm
#from loss import hellinger
import math
import torch.nn.functional as F


#Gaussian Spherical label smoothing
class GS_generator:
    def __init__(self, num_pts):
        self.num_pts = num_pts
    
    def generate_pts(self):
        #number of points of interpolation
        n = self.num_pts

        Ratio = (1 + 5**0.5)/2
        i = np.arange(0,n)
        theta = 2 * np.pi * i / Ratio
        phi = np.arccos(1 - 2*(i+0.5)/n)

        #print(theta.shape, phi.shape)

        xs, ys, zs = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)

        #GS_pts
        GS_pts = np.concatenate((xs.reshape((-1,1)), ys.reshape((-1,1)), zs.reshape((-1,1))), axis=1)

        return xs, ys, zs, GS_pts


def mkdir(dir_path):
    """
    build directory
    :param dir_path:
    :return:
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def the300w_lp_bbox(txt_path):
    """ read bounding box from 300W-LP dataset
    """
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()

    return list(map(int, lines[1].split(',')))


def norm_vector(v):
    """
    normalization vector
    :param v: vector
    :return:
    """
    vector_len = v.norm(dim=-1)
    v = v / vector_len.unsqueeze(dim=-1)

    return v


#_SQRT2 = np.sqrt(2, dtype='float32')     # sqrt(2) with default precision np.float64
#_SQRT2 = torch.tensor(_SQRT2).cuda(0)

def hellinger(p, q):
#    #_SQRT2 = np.sqrt(2)
     return torch.norm(torch.sqrt(p) - torch.sqrt(q), dim=1) / math.sqrt(2)
#    #return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def vector_cos(u, v):
    """
    compute cos value between two vectors
    :param u:
    :param v:
    :return:
    """
    assert u.shape == v.shape, 'shape of two vectors should be same'
    cos_value = torch.sum(u * v, dim=1) / torch.sqrt(torch.sum(u ** 2, dim=1) * torch.sum(v ** 2, dim=1))
    cos_value = torch.clamp(cos_value, min=float(-1.0+10.0**(-4)),max=float(1.0-10.0**(-4)))
    return cos_value


def load_filtered_stat_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def softmax(input):
    """
    implementation of softmax with numpy
    :param input:
    :return:
    """
    input = input - np.max(input)
    input_exp = np.exp(input)
    input_exp_sum = np.sum(input_exp)

    return input_exp / input_exp_sum + (10 ** -6)


def draw_bbox(img, bbox):
    """
    draw face bounding box
    :param img:np.ndarray(H,W,C)
    :param bbox: list[x1,y1,x2,y2]
    :return:
    """
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255))
    return img


def draw_front(img, x, y, width, tdx=None, tdy=None, size=100, color=(0, 255, 0)):
    """
    draw face orientation vector in image
    :param img: face image
    :param x: x of face orientation vector,integer
    :param y: y of face orientation vector,integer
    :param tdx: x of start point,integer
    :param tdy: y of start point,integer
    :param size: length of face orientation vector
    :param color:
    :return:
    """

    size = width
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    x2 = tdx + size * x
    y2 = tdy - size * y
    y2 = tdy + size * y
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), color, tipLength=0.2, thickness=5)
    return img


def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=60):
    """
    :param img: original images.[np.ndarray]
    :param yaw:
    :param pitch:
    :param roll:
    :param tdx: x-axis for start point
    :param tdy: y-axis for start point
    :param size: line size
    :return:
    """
    pitch = pitch
    yaw = -yaw
    roll = roll

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 255, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)

    return img

def remove_distortion(img):
    DIM = (960, 720)
    h, w, c = img.shape

    wt = 960
    ht = 720
    border = [int((w-wt)/2), int((h-ht)/2), int(w - (w-wt)/2), int(h - (h-ht)/2)]
    
    K = np.array([[424.57214422800234, 0.0, 464.31976295418264], 
              [0.0, 424.9291201199454, 362.78142329711255], 
              [0.0, 0.0, 1.0]])

    D = np.array([[-0.02364380260312553], [0.03507545568167827], [-0.059312268236712096], [0.03479088452999722]])
    
    crop_img = img[border[1]:border[3],border[0]:border[2],:]
    #print(crop_img.shape)
    #cv2.imshow("cropped", crop_img)  # uncomment this line if error messages show up.
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(crop_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img


def draw_3d_coor(v1, v2, v3, img, ax):

    zero = np.zeros(3) 
    # plot test data
    x, y, z = zip(zero, v1)
    plt.plot(y, x, z, '-r', linewidth=3)
    
    x, y, z = zip(zero, v2)
    plt.plot(y, x, z, '-g', linewidth=3)

    x, y, z = zip(zero, v3)
    plt.plot(y, x, z, '-b', linewidth=3)
    
    

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.draw()
    #print("draw")
    plt.pause(0.0000001)
    plt.cla()

def get_label_from_txt(txt_path):
    with open(txt_path, 'r') as fr:
        line = fr.read().splitlines()
    line = line[0].split(' ')
    label = [float(i) for i in line]

    return label


def get_front_vector(txt_path):
    with open(txt_path, 'r') as fr:
        line = fr.read().splitlines()
    line = line[0].split(',')
    label = [float(i) for i in line]

    return label


def get_info_from_txt(txt_path):
    with open(txt_path, 'r') as fr:
        lines = fr.read().splitlines()
    
    line = lines[0].split(' ')
    label1 = [float(i) for i in line]

    line = lines[1].split(' ')
    label2 = [float(i) for i in line]

    line = lines[2].split(' ')
    label3 = [float(i) for i in line]

    line = lines[3].split(' ')
    label4 = [float(i) for i in line]

    return [label1,label2,label3,label4]


def normalizeQuat(quat):
    """
    Args:
        quat: Size(batch_size, 4)
    """
    normQuat = quat / torch.sqrt(torch.sum(quat ** 2, dim=-1, keepdim=True))
    return normQuat

def normalizeVec(vec):
    """
    Args:
        vec: Size(batch_size, 3)
    """
    if isinstance(vec, torch.Tensor):
        normVec = vec / torch.sqrt(torch.sum(vec ** 2, dim = -1, keepdim=True))
    elif isinstance(vec, np.ndarray):
        normVec = vec / np.sqrt(np.sum(vec ** 2, axis= -1, keepdims=True))
    return normVec

def isRotationMatrix(R):
    tag = False
    if isinstance(R, np.ndarray):
        I = np.identity(R.shape[0])
        if np.all(np.matmul(R, R.T) - I < 1e-6) and np.linalg.det(R) - 1 < 1e-6: 
            tag = True
    elif isinstance(R, torch.Tensor):
        I = torch.eye(R.shape[0]).to(R.device)
        if torch.all(torch.matmul(R, R.T) - I < 1e-6) and torch.linalg.det(R) - 1 < 1e-6:
            tag = True
            
    return tag  

def the300w_lp_quat2euler(quat, degrees=True):
    from scipy.spatial.transform import Rotation as R
    #assert np.abs(np.sum(np.array(quat) ** 2) - 1.) < 1e-6
    if isinstance(quat, torch.Tensor):
        quat = quat.cpu().numpy()
    r = R.from_quat(quat)
    pyr = r.as_euler('XYZ', degrees=degrees)
    pyr[:, 1] *= -1
    pyr = torch.from_numpy(pyr)
    return pyr[:,0], pyr[:,1], pyr[:,2]


def the300w_lp_Euler2R(rx, ry, rz, degrees):
    '''
    rx: pitch in degrees
    ry: yaw in degrees
    rz: roll in degrees
    '''
    if degrees:
        rx = rx / 180. * 3.1415927410125732
        ry = ry / 180. * 3.1415927410125732
        rz = rz / 180. * 3.1415927410125732
    if isinstance(rx, torch.Tensor):
        ry *= -1
        R_x = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, torch.cos(rx), -torch.sin(rx)],
                            [0.0, torch.sin(rx), torch.cos(rx)]])

        R_y = torch.tensor([[torch.cos(ry), 0.0, torch.sin(ry)],
                            [0.0, 1.0, 0.0],
                            [-torch.sin(ry), 0.0, torch.cos(ry)]])

        R_z = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0.0],
                            [torch.sin(rz), torch.cos(rz), 0.0],
                            [0.0, 0.0, 1.0]])
        R = torch.matmul(R_x, torch.matmul(R_y, R_z))
    else:
        ry *= -1
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx)],
                        [0.0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                        [np.sin(rz), np.cos(rz), 0.0],
                        [0.0, 0.0, 1.0]])
                        
        R = R_x @ R_y @ R_z
    return R


def the300w_lp_R2Euler(R, degrees=True):
        yaw = np.arcsin(R[0, 2])
        roll = np.arctan2(-R[0, 1], R[0, 0])
        pitch = np.arctan2(-R[1, 2], R[2, 2])
        yaw *= -1
        pyr_rad = np.array([pitch, yaw, roll])
        if degrees == True:
            pyr_deg = pyr_rad * 180. / np.pi
            return pyr_deg
        return pyr_rad


def the300w_lp_R2axisAngle(R, degrees=True):
    if isinstance(R, np.ndarray):
        trR = R[0, 0] + R[1, 1] + R[2, 2]
        angle_rad = np.arccos((trR - 1) / 2)
        angle_deg = angle_rad * 180. / np.pi

        m01 = R[0, 1]
        m10 = R[1, 0]
        m12 = R[1, 2]
        m21 = R[2, 1]
        m02 = R[0, 2]
        m20 = R[2, 0]
        x = (m21 - m12) / np.sqrt((m21 - m12) ** 2 + (m02 - m20) ** 2+ (m10 - m01) ** 2)
        y = (m02 - m20) / np.sqrt((m21 - m12) ** 2 + (m02 - m20) ** 2+ (m10 - m01) ** 2)
        z = (m10 - m01) / np.sqrt((m21 - m12) ** 2 + (m02 - m20) ** 2+ (m10 - m01) ** 2)
        axis = np.array([[x], [y], [z]])
        assert np.all(np.abs(np.matmul(R, axis) - axis) < 1e-6)
        if degrees:
            return axis, angle_deg
        else:
            return axis, angle_rad
    if isinstance(R, torch.Tensor):
        device = R.get_device()
        trR = R[0, 0] + R[1, 1] + R[2, 2]
        angle_rad = torch.arccos((trR - 1) / 2)
        torch.pi = torch.acos(torch.zeros(1)).item()
        angle_deg = angle_rad * 180. / torch.pi

        m01 = R[0, 1]
        m10 = R[1, 0]
        m12 = R[1, 2]
        m21 = R[2, 1]
        m02 = R[0, 2]
        m20 = R[2, 0]
        x = (m21 - m12) / torch.sqrt((m21 - m12) ** 2 + (m02 - m20) ** 2+ (m10 - m01) ** 2)
        y = (m02 - m20) / torch.sqrt((m21 - m12) ** 2 + (m02 - m20) ** 2+ (m10 - m01) ** 2)
        z = (m10 - m01) / torch.sqrt((m21 - m12) ** 2 + (m02 - m20) ** 2+ (m10 - m01) ** 2)
        axis = torch.tensor([[x], [y], [z]], device=device)
        assert torch.all(torch.abs(torch.matmul(R, axis) - axis) < 1e-6)
        if degrees:
            return axis, angle_deg
        else:
            return axis, angle_rad
    

def the300w_lp_axisAngle2R(axis, angle, degrees=True):
    axis = axis.reshape(3, 1)
    if isinstance(axis, torch.Tensor):
        antiSymMat = torch.tensor(
            [
                [          0, -axis[2, 0],  axis[1, 0]],
                [ axis[2, 0],           0, -axis[0, 0]],
                [-axis[1, 0],  axis[0, 0],           0]
            ]
        )
        if degrees:
            torch.pi = torch.acos(torch.zeros(1)).item()
            angle_rad = angle * torch.pi / 180.
        else:
            angle_rad = angle
        R = torch.cos(angle_rad) * torch.eye(3) + (1 - torch.cos(angle_rad)) * torch.matmul(axis, axis.T) + torch.sin(angle_rad) * antiSymMat 
        return R
    if isinstance(axis, np.ndarray):
        antiSymMat = np.array(
            [
                [          0, -axis[2, 0],  axis[1, 0]],
                [ axis[2, 0],           0, -axis[0, 0]],
                [-axis[1, 0],  axis[0, 0],           0]
            ]
        )
        if degrees:
            angle_rad = angle  * np.pi / 180.
        else:
            angle_rad = angle
        R = np.cos(angle_rad) * np.eye(3) + (1 - np.cos(angle_rad)) * np.matmul(axis, axis.T) + np.sin(angle_rad) * antiSymMat 
        return R


def euler_from_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()
    line = lines[0].split(',')
    eulers = [float(i) for i in line]
    return eulers


def quat_from_file(txt_path):
    with open(txt_path) as f:
        lines = f.read().splitlines()
    line = lines[5].split(",")
    quats = [float(i) for i in line]

    return quats


def lie_from_file(txt_path):
    with open(txt_path) as f:
        lines = f.read().splitlines()
    line = lines[6].split(",")
    print(txt_path)
    print(line)
    lies = [float(i) for i in line]
    return lies


def vector_from_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()
    line1 = lines[2].split(',')[:3]
    left_vector = [float(i) for i in line1]

    line2 = lines[3].split(',')[:3]
    down_vector = [float(i) for i in line2]

    line3 = lines[4].split(',')[:3]
    front_vector = [float(i) for i in line3]

    return left_vector, down_vector, front_vector


def Bbox300W(txt_path):
    with open(txt_path, 'r') as f:
         lines = f.read().splitlines()

    return lines[1].split(',')
 

def degress_score(cos_value, error_degrees):
    """
    get collect score
    :param cos_value: cos value of two vectors
    :param error_degrees: degrees error limit value,integer
    :return:
    """
    score = torch.tensor([1.0 if i > cos(error_degrees * np.pi / 180) else 0.0 for i in cos_value])
    return score


def get_transform(rx, ry, rz):
    '''
    Args:
        rx, ry, rz: rotation along x, y, z axes (in radians)
    Returns:
        transform: 3*3 rotation matrix
    '''
    R_x = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(rx), np.sin(rx)],
                    [0.0, -np.sin(rx), np.cos(rx)]])

    R_y = np.array([[np.cos(ry), 0.0, -np.sin(ry)],
                    [0.0, 1.0, 0.0],
                    [np.sin(ry), 0.0, np.cos(ry)]])

    R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                    [np.sin(rz), np.cos(rz), 0.0],
                    [0.0, 0.0, 1.0]])
    
    # x = np.array([1.0, 0.0, 0.0])
    # y = np.array([0.0, 1.0, 0.0])
    # z = np.array([0.0, 0.0, 1.0])
    # n = np.array([1.0, 1.0, 0.0])
    return R_z @ R_y @ R_x


def get_attention_vector(quat):
    """
    get face orientation vector from quaternion
    :param quat:
    :return:
    """
    dcm = R.quat2dcm(quat)
    v_front = np.mat([[0], [0], [1]])
    v_front = dcm * v_front
    v_front = np.array(v_front).reshape(3)

    # v_top = np.mat([[0], [1], [0]])
    # v_top = dcm * v_top
    # v_top = np.array(v_top).reshape(3)

    # return np.hstack([v_front, v_top])
    return v_front


def get_vectors(info):

    # camera (x, y, z)
    # We don't use them for now
    xc_val = float(info[0][0])
    yc_val = float(info[0][1])
    zc_val = float(info[0][2])

    # camera (roll, pitch, yaw)
    pitchc_val = float(info[1][0])
    yawc_val = float(info[1][1])
    rollc_val = float(info[1][2])

    # --------------------------------

    # object (x, y, z)
    xo_val = float(info[2][0])
    yo_val = float(info[2][1])
    zo_val = float(info[2][2])

    # object (roll, pitch, yaw)
    pitcho_val = float(info[3][0])
    yawo_val = float(info[3][1])
    rollo_val = float(info[3][2])

    # [roll, pitch, yaw] of cameras& objects in the world
    rpy_cw = np.array([rollc_val, pitchc_val, yawc_val])
    rpy_ow = np.array([rollo_val, pitcho_val, yawo_val])

    rpy_cw = [math.radians(x) for x in rpy_cw]
    rpy_ow = [math.radians(x) for x in rpy_ow]

    # get the transformations
    T_wo = get_transform(rpy_ow[0], rpy_ow[1], rpy_ow[2])
    T_wc = get_transform(rpy_cw[0], rpy_cw[1], rpy_cw[2])

    vec_ocx = np.linalg.inv(T_wc) @ T_wo @ np.array([1.0, 0.0, 0.0])
    vec_ocy = np.linalg.inv(T_wc) @ T_wo @ np.array([0.0, 1.0, 0.0])
    vec_ocz = np.linalg.inv(T_wc) @ T_wo @ np.array([0.0, 0.0, 1.0])

    return vec_ocx, vec_ocy, vec_ocz


def rotationMatrixToRollPitchYaw(R) :
    """
    Convert 3*3 rotation matrix to roll pitch yaw in radians
    Args:
        R: 3*3 rotation matrix
    Returns:
        [roll, pitch, yaw] in degrees
    """
    # assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z]) * -1 * 180.0 / np.pi


def smooth_one_hot(true_labels, classes, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    #true_labels = torch.LongTensor([true_labels])
    #true_labels = true_labels.type_as(torch.FloatTensor())
    confidence = 1.0 - smoothing
    #print(true_labels.size(0))
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape) #device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist[0]


def get_soft_label(cls_label, num_classes, slop = 1, dis_coef = 0.5, coef = 1):
    """
    compute soft label replace one-hot label
    :param cls_label:ground truth class label
    :param num_classes:mount of classes
    :return:
    """

    # def metrix_fun(a, b):
    #     torch.IntTensor(a)
    #     torch.IntTensor(b)
    #     metrix_dis = (a - b) ** 2
    #     return metrix_dis
    def metrix_fun(a, b):
        a = a.type_as(torch.FloatTensor())
        b = b.type_as(torch.FloatTensor())
        metrix_dis = torch.abs(a - b) ** dis_coef
        #metrix_dis = (a * slop - b * slop) ** dis_coef
        #print(metrix_dis)
        return metrix_dis 

    def exp(x):
        x = x.type_as(torch.FloatTensor())
        return torch.exp(x)

    rt = torch.IntTensor([cls_label]*num_classes)  # must be torch.IntTensor or torch.LongTensor
    rk = torch.IntTensor([idx for idx in range(1, num_classes + 1, 1)])
    metrix_vector = exp(-metrix_fun(rt, rk)) * coef

    return metrix_vector / torch.sum(metrix_vector)

def computeLoss_quat(a, b, c, d, logits, reg_criterion):
    q1, q2, q3, q4 = logits

    q1, q2, q3, q4 = normalizeQuat(torch.cat((q1,q2,q3,q4), dim=-1))

    q1_loss = reg_criterion(a, q1)
    q2_loss = reg_criterion(b, q2)
    q3_loss = reg_criterion(c, q3)
    q4_loss = reg_criterion(d, q4)

    loss = [q1_loss, q2_loss, q3_loss, q4_loss]

    q1_error = torch.mean(torch.abs(a-q1))
    q2_error = torch.mean(torch.abs(b-q2))
    q3_error = torch.mean(torch.abs(c-q3))
    q4_error = torch.mean(torch.abs(d-q4))

    return loss, q1_error, q2_error, q3_error, q4_error


def computeLoss_eulerSoft(pitch, yaw, roll, soft_pitch, soft_yaw, soft_roll, logits, reg_criterion, cls_criterion, softmax, idx_tensor):
    v1, v2, v3 = logits
    #print(soft_pitch.shape)
    v1_cls_loss = cls_criterion(v1, soft_pitch[:,0])
    v2_cls_loss = cls_criterion(v2, soft_yaw[:,0])
    v3_cls_loss = cls_criterion(v3, soft_roll[:,0])

    pitch_predicted = softmax(v1)
    yaw_predicted = softmax(v2)
    roll_predicted = softmax(v3)

    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=-1) * 3 - 99
    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=-1) * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, dim=-1) * 3 - 99

    #print(pitch_predicted.shape)

    v1_reg_loss = reg_criterion(pitch_predicted, pitch[:,0])
    v2_reg_loss = reg_criterion(yaw_predicted, yaw[:,0])
    v3_reg_loss = reg_criterion(roll_predicted, roll[:,0])

    v1_loss = v1_cls_loss + v1_reg_loss
    v2_loss = v2_cls_loss + v2_reg_loss
    v3_loss = v3_cls_loss + v3_reg_loss

    loss = [v1_loss, v2_loss, v3_loss]

    v1_error = torch.mean(torch.abs(pitch-pitch_predicted))
    v2_error = torch.mean(torch.abs(yaw-yaw_predicted))
    v3_error = torch.mean(torch.abs(roll-roll_predicted))

    return loss, v1_error, v2_error, v3_error




def computeLoss_euler(pitch, yaw, roll, logits, reg_criterion):
    v1, v2, v3 = logits
    v1_loss = reg_criterion(pitch, v1)
    v2_loss = reg_criterion(yaw, v2)
    v3_loss = reg_criterion(roll, v3)

    loss = [v1_loss, v2_loss, v3_loss]

    v1_error = torch.mean(torch.abs(pitch-v1))
    v2_error = torch.mean(torch.abs(yaw-v2))
    v3_error = torch.mean(torch.abs(roll-v3))

    return loss, v1_error, v2_error, v3_error


def computeLoss_vector(reg_v1, reg_v2, reg_v3, logits, reg_criterion):
    pred_v1, pred_v2, pred_v3 = logits

    loss_v1 = reg_criterion(pred_v1, reg_v1)
    loss_v2 = reg_criterion(pred_v2, reg_v2)
    loss_v3 = reg_criterion(pred_v3, reg_v3)

    loss = [loss_v1, loss_v2, loss_v3]

    # get predicted vector errors
    cos_value_v1 = vector_cos(pred_v1, reg_v1)
    degree_error_v1 = torch.mean(torch.acos(cos_value_v1) * 180 / np.pi)

    cos_value_v2 = vector_cos(pred_v2, reg_v2)
    degree_error_v2 = torch.mean(torch.acos(cos_value_v2) * 180 / np.pi)

    cos_value_v3 = vector_cos(pred_v3, reg_v3)
    degree_error_v3 = torch.mean(torch.acos(cos_value_v3) * 180 / np.pi)

    return loss, degree_error_v1, degree_error_v2, degree_error_v3




def computeLoss_GS(cls_label_v1, cls_label_v2, cls_label_v3,
                    vector_label_v1, vector_label_v2, vector_label_v3,
                    logits, EU_loss, istopk, topk, Softmax, Sigmoid, cls_criterion, reg_criterion, l_targets, d_targets, f_targets, params, gpu, cls_coef=1):
    
    num_classes, alpha, beta, cls_type, reg_type, add_ortho = params

    # get x,y,z continue label
    x_reg_label_v1 = vector_label_v1[:, 0]
    y_reg_label_v1 = vector_label_v1[:, 1]
    z_reg_label_v1 = vector_label_v1[:, 2]

    x_reg_label_v2 = vector_label_v2[:, 0]
    y_reg_label_v2 = vector_label_v2[:, 1]
    z_reg_label_v2 = vector_label_v2[:, 2]

    x_reg_label_v3 = vector_label_v3[:, 0]
    y_reg_label_v3 = vector_label_v3[:, 1]
    z_reg_label_v3 = vector_label_v3[:, 2]

    v1, v2, v3 = logits

    if cls_type == "KLDiv":
        cls_loss_v1 = cls_criterion(F.log_softmax(v1,dim=-1), F.softmax(cls_label_v1,dim=-1))

        cls_loss_v2 = cls_criterion(F.log_softmax(v2,dim=-1), F.softmax(cls_label_v2,dim=-1))

        cls_loss_v3 = cls_criterion(F.log_softmax(v3,dim=-1), F.softmax(cls_label_v3,dim=-1))

    elif cls_type == "CrossEntropy":
        cls_loss_v1 = cls_criterion(v1, cls_label_v1.long())
        cls_loss_v2 = cls_criterion(v2, cls_label_v2.long())
        cls_loss_v3 = cls_criterion(v3, cls_label_v3.long())
    
    if EU_loss:
        cls_loss_v1 += reg_criterion(F.softmax(v1, dim=-1), F.softmax(cls_label_v1, dim=-1))
        cls_loss_v2 += reg_criterion(F.softmax(v2, dim=-1), F.softmax(cls_label_v2, dim=-1))
        cls_loss_v3 += reg_criterion(F.softmax(v3, dim=-1), F.softmax(cls_label_v3, dim=-1))
    

    # get prediction vector(get continue value from classify result)
    x_reg_pred_v1, y_reg_pred_v1, z_reg_pred_v1, vector_pred_v1 = classify2vector_GS(v1, Softmax, istopk, topk, num_classes, gpu)
    x_reg_pred_v2, y_reg_pred_v2, z_reg_pred_v2, vector_pred_v2 = classify2vector_GS(v2, Softmax, istopk, topk, num_classes, gpu)
    x_reg_pred_v3, y_reg_pred_v3, z_reg_pred_v3, vector_pred_v3 = classify2vector_GS(v3, Softmax, istopk, topk, num_classes, gpu)


    # Regression loss
    if reg_type == "value":
        x_reg_loss_v1 = reg_criterion(x_reg_pred_v1, x_reg_label_v1)
        y_reg_loss_v1 = reg_criterion(y_reg_pred_v1, y_reg_label_v1)
        z_reg_loss_v1 = reg_criterion(z_reg_pred_v1, z_reg_label_v1)

        x_reg_loss_v2 = reg_criterion(x_reg_pred_v2, x_reg_label_v2)
        y_reg_loss_v2 = reg_criterion(y_reg_pred_v2, y_reg_label_v2)
        z_reg_loss_v2 = reg_criterion(z_reg_pred_v2, z_reg_label_v2)

        x_reg_loss_v3 = reg_criterion(x_reg_pred_v3, x_reg_label_v3)
        y_reg_loss_v3 = reg_criterion(y_reg_pred_v3, y_reg_label_v3)
        z_reg_loss_v3 = reg_criterion(z_reg_pred_v3, z_reg_label_v3)
    
    if add_ortho:
        cp_v3 = torch.cross(vector_pred_v1, vector_pred_v2, dim=-1)
        cp_v2 = -1.0 * torch.cross(vector_pred_v1, vector_pred_v3, dim=-1)
        cp_v1 = torch.cross(vector_pred_v2, vector_pred_v3, dim=-1)

        ortho_loss_v1 = reg_criterion(vector_pred_v1, cp_v1)
        ortho_loss_v2 = reg_criterion(vector_pred_v2, cp_v2)
        ortho_loss_v3 = reg_criterion(vector_pred_v3, cp_v3)

    #-----------cls+reg+ortho loss-------------------------
    loss_v1 = cls_coef * cls_loss_v1 + alpha * (x_reg_loss_v1 + y_reg_loss_v1 + z_reg_loss_v1) + beta * ortho_loss_v1
    loss_v2 = cls_coef * cls_loss_v2 + alpha * (x_reg_loss_v2 + y_reg_loss_v2 + z_reg_loss_v2) + beta * ortho_loss_v2
    loss_v3 = cls_coef * cls_loss_v3 + alpha * (x_reg_loss_v3 + y_reg_loss_v3 + z_reg_loss_v3) + beta * ortho_loss_v3


    loss = [loss_v1, loss_v2, loss_v3]

    reg_loss_v1 = x_reg_loss_v1 + y_reg_loss_v1 + z_reg_loss_v1
    reg_loss_v2 = x_reg_loss_v2 + y_reg_loss_v2 + z_reg_loss_v2
    reg_loss_v3 = x_reg_loss_v3 + y_reg_loss_v3 + z_reg_loss_v3

    # get predicted vector errors
    cos_value_v1 = vector_cos(vector_pred_v1, vector_label_v1)
    degree_error_v1 = torch.mean(torch.acos(cos_value_v1) * 180 / np.pi)

    cos_value_v2 = vector_cos(vector_pred_v2, vector_label_v2)
    degree_error_v2 = torch.mean(torch.acos(cos_value_v2) * 180 / np.pi)

    cos_value_v3 = vector_cos(vector_pred_v3, vector_label_v3)
    degree_error_v3 = torch.mean(torch.acos(cos_value_v3) * 180 / np.pi)

    return loss, degree_error_v1, degree_error_v2, degree_error_v3, cls_loss_v1, cls_loss_v2, cls_loss_v3, reg_loss_v1, reg_loss_v2, reg_loss_v3



def computeLoss(cls_label_v1, cls_label_v2, cls_label_v3,
                vector_label_v1, vector_label_v2, vector_label_v3, 
                logits, softmax, sigmoid, cls_criterion, reg_criterion, l_targets, d_targets, f_targets, params, cls_coef=1):

    num_classes, alpha, beta, cls_type, reg_type, add_ortho = params

    # get x,y,z cls label
    x_cls_label_v1 = cls_label_v1[:, 0]
    y_cls_label_v1 = cls_label_v1[:, 1]
    z_cls_label_v1 = cls_label_v1[:, 2]

    x_cls_label_v2 = cls_label_v2[:, 0]
    y_cls_label_v2 = cls_label_v2[:, 1]
    z_cls_label_v2 = cls_label_v2[:, 2]

    x_cls_label_v3 = cls_label_v3[:, 0]
    y_cls_label_v3 = cls_label_v3[:, 1]
    z_cls_label_v3 = cls_label_v3[:, 2]

    # get x,y,z continue label
    x_reg_label_v1 = vector_label_v1[:, 0]
    y_reg_label_v1 = vector_label_v1[:, 1]
    z_reg_label_v1 = vector_label_v1[:, 2]

    x_reg_label_v2 = vector_label_v2[:, 0]
    y_reg_label_v2 = vector_label_v2[:, 1]
    z_reg_label_v2 = vector_label_v2[:, 2]

    x_reg_label_v3 = vector_label_v3[:, 0]
    y_reg_label_v3 = vector_label_v3[:, 1]
    z_reg_label_v3 = vector_label_v3[:, 2]

    x_pred_v1, y_pred_v1, z_pred_v1, x_pred_v2, y_pred_v2, z_pred_v2, x_pred_v3, y_pred_v3, z_pred_v3 = logits

    # -------------------------------------------BCELoss(for classify, manually apply softmax layer)---------------------------------------------
    if cls_type == "BCE":
        assert ((cls_label_v1 >= 0.) & (cls_label_v1 <= 1.)).all()
        x_cls_loss_v1 = cls_criterion(sigmoid(x_pred_v1), x_cls_label_v1)
        y_cls_loss_v1 = cls_criterion(sigmoid(y_pred_v1), y_cls_label_v1)
        z_cls_loss_v1 = cls_criterion(sigmoid(z_pred_v1), z_cls_label_v1)

        assert ((cls_label_v2 >= 0.) & (cls_label_v2 <= 1.)).all()
        x_cls_loss_v2 = cls_criterion(sigmoid(x_pred_v2), x_cls_label_v2)
        y_cls_loss_v2 = cls_criterion(sigmoid(y_pred_v2), y_cls_label_v2)
        z_cls_loss_v2 = cls_criterion(sigmoid(z_pred_v2), z_cls_label_v2)

        assert ((cls_label_v3 >= 0.) & (cls_label_v3 <= 1.)).all()
        x_cls_loss_v3 = cls_criterion(sigmoid(x_pred_v3), x_cls_label_v3)
        y_cls_loss_v3 = cls_criterion(sigmoid(y_pred_v3), y_cls_label_v3)
        z_cls_loss_v3 = cls_criterion(sigmoid(z_pred_v3), z_cls_label_v3)

    elif cls_type == 'CrossEntropy':
        x_cls_loss_v1 = cls_criterion(x_pred_v1, l_targets[:,0])
        y_cls_loss_v1 = cls_criterion(y_pred_v1, l_targets[:,1])
        z_cls_loss_v1 = cls_criterion(z_pred_v1, l_targets[:,2])

        x_cls_loss_v2 = cls_criterion(x_pred_v2, d_targets[:,0])
        y_cls_loss_v2 = cls_criterion(y_pred_v2, d_targets[:,1])
        z_cls_loss_v2 = cls_criterion(z_pred_v2, d_targets[:,2])
        
        x_cls_loss_v3 = cls_criterion(x_pred_v3, f_targets[:,0])
        y_cls_loss_v3 = cls_criterion(y_pred_v3, f_targets[:,1])
        z_cls_loss_v3 = cls_criterion(z_pred_v3, f_targets[:,2])


    #----------------------------------------FocalLoss-----------------------------------------
    elif cls_type == 'FocalLoss':
        x_cls_loss_v1 = cls_criterion(x_pred_v1, l_targets[:,0])
        y_cls_loss_v1 = cls_criterion(y_pred_v1, l_targets[:,1])
        z_cls_loss_v1 = cls_criterion(z_pred_v1, l_targets[:,2])

        x_cls_loss_v2 = cls_criterion(x_pred_v2, d_targets[:,0])
        y_cls_loss_v2 = cls_criterion(y_pred_v2, d_targets[:,1])
        z_cls_loss_v2 = cls_criterion(z_pred_v2, d_targets[:,2])

        x_cls_loss_v3 = cls_criterion(x_pred_v3, f_targets[:,0])
        y_cls_loss_v3 = cls_criterion(y_pred_v3, f_targets[:,1])
        z_cls_loss_v3 = cls_criterion(z_pred_v3, f_targets[:,2])


    # -------------------------------------------KL Divergence Loss-------------------------------------
    elif cls_type == "KLDiv":
        x_cls_loss_v1 = cls_criterion((softmax(x_pred_v1)+10e-6).log(), x_cls_label_v1+10e-6)
        y_cls_loss_v1 = cls_criterion((softmax(y_pred_v1)+10e-6).log(), y_cls_label_v1+10e-6)
        z_cls_loss_v1 = cls_criterion((softmax(z_pred_v1)+10e-6).log(), z_cls_label_v1+10e-6)

        x_cls_loss_v2 = cls_criterion((softmax(x_pred_v2)+10e-6).log(), x_cls_label_v2+10e-6)
        y_cls_loss_v2 = cls_criterion((softmax(y_pred_v2)+10e-6).log(), y_cls_label_v2+10e-6)
        z_cls_loss_v2 = cls_criterion((softmax(z_pred_v2)+10e-6).log(), z_cls_label_v2+10e-6)

        x_cls_loss_v3 = cls_criterion((softmax(x_pred_v3)+10e-6).log(), x_cls_label_v3+10e-6)
        y_cls_loss_v3 = cls_criterion((softmax(y_pred_v3)+10e-6).log(), y_cls_label_v3+10e-6)
        z_cls_loss_v3 = cls_criterion((softmax(z_pred_v3)+10e-6).log(), z_cls_label_v3+10e-6)

    length = x_pred_v1.shape[0]


    # get prediction vector(get continue value from classify result)
    x_reg_pred_v1, y_reg_pred_v1, z_reg_pred_v1, vector_pred_v1 = classify2vector(x_pred_v1, y_pred_v1, z_pred_v1, softmax, num_classes)
    x_reg_pred_v2, y_reg_pred_v2, z_reg_pred_v2, vector_pred_v2 = classify2vector(x_pred_v2, y_pred_v2, z_pred_v2, softmax, num_classes)
    x_reg_pred_v3, y_reg_pred_v3, z_reg_pred_v3, vector_pred_v3 = classify2vector(x_pred_v3, y_pred_v3, z_pred_v3, softmax, num_classes)

    # Regression loss
    if reg_type == "value":
        x_reg_loss_v1 = reg_criterion(x_reg_pred_v1, x_reg_label_v1)
        y_reg_loss_v1 = reg_criterion(y_reg_pred_v1, y_reg_label_v1)
        z_reg_loss_v1 = reg_criterion(z_reg_pred_v1, z_reg_label_v1)

        x_reg_loss_v2 = reg_criterion(x_reg_pred_v2, x_reg_label_v2)
        y_reg_loss_v2 = reg_criterion(y_reg_pred_v2, y_reg_label_v2)
        z_reg_loss_v2 = reg_criterion(z_reg_pred_v2, z_reg_label_v2)

        x_reg_loss_v3 = reg_criterion(x_reg_pred_v3, x_reg_label_v3)
        y_reg_loss_v3 = reg_criterion(y_reg_pred_v3, y_reg_label_v3)
        z_reg_loss_v3 = reg_criterion(z_reg_pred_v3, z_reg_label_v3)

        #-----------cls+reg loss-------------------------
        loss_v1 = cls_coef * (x_cls_loss_v1 + y_cls_loss_v1 + z_cls_loss_v1) + alpha * (x_reg_loss_v1 + y_reg_loss_v1 + z_reg_loss_v1)
        loss_v2 = cls_coef * (x_cls_loss_v2 + y_cls_loss_v2 + z_cls_loss_v2) + alpha * (x_reg_loss_v2 + y_reg_loss_v2 + z_reg_loss_v2)
        loss_v3 = cls_coef * (x_cls_loss_v3 + y_cls_loss_v3 + z_cls_loss_v3) + alpha * (x_reg_loss_v3 + y_reg_loss_v3 + z_reg_loss_v3)


    #-------------------------acos loss---------------------------------
    if reg_type == 'acos':
        reg_loss_v1 = reg_criterion(torch.acos(vector_cos(vector_label_v1, vector_pred_v1)), torch.tensor(np.array([0.0]*length, dtype=np.float32)).cuda(0))
        reg_loss_v2 = reg_criterion(torch.acos(vector_cos(vector_label_v2, vector_pred_v2)), torch.tensor(np.array([0.0]*length, dtype=np.float32)).cuda(0))
        reg_loss_v3 = reg_criterion(torch.acos(vector_cos(vector_label_v3, vector_pred_v3)), torch.tensor(np.array([0.0]*length, dtype=np.float32)).cuda(0))
        
        #------------cls+reg loss-------------------
        loss_v1 = cls_coef * (x_cls_loss_v1 + y_cls_loss_v1 + z_cls_loss_v1) + alpha * reg_loss_v1
        loss_v2 = cls_coef * (x_cls_loss_v2 + y_cls_loss_v2 + z_cls_loss_v2) + alpha * reg_loss_v2
        loss_v3 = cls_coef * (x_cls_loss_v3 + y_cls_loss_v3 + z_cls_loss_v3) + alpha * reg_loss_v3

    # if add ortho loss
    if add_ortho:
        loss_ortho_12 = reg_criterion(torch.sum(vector_pred_v1 * vector_pred_v2, axis=1), torch.tensor(np.array([0.0]*length, dtype=np.float32)).cuda(0))
        loss_ortho_13 = reg_criterion(torch.sum(vector_pred_v1 * vector_pred_v3, axis=1), torch.tensor(np.array([0.0]*length, dtype=np.float32)).cuda(0))
        loss_ortho_23 = reg_criterion(torch.sum(vector_pred_v2 * vector_pred_v3, axis=1), torch.tensor(np.array([0.0]*length, dtype=np.float32)).cuda(0))

        #-----------total loss
        loss_v1 += beta * (loss_ortho_12 + loss_ortho_13)
        loss_v2 += beta * (loss_ortho_12 + loss_ortho_23)
        loss_v3 += beta * (loss_ortho_23 + loss_ortho_13)


    loss = [loss_v1, loss_v2, loss_v3]

    # get predicted vector errors
    cos_value_v1 = vector_cos(vector_pred_v1, vector_label_v1)
    degree_error_v1 = torch.mean(torch.acos(cos_value_v1) * 180 / np.pi)

    cos_value_v2 = vector_cos(vector_pred_v2, vector_label_v2)
    degree_error_v2 = torch.mean(torch.acos(cos_value_v2) * 180 / np.pi)

    cos_value_v3 = vector_cos(vector_pred_v3, vector_label_v3)
    degree_error_v3 = torch.mean(torch.acos(cos_value_v3) * 180 / np.pi)

    return loss, degree_error_v1, degree_error_v2, degree_error_v3


def classify2vector_GS(v, Softmax, istopk, topk, num_classes, gpu):
    """
    get vector from classifier results
    :param v: (batch_size, num_pts)
    :param softmax: softmax function
    :param GS_pts: (batch_size, num_pts, 3)
    :param num_classes: number of class, integer
    :return:
    """
    #placeholder for result
    output = torch.zeros(v.shape[0], 3) #(batch_size, 3)
    output = output.cuda(gpu)

    v = Softmax(v)
    
    if istopk:
        #compute topk results
        probs, idxs = torch.topk(v, topk, dim=-1)

        _, _, _, GS_pts = GS_generator(num_classes).generate_pts()
        GS_pts = torch.FloatTensor(GS_pts).cuda(gpu)
        probs = probs.cuda(gpu)
        idxs = idxs.cuda(gpu)

        for i in range(v.shape[0]):
            for j in range(topk):
                output[i] += GS_pts[idxs[i][j]] * probs[i][j]

    else:
        _, _, _, GS_pts = GS_generator(num_classes).generate_pts()
        GS_pts = torch.FloatTensor(GS_pts).cuda(gpu)
        output = torch.matmul(v, GS_pts)
    

    #output = torch.matmul(v, GS_pts)

    pred_vector = norm_vector(output)

    # split to x,y,z
    x_reg = pred_vector[:, 0]
    y_reg = pred_vector[:, 1]
    z_reg = pred_vector[:, 2]

    return x_reg, y_reg, z_reg, pred_vector


def classify2vector(x, y, z, softmax, num_classes):
    """
    get vector from classify results
    :param x: fc_x output,np.ndarray(66,)
    :param y: fc_y output,np.ndarray(66,)
    :param z: fc_z output,np.ndarray(66,)
    :param softmax: softmax function
    :param num_classes: number of classify, integer
    :return:
    """
    #idx_tensor = [idx for idx in range(num_classes)]
    #idx_tensor = np.linspace(-1, 1, num_classes)
    #idx_tensor = torch.FloatTensor(idx_tensor).cuda(1)
    idx_tensor = np.linspace(-1, 1, num_classes)
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)

    x_probability = softmax(x)
    y_probability = softmax(y)
    z_probability = softmax(z)

    #x_pred = torch.sum(x_probability * idx_tensor, dim=-1) * (198 // num_classes) - 96
    #y_pred = torch.sum(y_probability * idx_tensor, dim=-1) * (198 // num_classes) - 96
    #z_pred = torch.sum(z_probability * idx_tensor, dim=-1) * (198 // num_classes) - 96
    x_pred = torch.sum(x_probability * idx_tensor, dim=-1)
    y_pred = torch.sum(y_probability * idx_tensor, dim=-1)
    z_pred = torch.sum(z_probability * idx_tensor, dim=-1)

    pred_vector = torch.stack([x_pred, y_pred, z_pred]).transpose(1, 0)
    pred_vector = norm_vector(pred_vector)
    #print(pred_vector)

    # split to x,y,z
    x_reg = pred_vector[:, 0]
    y_reg = pred_vector[:, 1]
    z_reg = pred_vector[:, 2]

    return x_reg, y_reg, z_reg, pred_vector


def show_loss_distribute(loss_dict, analysis_dir, snapshot_name):
    """

    :param loss_dict: {'angles':[[p,y,r],[],...],'degrees':[]}
    :param analysis_dir:directory for saving image
    :param snapshot_name:model snapshot name
    :return:
    """
    #plt.switch_backend('agg')

    detail = snapshot_name

    n = len(loss_dict["img_name"])
    x = [i+1 for i in range(n)]
    front_error = loss_dict['degree_error_f']
    right_error = loss_dict['degree_error_r']
    up_error = loss_dict['degree_error_u']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    fig.suptitle('Error distribution')
    ax1.scatter(x, front_error)
    ax2.scatter(x, right_error)
    ax3.scatter(x, up_error)
    plt.show()

    #angles = np.array(loss_dict['angles']) * 180 / np.pi
    #degrees_error = np.array(loss_dict['degree_error'])

    #plt.subplots(figsize=(30, 10))
    

    # figure pitch,yaw,roll
    #for i, name in enumerate(['Pitch', 'Yaw', 'Roll']):
    #    plt.subplot(1, 3, i + 1)
    #    plt.xlim(-100, 105)
    #    plt.xticks([j for j in range(-100, 105, 20)], [j for j in range(-100, 105, 20)])
    #    plt.ylim(-100, 105)
    #    plt.yticks([j for j in range(-100, 105, 10)], [j for j in range(-100, 105, 10)])
    #    plt.scatter(angles[:, i], degrees_error, linewidths=0.2)
    #    plt.title(name + ":Loss distribution(" + detail + ")")
    #    plt.xlabel(name + ":GT")
    #    plt.ylabel(name + ":Loss(degree-error)")
    #    plt.grid()

    plt.savefig(os.path.join(analysis_dir, detail + '.png'))


def collect_score(degree_dict, save_dir):
    """

    :param save_dir:
    :return:
    """
    plt.switch_backend('agg')
    x = np.array(range(0, 181, 5))
    degree_error = degree_dict['degree_error']
    mount = np.zeros(len(x))
    for j in range(len(x)):
        mount[j] = sum(degree_error < x[j])
    y = mount / len(degree_error)
    plt.plot(x, y, c="red", label="MobileNetV2")
    plt.legend(loc='lower right', fontsize='x-small')
    plt.xlabel('degrees upper limit')
    plt.ylabel('accuracy')
    plt.xlim(0, 105)
    plt.ylim(0., 1.05)
    plt.xticks([j for j in range(0, 105, 5)], [j for j in range(0, 105, 5)])
    plt.yticks([j / 100 for j in range(0, 105, 5)], [j / 100 for j in range(0, 105, 5)])
    plt.title("accuracy under degree upper limit")
    plt.grid()
    plt.savefig(save_dir + '/collect_score.png')
