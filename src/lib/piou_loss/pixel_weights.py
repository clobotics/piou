from torch.nn import Module
from torch.autograd import Function
import torch.nn.functional as F
from torch.autograd import Variable
import math
import shapely
from shapely.geometry import Polygon, MultiPoint
import pixel_weights_cpu
import pixel_weights_cuda
import torch
import numpy as np
import cv2
class PiousFunction(Function):
    @staticmethod
    def forward(ctx, loc_p, loc_t, grid, k, is_hard):
        ctx.k = k
        if loc_p.is_cuda:
            ctx.grad_loc_memory = torch.zeros(
                (loc_p.size(0), 5), dtype=torch.float32).cuda()
            pious = pixel_weights_cuda.forward_cuda(
                loc_p, loc_t, grid, k, is_hard, ctx.grad_loc_memory)
        else:
            pious = pixel_weights_cpu.forward_cpu(loc_p, loc_t, grid, k)
        return pious

    @staticmethod
    def backward(ctx, grad_pious):
        grad_loc_memory = ctx.grad_loc_memory
        if grad_pious.is_cuda:
            grad_loc_p = pixel_weights_cuda.backward_cuda(
                grad_pious, grad_loc_memory)
            return grad_loc_p, None, None, None, None
        else:
            return None, None, None, None, None

class Pious(Module):
    def __init__(self, k=10, is_hard=False):
        super(Pious, self).__init__()
        self.k = k
        self.is_hard = is_hard
    def forward(self, loc_p, loc_t, grid):
        pious = PiousFunction.apply(
            loc_p, loc_t, grid, self.k, self.is_hard)
        return pious

class JaccardRFunction(Function):
    @staticmethod
    def forward(ctx, loc_p, loc_t, grid):
        if loc_p.is_cuda:
            pious = pixel_weights_cuda.overlap_r_cuda(
                loc_p, loc_t, grid)
            return pious
        else:
            assert 1==0, 'Not Support'

class JaccardR(Module):
    def forward(self, loc_p, loc_t, grid):
        pious = JaccardRFunction.apply(
            loc_p, loc_t, grid)
        return pious

def template_pixels(height, width):
    xv, yv = torch.meshgrid(
        [torch.arange(-100, width + 100), torch.arange(-100, height + 100)])
    xy = torch.stack((xv, yv), -1)
    grid_xy = xy.reshape(-1, 2).float() + 0.5

    return grid_xy


def rbox2corners(rboxes):
    w_sin = 0.5 * rboxes[:, 2:3] * torch.sin(rboxes[:, 4:5])
    w_cos = 0.5 * rboxes[:, 2:3] * torch.cos(rboxes[:, 4:5])
    h_sin = 0.5 * rboxes[:, 3:4] * torch.sin(rboxes[:, 4:5])
    h_cos = 0.5 * rboxes[:, 3:4] * torch.cos(rboxes[:, 4:5])

    cornerx0 = rboxes[:, :1] + w_cos + h_sin
    cornery0 = rboxes[:, 1:2] - w_sin + h_cos
    cornerx1 = rboxes[:, :1] - w_cos + h_sin
    cornery1 = rboxes[:, 1:2] + w_sin + h_cos
    cornerx2 = rboxes[:, :1] - w_cos - h_sin
    cornery2 = rboxes[:, 1:2] + w_sin - h_cos
    cornerx3 = rboxes[:, :1] + w_cos - h_sin
    cornery3 = rboxes[:, 1:2] - w_sin - h_cos
    corners = torch.cat([cornerx0, cornery0, cornerx1,
                         cornery1, cornerx2, cornery2, cornerx3, cornery3], -1)
    return corners

def template_w_pixels(width):
    x = torch.tensor(torch.arange(-100, width + 100))
    grid_x = x.float() + 0.5
    return grid_x


def rbox2corners_torch(loc):
    cos_w = 0.5 * loc[:, 2:3] * torch.cos(loc[:, 4:5])
    sin_w = 0.5 * loc[:, 2:3] * torch.sin(loc[:, 4:5])
    cos_h = 0.5 * loc[:, 3:4] * torch.cos(loc[:, 4:5])
    sin_h = 0.5 * loc[:, 3:4] * torch.sin(loc[:, 4:5])
    x0 = loc[:, 0:1] + cos_w + sin_h
    y0 = loc[:, 1:2] - sin_w + cos_h
    x1 = loc[:, 0:1] - cos_w + sin_h
    y1 = loc[:, 1:2] + sin_w + cos_h
    x2 = loc[:, 0:1] - cos_w - sin_h
    y2 = loc[:, 1:2] + sin_w - cos_h
    x3 = loc[:, 0:1] + cos_w - sin_h
    y3 = loc[:, 1:2] - sin_w - cos_h
    rbox = torch.cat((x0, y0, x1, y1, x2, y2, x3, y3), -1)
    return rbox

def rbox2corners_cpu(cx, cy, cwidth, cheight, angle):
    x0 = cx + 0.5 * cwidth * math.cos(angle) + 0.5 * cheight * math.sin(angle)
    y0 = cy - 0.5 * cwidth * math.sin(angle) + 0.5 * cheight * math.cos(angle)
    x1 = cx - 0.5 * cwidth * math.cos(angle) + 0.5 * cheight * math.sin(angle)
    y1 = cy + 0.5 * cwidth * math.sin(angle) + 0.5 * cheight * math.cos(angle)
    x2 = cx - 0.5 * cwidth * math.cos(angle) - 0.5 * cheight * math.sin(angle)
    y2 = cy + 0.5 * cwidth * math.sin(angle) - 0.5 * cheight * math.cos(angle)
    x3 = cx + 0.5 * cwidth * math.cos(angle) - 0.5 * cheight * math.sin(angle)
    y3 = cy - 0.5 * cwidth * math.sin(angle) - 0.5 * cheight * math.cos(angle)
    rbox = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
    return rbox

# loc --> num x dim x 5
# grid_xy --> num x dim x 2


def kernel_function(dis, k, t):
    # clamp to avoid nan
    factor = torch.clamp(-k * (dis - t), -50, 50)
    return 1.0 - 1.0 / (torch.exp(factor) + 1)

# loc --> num x dim x 5
# grid_xy --> num x dim x 2


def pixel_weights(loc, grid_xy, k):

    xx = torch.pow(loc[:, :, 0:2], 2).sum(2)
    yy = torch.pow(grid_xy, 2).sum(2)
    dis = xx + yy
    # dis - 2 * x * yT
    dis.addmm_(1, -2, loc[:, 0, 0:2], grid_xy[0].t())
    dis = dis.clamp(min=1e-9).sqrt()  # for numerical stability

    a1 = loc[:, :, -1] - torch.acos((grid_xy[:, :, 0] - loc[:, :, 0]) / dis)
    a2 = loc[:, :, -1] + torch.acos((grid_xy[:, :, 0] - loc[:, :, 0]) / dis)
    a = torch.where(loc[:, :, 1] > grid_xy[:, :, 1], a1, a2)

    dis_w = dis * torch.abs(torch.cos(a))
    dis_h = dis * torch.abs(torch.sin(a))
    # return dis_h
    pixel_weights = kernel_function(
        dis_w, k, loc[:, :, 2] / 2.) * kernel_function(dis_h, k, loc[:, :, 3] / 2.)

    return pixel_weights


def PIoU(loc_p, loc_t, grid_xy, k=10):

    num = loc_p.size(0)
    dim = grid_xy.size(0)

    loc_pp = loc_p.unsqueeze(1).expand(num, dim, 5)
    loc_tt = loc_t.unsqueeze(1).expand(num, dim, 5)
    grid_xyxy = grid_xy.unsqueeze(0).expand(num, dim, 2)

    pixel_p_weights = pixel_weights(loc_pp, grid_xyxy, k)
    pixel_t_weights = pixel_weights(loc_tt, grid_xyxy, k)

    inter_pixel_area = pixel_p_weights * pixel_t_weights
    intersection_area = torch.sum(inter_pixel_area, 1)
    union_pixel_area = pixel_p_weights + \
        pixel_t_weights - inter_pixel_area
    union_area = torch.sum(union_pixel_area, 1)
    pious = intersection_area / (union_area + 1e-9)
    return torch.sum(1 - pious), pious

def HPIoU(loc_p, loc_t, grid_xy, k=10):

    num = loc_p.size(0)
    dim = grid_xy.size(0)

    loc_pp = loc_p.unsqueeze(1).expand(num, dim, 5)
    loc_tt = loc_t.unsqueeze(1).expand(num, dim, 5)
    grid_xyxy = grid_xy.unsqueeze(0).expand(num, dim, 2)

    area_p = loc_p[:, 2] * loc_p[:, 3]
    area_t = loc_t[:, 2] * loc_t[:, 3]

    pixel_p_weights = pixel_weights(loc_pp, grid_xyxy, k)
    pixel_t_weights = pixel_weights(loc_tt, grid_xyxy, k)

    inter_pixel_area = pixel_p_weights * pixel_t_weights
    intersection_area = torch.sum(inter_pixel_area, 1)

    union_area = area_p + area_t - intersection_area
    pious = intersection_area / (union_area + 1e-9)

    return torch.sum(1 - pious), pious

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b when the angles is same: 0.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A, 5].
      box_b: (tensor) bounding boxes, Shape: [B, 5].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def get_grid(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b when the angles is same: 0.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A, 5].
      box_b: (tensor) bounding boxes, Shape: [B, 5].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    grid = torch.cat((min_xy, max_xy), -1)
    inter = torch.clamp((max_xy - min_xy), min=0)
    return grid, inter[:, :, 0] * inter[:, :, 1]

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:4] / 2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:4] / 2, 
                      boxes[:, 4:5]), 1)  # xmax, ymax

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    print('inter: ', inter)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / (union + 1e-9)

def test_jaccardr():
    jaccardr_f = JaccardR()
    for item in range(100):
        print('item: ', item)
        loc_pxy = torch.randint(10, 300, [10, 2]).float()
        loc_pwh = torch.randint(10, 200, [10, 2]).float()
        loc_pa = torch.randint(-180, 180, [10, 1]) / 180.0 * 3.141593
        loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)

        loc_txy = torch.randint(10, 300, [20, 2]).float()
        loc_twh = torch.randint(10, 200, [20, 2]).float()
        loc_ta = torch.randint(-180, 180, [20, 1]) / 180.0 * 3.141593
        loc_t = torch.cat((loc_txy, loc_twh, loc_ta), -1)
        loc_p = loc_p.float()
        loc_t = loc_t.float()
        corners_p = rbox2corners_torch(loc_p)
        corners_t = rbox2corners_torch(loc_t)
        if torch.cuda.is_available():
            print('------------pixel_weights_gpu test on gpu------------')
            corners_p = corners_p.cuda()
            corners_t = corners_t.cuda()
            # print('corners_p: ', corners_p.shape)
            # print('corners_t: ', corners_t.shape)
            jaccard_r_gpu = jaccardr_f(corners_p, corners_t)
            # print('jaccard_r_gpu: ', jaccard_r_gpu[jaccard_r_gpu > 0.001])
        else:
            print('You device have not a GPU')

        print('..................polygon................')
        loc_p = loc_p.cpu().numpy()
        loc_t = loc_t.cpu().numpy()
        pious = []
        for i in range(loc_p.shape[0]):
            for j in range(loc_t.shape[0]):
                corners_p = rbox2corners_cpu(loc_p[i][0], loc_p[i][1], loc_p[i]
                                        [2], loc_p[i][3], loc_p[i][4])
                corners_t = rbox2corners_cpu(loc_t[j][0], loc_t[j][1], loc_t[j]
                                        [2], loc_t[j][3], loc_t[j][4])
                p1 = Polygon(corners_p)
                p2 = Polygon(corners_t)
                inter_area = p1.intersection(p2).area
                iou = inter_area / (loc_p[i][2] * loc_p[i]
                                    [3] + loc_t[j][2] * loc_t[j][3] - inter_area)
                pious.append(round(iou, 4))
        pious_np = np.array(pious)
        pious_np = pious_np.reshape(loc_p.shape[0], -1)
        # print('pious: ', pious_np[pious_np > 0.001])
        gap = np.sum(np.abs(pious_np - jaccard_r_gpu.cpu().numpy()))
        if gap < 0.005:
            print('-------------->pass gap: ', gap)
        else:
            print('-------------->error gap')
            assert 1 == 0, gap

def test_hiou():
    jaccardr_f = JaccardR()
    loc_pxy = torch.randint(10, 300, [3, 2]).float()
    loc_pwh = torch.randint(10, 200, [3, 2]).float()
    loc_pa = torch.randint(-180, 180, [3, 1]) / 180.0 * 3.141593
    loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)

    loc_txy = torch.randint(10, 300, [4, 2]).float()
    loc_twh = torch.randint(10, 200, [4, 2]).float()
    loc_ta = torch.randint(-180, 180, [4, 1]) / 180.0 * 3.141593
    loc_t = torch.cat((loc_txy, loc_twh, loc_ta), -1)
    loc_p = loc_p.float()
    loc_t = loc_t.float()
    corners_p = rbox2corners_torch(loc_p)
    corners_t = rbox2corners_torch(loc_t)
    xmin_p, _ = torch.min(corners_p[:, 0::2], -1, keepdim=True)
    ymin_p, _ = torch.min(corners_p[:, 1::2], -1, keepdim=True)
    xmax_p, _ = torch.max(corners_p[:, 0::2], -1, keepdim=True)
    ymax_p, _ = torch.max(corners_p[:, 1::2], -1, keepdim=True)
    xmin_t, _ = torch.min(corners_t[:, 0::2], -1, keepdim=True)
    ymin_t, _ = torch.min(corners_t[:, 1::2], -1, keepdim=True)
    xmax_t, _ = torch.max(corners_t[:, 0::2], -1, keepdim=True)
    ymax_t, _ = torch.max(corners_t[:, 1::2], -1, keepdim=True)
    box_p = torch.cat((xmin_p, ymin_p, xmax_p, ymax_p), -1)
    print('ymin_p: ', ymin_p)
    print('corners_p: ', corners_p)
    box_t = torch.cat((xmin_t, ymin_t, xmax_t, ymax_t), -1)
    print('box_t: ', (box_t[:, 3] - box_t[:, 1]) * (box_t[:, 2] - box_t[:, 0]))
    overlaps = jaccard(box_p, box_t)
    corners_p = corners_p.cuda()
    corners_t = corners_t.cuda()
    jaccard_r_gpu = jaccardr_f(corners_p, corners_t)
    print('jaccard_r_gpu: ', jaccard_r_gpu)
    print('overlaps: ', overlaps)


def corners_box_form(corners):
    xmin, _ = torch.min(corners[:, 0::2], -1, keepdim=True)
    ymin, _ = torch.min(corners[:, 1::2], -1, keepdim=True)
    xmax, _ = torch.max(corners[:, 0::2], -1, keepdim=True)
    ymax, _ = torch.max(corners[:, 1::2], -1, keepdim=True)
    corners_box = torch.cat((xmin, ymin, xmax, ymax), -1)
    return corners_box


def jaccard_fast(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,5]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,5]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    grid = torch.cat((min_xy, max_xy), -1)
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter_area)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter_area)  # [A,B]
    union = area_a + area_b - inter_area
    h_iou = inter_area / union  # [A,B]
    return torch.cat((grid, h_iou.reshape(h_iou.size(0), -1, 1)), -1)


def test_pious_fast():

    jaccardr_f = JaccardR()
    loc_pxy = torch.randint(10, 300, [3, 2]).float()
    loc_pwh = torch.randint(10, 200, [3, 2]).float()
    loc_pa = torch.randint(-180, 180, [3, 1]) / 180.0 * 3.141593
    loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)

    loc_txy = torch.randint(10, 300, [4, 2]).float()
    loc_twh = torch.randint(10, 200, [4, 2]).float()
    loc_ta = torch.randint(-180, 180, [4, 1]) / 180.0 * 3.141593
    loc_t = torch.cat((loc_txy, loc_twh, loc_ta), -1)
    loc_p = loc_p.float()
    loc_t = loc_t.float()
    corners_p = rbox2corners_torch(loc_p)
    corners_t = rbox2corners_torch(loc_t)
    grid = jaccard_fast(corners_box_form(corners_p), corners_box_form(corners_t))
    overlaps = jaccardr_f(loc_p.cuda(), loc_t.cuda(), grid.cuda())
    print('grid: ', grid.shape)
    print('overlaps: ', overlaps)

    print('..................polygon................')
    loc_p = loc_p.cpu().numpy()
    loc_t = loc_t.cpu().numpy()
    pious = []
    for i in range(loc_p.shape[0]):
        for j in range(loc_t.shape[0]):
            corners_p = rbox2corners_cpu(loc_p[i][0], loc_p[i][1], loc_p[i]
                                     [2], loc_p[i][3], loc_p[i][4])
            corners_t = rbox2corners_cpu(loc_t[j][0], loc_t[j][1], loc_t[j]
                                     [2], loc_t[j][3], loc_t[j][4])
            p1 = Polygon(corners_p)
            p2 = Polygon(corners_t)
            inter_area = p1.intersection(p2).area
            iou = inter_area / (loc_p[i][2] * loc_p[i]
                                [3] + loc_t[j][2] * loc_t[j][3] - inter_area)
            pious.append(round(iou, 4))
    pious_np = np.array(pious)
    pious_np = pious_np.reshape(loc_p.shape[0], -1)
    print('pious: ', pious_np)



def test_pious():
    for item in range(100):
        print('item: ', item)
        loc_pxy = torch.randint(10, 300, [10, 2]).float()
        loc_pwh = torch.randint(20, 200, [10, 2]).float()
        loc_pa = torch.randint(-180, 180, [10, 1]) / 180.0 * 3.141593
        loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)

        loc_txy = torch.randint(10, 300, [10, 2]).float()
        loc_twh = torch.randint(20, 200, [10, 2]).float()
        loc_ta = torch.randint(-180, 180, [10, 1]) / 180.0 * 3.141593
        loc_t = torch.cat((loc_txy, loc_twh, loc_ta), -1)
        grid_xy = template_pixels(512, 512)
        grid_x = template_w_pixels(512)
        num = loc_p.size(0)
        dim = grid_xy.size(0)
        loc_p = loc_p.float()
        loc_t = loc_t.float()
        grid_xy = grid_xy.float()

        loc_p = Variable(loc_p, requires_grad=True)

        piou_loss, pious_big = PIoU(loc_p, loc_t.data, grid_xy.data, 10)
        piou_loss.backward()
        grad_loc_p_big = loc_p.grad
        PiousF = Pious(10)
        # print('pious_big: ', pious_big)
        # print('grad_loc_p_big: ', grad_loc_p_big)

        if torch.cuda.is_available():
            print('------------pixel_weights_gpu test on gpu------------')
            loc_p = loc_p.cuda()
            loc_t = loc_t.cuda()
            grid_xy = grid_xy.cuda()
            grid_x = grid_x.cuda()
            loc_p = Variable(loc_p, requires_grad=True)

            pious_gpu = PiousF(loc_p, loc_t, grid_x)
            # # print('pious_gpu: ', pious_gpu)
            piou = torch.sum(1 - pious_gpu)
            piou.backward()
            loc_p_grad = loc_p.grad
            print('gap piou: ', torch.sum(pious_big.cuda() - pious_gpu) / 10)
            # # print('gap: ', grad_loc_p_big, loc_p_grad)
            print('gap grad piou: ', torch.sum(grad_loc_p_big.cuda() - loc_p_grad) / 10)

            # print('grad: ')
            # print('grad: ', loc_p_grad)
        else:
            print('You device have not a GPU')

def test_hpious():
    for item in range(10):
        print('item: ', item)
        loc_pxy = torch.randint(10, 300, [10, 2]).float()
        loc_pwh = torch.randint(20, 200, [10, 2]).float()
        loc_pa = torch.randint(-180, 180, [10, 1]) / 180.0 * 3.141593
        loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)

        loc_txy = torch.randint(10, 300, [10, 2]).float()
        loc_twh = torch.randint(20, 200, [10, 2]).float()
        loc_ta = torch.randint(-180, 180, [10, 1]) / 180.0 * 3.141593
        loc_t = torch.cat((loc_txy, loc_twh, loc_ta), -1)
        grid_xy = template_pixels(512, 512)
        grid_x = template_w_pixels(512)
        num = loc_p.size(0)
        dim = grid_xy.size(0)
        loc_p = loc_p.float()
        loc_t = loc_t.float()
        grid_xy = grid_xy.float()

        loc_p = Variable(loc_p, requires_grad=True)

        hpiou_loss, hpious_big = HPIoU(loc_p, loc_t.data, grid_xy.data, 10)
        # print('hpious_big: ', hpious_big)
        hpiou_loss.backward()
        grad_loc_p_big = loc_p.grad
        HPiousF = Pious(k=10, is_hard=True)

        if torch.cuda.is_available():
            print('------------pixel_weights_gpu test on gpu------------')
            loc_p = loc_p.cuda()
            loc_t = loc_t.cuda()
            grid_xy = grid_xy.cuda()
            grid_x = grid_x.cuda()
            loc_p = Variable(loc_p, requires_grad=True)

            hpious_gpu = HPiousF(loc_p, loc_t, grid_x)
            # print('hpious_gpu: ', hpious_gpu)
            piou = torch.sum(1 - hpious_gpu)
            piou.backward()
            loc_p_grad = loc_p.grad
            print('gap piou: ', torch.sum(hpious_big.cuda() - hpious_gpu) / 10)
            print('gap grad: ', grad_loc_p_big, loc_p_grad)
            print('gap grad piou: ', torch.sum(grad_loc_p_big.cuda() - loc_p_grad) / 10)

            # print('grad: ')
            # print('grad: ', loc_p_grad)
        else:
            print('You device have not a GPU')
        # print('..................polygon................')
        # loc_p = loc_p.detach().cpu().numpy()
        # loc_t = loc_t.detach().cpu().numpy()
        # pious = []
        # for i in range(loc_p.shape[0]):
        #     corners_p = rbox2corners_cpu(loc_p[i][0], loc_p[i][1], loc_p[i]
        #                             [2], loc_p[i][3], loc_p[i][4])
        #     corners_t = rbox2corners_cpu(loc_t[i][0], loc_t[i][1], loc_t[i]
        #                             [2], loc_t[i][3], loc_t[i][4])
        #     p1 = Polygon(corners_p)
        #     p2 = Polygon(corners_t)
        #     inter_area = p1.intersection(p2).area
        #     iou = inter_area / (loc_p[i][2] * loc_p[i]
        #                         [3] + loc_t[i][2] * loc_t[i][3] - inter_area)
        #     pious.append(round(iou, 4))
        # print('pious: ', pious)

def test_corners():
    loc_pxy = torch.randint(10, 300, [3, 2]).float()
    loc_pwh = torch.randint(20, 200, [3, 2]).float()
    loc_pa = torch.randint(-180, 180, [3, 1]) / 180.0 * 3.141593
    loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)
    corners_p = rbox2corners_torch(loc_p)
    print('corners_p: ', corners_p)
    rbox = loc_p.cpu().numpy()
    for item in range(3):
        point = cv2.boxPoints(((rbox[item][0], rbox[item][1]), (rbox[item][2], rbox[item][3]), rbox[item][4]))
        print('point: ', point)

def test_resize():
    loc_pxy = torch.randint(10, 300, [3, 2]).float()
    loc_pwh = torch.randint(20, 200, [3, 2]).float()
    loc_pa = torch.randint(-180, 180, [3, 1]) / 180.0 * 3.141593
    loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)
    corners_p = rbox2corners_torch(loc_p)
    loc_p_resize = loc_p[:, 0:4] * 0.25
    print('loc_p_resize: ', loc_p_resize)
    corners_p_s = corners_p * 0.25
    corners_p_s = corners_p_s.numpy()
    for item in range(3):
        box = np.int0([corners_p_s[item][0], corners_p_s[item][1],
                       corners_p_s[item][2], corners_p_s[item][3],
                       corners_p_s[item][4], corners_p_s[item][5],
                       corners_p_s[item][6], corners_p_s[item][7]])
        box = box.reshape([-1, 2])
        rect = cv2.minAreaRect(box)
        rwidth, rheight = rect[1]
        rangle = rect[2]
        if rwidth > rheight:
            rangle = np.abs(rangle)
        else:
            temp = rwidth
            rwidth = rheight
            rheight = temp
            rangle = np.abs(rangle) + 90
        rbox = [rect[0][0], rect[0][1], rwidth, rheight, rangle / 180.0 * 3.141593]
        print('rbox: ', rbox)
     



if __name__ == '__main__':
    # test_jaccardr()
    # test_hiou()
    test_pious()
    # PIoU()
    # test_pious_fast()
    # test_hpious()
    # test_corners()
    # test_resize()

        
