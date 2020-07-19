import torch
from torch.nn import Module
from torch.autograd import Function
import torch.nn.functional as F
from torch.autograd import Variable
# rate / cx
def test_rate_cx():
    print('--------------rate / cx-----------------')
    loc = torch.rand((2, 5))
    loc = Variable(loc, requires_grad=True)
    xy = torch.rand((2, 5))
    dis = F.pairwise_distance(loc[:, 0:2], xy[:, 0:2])
    rate = (loc[:, 0] - xy[:, 0]) / dis
    rate_sum = torch.sum(rate)
    rate_sum.backward()
    loc_dxdy = loc[:, 0:2] - xy[:, 0:2]
    dydy = loc_dxdy[:, 1] * loc_dxdy[:, 1]
    dxdy = loc_dxdy[:, 0] * loc_dxdy[:, 1]
    print('dis: ', dis)
    print('loc_grad: ', loc.grad)
    print('simulate loc_grad')
    dcx = dydy / dis / dis / dis
    dcy = -dxdy / dis / dis / dis
    print('dcx: ', dcx)
    print('dcy: ', dcy)

# a / cx, cy, angle
def test_a_cx():
    print('--------------a / cx-----------------')
    loc = torch.rand((2, 5))
    loc = Variable(loc, requires_grad=True)
    xy = torch.rand((2, 5))
    dis = F.pairwise_distance(loc[:, 0:2], xy[:, 0:2])
    rate = (loc[:, 0] - xy[:, 0]) / dis
    cos_v = torch.acos(rate)
    a1 = loc[:, -1] + cos_v
    a2 = loc[:, -1] - cos_v
    a = torch.where(loc[:, 1] > xy[:, 1], a1, a2)
    a_sum = torch.sum(a)
    a_sum.backward()
    
    loc_dxdy = loc[:, 0:2] - xy[:, 0:2]
    print('dis: ', dis)
    print('loc_grad: ', loc.grad)
    print('simulate loc_grad')
    adcx = -loc_dxdy[:, 1] / dis / dis
    adcy = loc_dxdy[:, 0] / dis / dis

    print('adcx: ', adcx)
    print('adcy: ', adcy)

# dis_w / cx, cy, angle
def test_dis_cx():
    print('--------------a / cx-----------------')
    loc = torch.rand((2, 5))
    loc = Variable(loc, requires_grad=True)
    xy = torch.rand((2, 5))
    dis = F.pairwise_distance(loc[:, 0:2], xy[:, 0:2])
    rate = (loc[:, 0] - xy[:, 0]) / dis
    cos_v = torch.acos(rate)
    a1 = loc[:, -1] + cos_v
    a2 = loc[:, -1] - cos_v
    a = torch.where(loc[:, 1] > xy[:, 1], a1, a2)

    dis_w = dis * torch.abs(torch.cos(a))
    dis_w_sum = torch.sum(dis_w)
    dis_w_sum.backward()
    loc_dxdy = loc[:, 0:2] - xy[:, 0:2]
    print('loc_grad: ', loc.grad)
    print('simulate loc_grad')
    dcx = (loc_dxdy[:, 0] * torch.cos(a) + \
        loc_dxdy[:, 1] * torch.sin(a)) / dis
    dcy = (loc_dxdy[:, 1] * torch.cos(a) - \
        loc_dxdy[:, 0] * torch.sin(a)) / dis
    da = -torch.sin(a) * dis

    dcx = torch.where(torch.cos(a) > 0, dcx, -dcx)
    dcy = torch.where(torch.cos(a) > 0, dcy, -dcy)
    da = torch.where(torch.cos(a) > 0, da, -da)

    print('dcx: ', dcx)
    print('dcy: ', dcy)
    print('da: ', da)

# dis_h / cx, cy, angle
def test_dish_cx():
    print('--------------a / cx-----------------')
    loc = torch.rand((2, 5))
    loc = Variable(loc, requires_grad=True)
    xy = torch.rand((2, 5))
    dis = F.pairwise_distance(loc[:, 0:2], xy[:, 0:2])
    rate = (loc[:, 0] - xy[:, 0]) / dis
    cos_v = torch.acos(rate)
    a1 = loc[:, -1] + cos_v
    a2 = loc[:, -1] - cos_v
    a = torch.where(loc[:, 1] > xy[:, 1], a1, a2)

    dis_w = dis * torch.abs(torch.sin(a))
    dis_w_sum = torch.sum(dis_w)
    dis_w_sum.backward()
    loc_dxdy = loc[:, 0:2] - xy[:, 0:2]
    print('loc_grad: ', loc.grad)
    print('simulate loc_grad')
    dcy = (loc_dxdy[:, 0] * torch.cos(a) + \
        loc_dxdy[:, 1] * torch.sin(a)) / dis

    dcx = -(loc_dxdy[:, 1] * torch.cos(a) - \
        loc_dxdy[:, 0] * torch.sin(a)) / dis

    da = torch.cos(a) * dis
    dcx = torch.where(torch.sin(a) > 0, dcx, -dcx)
    dcy = torch.where(torch.sin(a) > 0, dcy, -dcy)
    da = torch.where(torch.sin(a) > 0, da, -da)

    print('dcx: ', dcx)
    print('dcy: ', dcy)
    print('da: ', da)
# p / f
def test_pf():
    loc_pxy = torch.randint(200, 300, [10, 2]).float()
    loc_pwh = torch.randint(100, 200, [10, 2]).float()
    loc_pa = torch.randint(-180, 180, [10, 1]) / 180.0 * 3.141593
    loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)
    loc_p = Variable(loc_p, requires_grad=True)
    factor_wh = -10 * (loc_p[:, 0:2] - loc_p[:, 2:4] / 2.)
    print(factor_wh)
    kerner_w = 1.0 / (torch.exp(factor_wh[:, 0]) + 1.0)
    kerner_h = 1.0 / (torch.exp(factor_wh[:, 1]) + 1.0)
    pixel_area = (1 - kerner_h) * (1 - kerner_w)
    pixel_area_sum = torch.sum(pixel_area)
    pixel_area_sum.backward()
    print('loc_grad: ', loc_p.grad)
    print('simulate loc_grad')
    print('cx_grad: ', 0 * pixel_area * kerner_h + -10 * pixel_area * kerner_w)
    print('cy_grad: ', -10 * pixel_area * kerner_h + 0 * pixel_area * kerner_w)
    print('w_grad: ', 0 * pixel_area * kerner_h + 5 * pixel_area * kerner_w)
    print('h_grad: ', 5 * pixel_area * kerner_h + 0 * pixel_area * kerner_w)


def template_pixels(height, width):
    xv, yv = torch.meshgrid(
        [torch.arange(width), torch.arange(height)])
    xy = torch.stack((xv, yv), -1)
    grid_xy = xy.reshape(-1, 2).float() + 0.5

    return grid_xy

def test_piou_pw():
    loc = torch.rand((9, 2))
    loc = Variable(loc, requires_grad=True)
    inter_pixel_area = loc[:, 0] * loc[:, 1]
    union_pixel_area = loc[:, 0] + loc[:, 1] - inter_pixel_area
    inter_area = torch.sum(inter_pixel_area, 0)
    union_area = torch.sum(union_pixel_area, 0)

    pious = inter_area / (union_area + 1e-9)
    pious = torch.sum(pious)
    pious.backward()
    print('loc_grad: ', loc.grad)
    print('simulate loc_grad')
    up = loc[:, 1] * union_area - inter_area * (1 - loc[:, 1])
    loc_grad = up / (union_area + 1e-9) / (union_area + 1e-9)
    print('loc_grad: ', loc_grad)

def test_piou():

    loc_pxy = torch.randint(10, 11, [2]).float()
    loc_pwh = torch.randint(10, 20, [2]).float()
    loc_pa = torch.randint(-180, 180, [1]) / 180.0 * 3.141593
    loc = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)
    loc = loc.unsqueeze(0).expand(900, 5)
    xy = template_pixels(30, 30)

    # loc = torch.randint(10, 11, [900, 2]).float()
    # xy = torch.randint(10, 11, [900, 2]).float()
    loc = Variable(loc, requires_grad=True)
    dis = F.pairwise_distance(loc[:, 0:2], xy[:, 0:2]) + 1e-9

    dxdy = loc[:, 0:2] - xy[:, 0:2]
    k = 10
    a1 = loc[:, -1] + torch.acos((loc[:, 0] - xy[:, 0]) / dis)
    a2 = loc[:, -1] - torch.acos((loc[:, 0] - xy[:, 0]) / dis)
    a = torch.where(loc[:, 1] > xy[:, 1], a1, a2)
    dis_w = dis * torch.abs(torch.cos(a))
    dis_h = dis * torch.abs(torch.sin(a))
    factor_w = -k * (dis_w - loc[:, 2] / 2.)
    factor_h = -k * (dis_h - loc[:, 3] / 2.)

    factor_w = torch.clamp(factor_w, -50, 50)
    factor_h = torch.clamp(factor_h, -50, 50)

    kerner_w = 1.0 / (torch.exp(factor_w) + 1)
    kerner_h = 1.0 / (torch.exp(factor_h) + 1)
    pixel_weight = (1.0 - kerner_w) * (1.0 - kerner_h)
    pixel_weights = torch.sum(pixel_weight)
    pixel_weights.backward()
    grad_loc = loc.grad
    index = grad_loc[:, 0] > 1e-3
    print('grad_loc: ', grad_loc[index][0])
    print('loc: ', loc[index][0])
    print('xy: ', xy[index][0])
    

    dis_w_dcx = (dxdy[:, 1] * torch.sin(a) + dxdy[:, 0] * torch.cos(a)) / dis
    dis_w_dcy = (dxdy[:, 1] * torch.cos(a) - dxdy[:, 0] * torch.sin(a)) / dis
    dis_w_da = -dis * torch.sin(a)

    dis_h_dcx = (dxdy[:, 0] * torch.sin(a) - dxdy[:, 1] * torch.cos(a)) / dis
    dis_h_dcy = (dxdy[:, 1] * torch.sin(a) + dxdy[:, 0] * torch.cos(a)) / dis
    dis_h_da = dis * torch.cos(a)

    dis_w_dcx = torch.where(torch.cos(a) > 0, dis_w_dcx, -dis_w_dcx)
    dis_w_dcy = torch.where(torch.cos(a) > 0, dis_w_dcy, -dis_w_dcy)
    dis_w_da = torch.where(torch.cos(a) > 0, dis_w_da, -dis_w_da)

    dis_h_dcx = torch.where(torch.sin(a) > 0, dis_h_dcx, -dis_h_dcx)
    dis_h_dcy = torch.where(torch.sin(a) > 0, dis_h_dcy, -dis_h_dcy)
    dis_h_da = torch.where(torch.sin(a) > 0, dis_h_da, -dis_h_da)

    fw_dcx = - k * dis_w_dcx
    fw_dcy = - k * dis_w_dcy
    fw_da = - k * dis_w_da
    fw_dw = k / 2.

    fh_dcx = - k * dis_h_dcx
    fh_dcy = - k * dis_h_dcy
    fh_da = - k * dis_h_da
    fh_dh = k / 2.
    kwp = kerner_w * pixel_weight
    khp = kerner_h * pixel_weight
    p_dcx = kwp * fw_dcx + khp * fh_dcx
    p_dcy = kwp * fw_dcy + khp * fh_dcy
    p_dw = kwp * fw_dw
    p_dh = khp * fh_dh
    p_da = kwp * fw_da + khp * fh_da
    print('p_dcx: ', p_dcx[index][0])
    print('p_dcy: ', p_dcy[index][0])
    print('p_dw: ', p_dw[index][0])
    print('p_dh: ', p_dh[index][0])
    print('p_da: ', p_da[index][0])






if __name__ == "__main__":
    # test_rate_cx()
    # test_a_cx()
    # test_dis_cx()
    # test_dish_cx()
    # test_pf()
    # test_piou_pw()
    test_piou()
