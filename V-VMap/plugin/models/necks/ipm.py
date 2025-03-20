import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models import NECKS


def get_campos(reference_points, ego2cam, img_shape):
    '''
        Find the each refence point's corresponding pixel in each camera
        Args: 
            reference_points: [B, num_query, 3]
            ego2cam: (B, num_cam, 4, 4)
        Outs:
            reference_points_cam: (B*num_cam, num_query, 2)
            mask:  (B, num_cam, num_query)
            num_query == W*H
    '''

    ego2cam = reference_points.new_tensor(ego2cam)  # (B, N, 4, 4)
    reference_points = reference_points.clone()

    B, num_query = reference_points.shape[:2]
    num_cam = ego2cam.shape[1]

    # reference_points (B, num_queries, 4)
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    reference_points = reference_points.view(
        B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)

    ego2cam = ego2cam.view(
        B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)

    # reference_points_cam (B, num_cam, num_queries, 4)
    reference_points_cam = (ego2cam @ reference_points).squeeze(-1)

    eps = 1e-9
    mask = (reference_points_cam[..., 2:3] > eps)

    reference_points_cam =\
        reference_points_cam[..., 0:2] / \
        reference_points_cam[..., 2:3] + eps

    reference_points_cam[..., 0] /= img_shape[1]
    reference_points_cam[..., 1] /= img_shape[0]

    # from 0~1 to -1~1
    reference_points_cam = (reference_points_cam - 0.5) * 2

    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))

    # (B, num_cam, num_query)
    mask = mask.view(B, num_cam, num_query)
    reference_points_cam = reference_points_cam.view(B*num_cam, num_query, 2)

    return reference_points_cam, mask

def construct_plane_grid(xbound, ybound, height: float, dtype=torch.float32):
    '''
        Returns:
            plane: H, W, 3
    '''

    xmin, xmax = xbound[0], xbound[1]
    num_x = int((xbound[1] - xbound[0]) / xbound[2])
    ymin, ymax = ybound[0], ybound[1]
    num_y = int((ybound[1] - ybound[0]) / ybound[2])

    x = torch.linspace(xmin, xmax, num_x, dtype=dtype)
    y = torch.linspace(ymin, ymax, num_y, dtype=dtype)

    # [num_y, num_x]
    y, x = torch.meshgrid(y, x)

    z = torch.ones_like(x) * height

    # [num_y, num_x, 3]
    plane = torch.stack([x, y, z], dim=-1)

    return plane
    
    
def bilinear_grid_sample(im,grid,align_corners = False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    device = x0.device

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0).to(device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0).to(device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0).to(device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0).to(device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

@NECKS.register_module()
class IPM(nn.Module):
    r"""
    Notes
    -----
    Adapted from https://github.com/Mrmoore98/VectorMapNet_code/blob/mian/plugin/models/backbones/ipm_backbone.py#L238.

    """
    def __init__(self,
                 xbound,
                 ybound,
                 zbound,
                 out_channels,
                 ):
        super().__init__()
        self.x_bound = xbound
        self.y_bound = ybound
        heights = [zbound[0]+i*zbound[2] for i in range(int((zbound[1]-zbound[0])//zbound[2])+1)]
        self.heights = heights

        self.outconvs = nn.Conv2d((out_channels+3)*len(heights), out_channels, kernel_size=3, stride=1, padding=1)  # same

        # bev_plane
        bev_planes = [construct_plane_grid(
            xbound, ybound, h) for h in self.heights]
        self.register_buffer('bev_planes', torch.stack(
            bev_planes),)  # nlvl,bH,bW,2

    def forward(self, cam_feat, ego2cam, img_shape):
        '''
            inverse project 
            Args:
                cam_feat: B*ncam, C, cH, cW
                img_shape: tuple(H, W)
            Returns:
                project_feat: B, C, nlvl, bH, bW
                bev_feat_mask: B, 1, nlvl, bH, bW
        '''
        B = ego2cam.shape[0]
        C = cam_feat.shape[1]
        bev_grid = self.bev_planes.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        nlvl, bH, bW = bev_grid.shape[1:4]
        bev_grid = bev_grid.flatten(1, 3)  # B, nlvl*W*H, 3

        # Find points in cam coords
        # bev_grid_pos: B*ncam, nlvl*bH*bW, 2
        bev_grid_pos, bev_cam_mask = get_campos(bev_grid, ego2cam, img_shape)
        # B*cam, nlvl*bH, bW, 2
        #bev_grid_pos = bev_grid_pos.unflatten(-2, (nlvl*bH, bW))
        bev_grid_pos = bev_grid_pos.view(B, nlvl*bH, bW, 2)
        # project feat from 2D to bev plane
        projected_feature = bilinear_grid_sample(
            cam_feat, bev_grid_pos).view(B, -1, C, nlvl, bH, bW)  # B,cam,C,nlvl,bH,bW

        # B,cam,nlvl,bH,bW
        #bev_feat_mask = bev_cam_mask.unflatten(-1, (nlvl, bH, bW))
        bev_feat_mask = bev_cam_mask.view(B, 1, nlvl, bH, bW)
        # eliminate the ncam
        # The bev feature is the sum of the 6 cameras
        bev_feat_mask = bev_feat_mask.unsqueeze(2)
        projected_feature = (projected_feature*bev_feat_mask).sum(1)
        num_feat = bev_feat_mask.sum(1)

        projected_feature = projected_feature / \
            num_feat.masked_fill(num_feat == 0, 1)

        # concatenate a position information
        # projected_feature: B, bH, bW, nlvl, C+3
        bev_grid = bev_grid.view(B, nlvl, bH, bW,
                                 3).permute(0, 4, 1, 2, 3)
        projected_feature = torch.cat(
            (projected_feature, bev_grid), dim=1)

        bev_feat, bev_feat_mask = projected_feature, bev_feat_mask.sum(1) > 0

        # multi level into a same
        bev_feat = bev_feat.flatten(1, 2)
        bev_feat = self.outconvs(bev_feat)

        return bev_feat
