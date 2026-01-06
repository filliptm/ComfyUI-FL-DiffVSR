import torch
import torch.nn.functional as F

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    if flow.dim() == 4 and flow.shape[1] == 2:
        flow = flow.permute(0, 2, 3, 1)  # [N, 2, H, W] -> [N, H, W, 2]

    assert x.size()[-2:] == flow.size()[1:3]
    _, _, H, W = x.size()

    # Ensure flow matches input dtype and device
    flow = flow.to(dtype=x.dtype, device=x.device)

    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=False)
    return output

def get_flow(of_model, target, source, rescale_factor=1):
    flows = of_model(target, source)
    flow = flows[-1]
    flow = F.interpolate(flow//rescale_factor, scale_factor=1/rescale_factor, mode='bilinear') if rescale_factor != 1 else flow
    flow = flow.permute(0, 2, 3, 1)  # permute to B, H, W, 2
    return flow

def compute_flow_magnitude(flow):
    flow_mag = flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2
    return flow_mag

def compute_flow_gradients(flow):
    B = flow.shape[0]
    H = flow.shape[1]
    W = flow.shape[2]

    device = flow.device
    flow_x_du = torch.zeros((B, H, W), device=device)
    flow_x_dv = torch.zeros((B, H, W), device=device)
    flow_y_du = torch.zeros((B, H, W), device=device)
    flow_y_dv = torch.zeros((B, H, W), device=device)

    flow_x = flow[:, :, :, 0]
    flow_y = flow[:, :, :, 1]

    flow_x_du[:, :, :-1] = flow_x[:, :, :-1] - flow_x[:, :, 1:]
    flow_x_dv[:, :-1, :] = flow_x[:, :-1, :] - flow_x[:, 1:, :]
    flow_y_du[:, :, :-1] = flow_y[:, :, :-1] - flow_y[:, :, 1:]
    flow_y_dv[:, :-1, :] = flow_y[:, :-1, :] - flow_y[:, 1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

def detect_occlusion(fw_flow, bw_flow):
    # inputs: flow_forward, flow_backward
    # return: occlusion mask
    tmp = bw_flow
    bw_flow = fw_flow
    fw_flow = tmp

    fw_flow_w = flow_warp(fw_flow.permute(0,3,1,2), bw_flow).permute(0,2,3,1)

    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2

    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    mask = torch.logical_or(mask1, mask2)
    occlusion = torch.ones((fw_flow.shape[0], fw_flow.shape[1], fw_flow.shape[2]), device=fw_flow.device)
    occlusion[mask == 1] = 0

    return occlusion

def get_flow_forward_backward(net, current, prev, rescale_factor=1):
    flow_forward = get_flow(net, current, prev, rescale_factor=rescale_factor)
    flow_backward = get_flow(net, prev, current, rescale_factor=rescale_factor)
    return flow_forward, flow_backward
