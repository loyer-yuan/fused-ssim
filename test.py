import torch
from fused_ssim import fused_ssim
from fused_ssim import fused_ssim_opt

# predicted_image, gt_image: [BS, CH, H, W]
# predicted_image is differentiable
# gt_image = torch.rand(2, 3, 1080, 1920)

gt_image1 = torch.rand(1, 3, 1080, 1920).cuda()
predicted_image1 = torch.nn.Parameter(torch.rand_like(gt_image1)).cuda()
torch.cuda.nvtx.range_push("fused_ssim")
ssim_value1 = fused_ssim(predicted_image1, gt_image1)
ssim_value2 = fused_ssim_opt(predicted_image1, gt_image1)
if ssim_value1 != ssim_value2:
    print("Error")
torch.cuda.nvtx.range_pop()

# gt_image2 = torch.rand(1, 3, 2160, 3840).cuda()
# predicted_image2 = torch.nn.Parameter(torch.rand_like(gt_image2)).cuda()
# torch.cuda.nvtx.range_push("fused_ssim")
# ssim_value2 = fused_ssim(predicted_image2, gt_image2)
# torch.cuda.nvtx.range_pop()