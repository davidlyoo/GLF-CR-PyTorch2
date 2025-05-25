import torch
import kernelconv2d_cuda

x = torch.randn(1,1,8,8, device='cuda')
k = torch.randn(1,1,3,3, device='cuda')
out = torch.empty(1,1,6,6, device='cuda')      # (H−K+1, W−K+1)

# 네 번째 인자로 out 텐서를 넘긴다.
ret = kernelconv2d_cuda.forward(x, k, 3, out)
print("return code:", ret)                     # 1 이면 성공
print("out.shape:", out.shape)
print("NaN in out?:", torch.isnan(out).any().item())