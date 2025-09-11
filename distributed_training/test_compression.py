# import torch
# import numpy as np

# from hivemind.compression.quantization import Uniform8BitQuantization, TwoBitUniformEF
# from hivemind.compression.base import CompressionInfo
# # create test tensor
# torch.manual_seed(0)
# x = torch.randn(1000)

# # prepare compressor objects
# quant8 = Uniform8BitQuantization()
# quant2 = TwoBitUniformEF(range_in_sigmas=3.0, stochastic=True, center_by_mean=True)

# # dummy CompressionInfo (not used by these quantizers except maybe for names)
# info = CompressionInfo(key=0, descriptor=None)

# # --- 8-bit quantization ---
# msg8 = quant8.compress(x, info)
# x_q8 = quant8.extract(msg8)
# err8 = torch.norm(x_q8 - x) / torch.norm(x)
# print(f"Relative error (8-bit, one step): {err8.item():.4f}")

# # # --- 2-bit quantization with EF ---
# # msg2 = quant2.compress(x, info)
# # x_q2 = quant2.extract(msg2)
# # err2 = torch.norm(x_q2 - x) / torch.norm(x)
# # print(f"Relative error (8-bit): {err8.item():.4f}")
# # print(f"Relative error (2-bit + EF, single step): {err2.item():.4f}")

# # iterative 2-bit with EF (let quant2 manage residuals internally)
# errors = []
# for step in range(1, 11):
#     msg2 = quant2.compress(x, info)
#     x_q2 = quant2.extract(msg2)
#     err = torch.norm(x_q2 - x) / torch.norm(x)
#     errors.append(err.item())

# print("\nRelative error (2-bit + EF):")
# for i, e in enumerate(errors, 1):
#     print(f"Step {i:2d}: {e:.4f}")

# print("\nFirst few values:")
# for i in range(5):
#     print(f"x={x[i]:+.3f}, q8={x_q8[i]:+.3f}, q2={x_q2[i]:+.3f}")
breakpoint()

import torch
import numpy as np
from hivemind.compression.quantization import Uniform8BitQuantization, TwoBitUniformEF
from hivemind.compression.base import CompressionInfo

torch.manual_seed(0)

# simulate a sequence of 10 gradient-like updates
num_steps = 10
tensor_size = 1000
updates = [torch.randn(tensor_size) * 0.1 for _ in range(num_steps)]

# compressors
quant8 = Uniform8BitQuantization()
quant2 = TwoBitUniformEF(
    range_in_sigmas=3.0, block_size=4096, stochastic=True, center_by_mean=True
)
info = CompressionInfo(key=0, descriptor=None)

# store cumulative reconstructed tensor
x_reconstructed_2bit = torch.zeros(tensor_size)

print(
    f"{'Step':>4} | {'Err 8-bit':>10} | {'Err 2-bit EF':>12} | {'Size 8-bit':>10} | {'Size 2-bit':>10}"
)

for step, update in enumerate(updates, 1):
    # --- 8-bit quantization (stateless) ---
    msg8 = quant8.compress(update, info)
    x_q8 = quant8.extract(msg8)
    err8 = torch.norm(x_q8 - update) / torch.norm(update)

    # --- 2-bit quantization with EF (stateful) ---
    msg2 = quant2.compress(update, info)  # EF tracked internally
    x_q2 = quant2.extract(msg2)
    err2 = torch.norm(x_q2 - update) / torch.norm(update)

    # accumulate reconstruction
    x_reconstructed_2bit += x_q2

    # sizes in bytes
    size8 = len(msg8.buffer)
    size2 = len(msg2.buffer)

    print(f"{step:4d} | {err8:10.4f} | {err2:12.4f} | {size8:10d} | {size2:10d}")
