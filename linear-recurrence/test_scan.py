import torch
from torch.utils.cpp_extension import load
import os 

module_path = os.path.dirname(__file__)

# Load the extension
parallel_scan_module = load(
    name="linear_recurrence",
    sources=["scan_wrapper.cpp", "scan_kernel.cu"],
    verbose=True  # Set this to True for helpful compilation output
)


B, D, L = 16, 64, 1024
input_tensor = torch.rand((B, D, L), device='cuda:0')
gate = torch.rand((B, D, L), device='cuda:0').sigmoid()

output_tensor = torch.zeros((B, D, L), device='cuda:0')

# Use the loaded extension
# breakpoint()

parallel_scan_module.parallel_scan(output_tensor, input_tensor, gate, B, D, L)

# output_tensor2 = input_tensor.cumsum(-1)

output_tensor2 = torch.zeros((B, D, L), device='cuda:0')
for i in range(L):
    if i == 0:
        output_tensor2[:, :, i] = input_tensor[:, :, i]
    else:
        output_tensor2[:, :, i] = gate[:, :, i] * output_tensor2[:, :, i-1] + input_tensor[:, :, i]

breakpoint()


