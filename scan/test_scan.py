import torch
from torch.utils.cpp_extension import load
import os 

module_path = os.path.dirname(__file__)

# Load the extension
parallel_scan_module = load(
    name="parallel_scan2",
    sources=["scan_wrapper.cpp", "scan_kernel.cu"],
    verbose=True  # Set this to True for helpful compilation output
)

B, D, L = 16, 64, 1024
input_tensor = torch.rand((B, D, L), device='cuda:0')
output_tensor = torch.zeros((B, D, L), device='cuda:0')

# Use the loaded extension
breakpoint()

parallel_scan_module.parallel_scan(output_tensor, input_tensor, B, D, L)

output_tensor2 = input_tensor.cumsum(-1)

breakpoint()

