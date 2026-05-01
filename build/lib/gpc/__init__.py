import os

# Set XLA flags for better performance
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "
