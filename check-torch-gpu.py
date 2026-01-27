# Check if CUDA is available
import torch

if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Get the name of the current GPU
    current_gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu_index)
    print(f"Current GPU: {gpu_name}")

    # Or iterate through all GPUs
    for i in range(num_gpus):
      gpu_name = torch.cuda.get_device_name(i)
      print(f"GPU {i}: {gpu_name}")
else:
    print("CUDA is not available. PyTorch is using the CPU.")
