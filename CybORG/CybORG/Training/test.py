import torch

print("MPS Available:", torch.backends.mps.is_available())  # Should print True
print("MPS Built:", torch.backends.mps.is_built())  # Should print True