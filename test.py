import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

devNumber = torch.cuda.current_device()

print(f"Current device number is: {devNumber}")

devName = torch.cuda.get_device_name(devNumber)

print(f"gpu NAME IS {devName}")