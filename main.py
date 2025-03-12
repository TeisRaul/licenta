import torch

# Încarcă fișierul .pth pe CPU
data = torch.load('data.pth', map_location=torch.device('cpu'))

# Verifică conținutul
print(data)