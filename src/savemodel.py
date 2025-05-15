import torch
from model import SimpleCNN   # 당신의 모델 정의

model = SimpleCNN()
model.eval()                  # 반드시 eval 모드
scripted = torch.jit.script(model)   # 또는 torch.jit.trace(model, dummy_input)
scripted.save("simplecnn.pt")        # <-- Netron에서 simplecnn.pt 열기
