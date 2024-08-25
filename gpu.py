import torch
# 判断是否支持 GPU 加速
if torch.cuda.is_available():
    print("GPU 加速可用")
else:
    print("GPU 加速不可用")
flag = torch.cuda.is_available()
print(flag)