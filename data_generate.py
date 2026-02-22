# 生成训练及测试用的仿真数据集，训练集num = 4000，测试集num = 1000
import torch
import math
import random
import matplotlib.pyplot as plt
points = 256
num = 4000

# 生成模拟数据，包括信号和基线漂移
def generate_data(n_points, a, mu, sigma, a2, freq,  phase, a3, freq2,  phase2):
    x = torch.linspace(0, n_points - 1, n_points)  # 在0到n_points-1之间生成n_points个均匀分布的数据点
    gaussian = a*((1 - torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))*v + 1-v)
    a4 = 5*(torch.poisson(torch.sqrt(gaussian))-torch.sqrt(gaussian))  # 泊松噪声,减torch.sqrt(gaussian)是为了让噪声期望为0
    true_signal = gaussian+a4
    baseline = ((a2 * torch.sin(2 * math.pi * x * freq / len(x) + phase))
                     + (a3 * torch.cos(2 * math.pi * x * freq2 / len(x) + phase2)))+1  #使基线在1上下浮动
    signal_with_drift = true_signal * baseline   # 将基线漂移和噪声添加到原始信号上
    return x, signal_with_drift, true_signal, baseline

# 生成数据
data_list = []
for _ in range(num):
    a = torch.empty(1).uniform_(800, 1200).item()  # 振幅随机在800到1200之 间
    mu = points / 2+torch.empty(1).uniform_(-5, 5).item()  # 中心位置
    sigma = torch.empty(1).uniform_(points / 15, points / 10).item()  # 标准差随机在points/15到points/10之间
    v= torch.empty(1).uniform_(0.7, 0.9).item()  # 条纹对比度
    a2 = torch.empty(1).uniform_(0.08, 0.09).item()  # 基线漂移的正弦振幅在30到70之间
    freq = torch.empty(1).uniform_(0.5, 1).item()  # 频率在0.5到1.5之间
    phase = torch.empty(1).uniform_(0, 2 * math.pi).item()  # 相位在0到2π之间
    a3 = torch.empty(1).uniform_(0.08, 0.09).item()  # 基线漂移的余弦振幅在30到70之间
    freq2 = torch.empty(1).uniform_(0.5, 1).item()  # 第二个频率在0.5到1.5之间
    phase2 = torch.empty(1).uniform_(0, 2 * math.pi).item()  # 第二个相位在0到2π之间


    x, signal_with_drift, true_signal, baseline = generate_data(n_points=points, a=a, mu=mu, sigma=sigma,
                                                                      a2=a2, freq=freq, phase=phase,
                                                                      a3=a3, freq2=freq2, phase2=phase2)
    data_list.append((x, signal_with_drift, true_signal, baseline))

# 打乱数据顺序
random.shuffle(data_list)
# 保存生成的数据
torch.save(data_list, 'generated_data_4000_wo_dc_[0]=0.pt')

print("Data generation complete and saved to 'generated_data_4000_wo_dc_[0]=0.pt'")


x, signal_with_drift, true_signal, baseline = random.choice(data_list)
# 创建主坐标轴
fig, ax1 = plt.subplots()
# 绘制带有基线漂移的信号 (主坐标轴)
ax1.plot(x.numpy(), signal_with_drift.numpy(), label='Signal with Drift', color='blue')
ax1.set_xlabel('Time')
ax1.set_ylabel('Signal', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')
# 创建次坐标轴
ax2 = ax1.twinx()
# 绘制基线漂移 (次坐标轴)
ax2.plot(x.numpy(), baseline.numpy(), label='Baseline', color='green', linestyle='--')
ax2.set_ylabel('Baseline', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')
# 控制次坐标轴的范围
ax2.set_ylim(0.5, 1.5)  # 设置次坐标轴 y 轴的范围
ax1.grid(True)  # 在主坐标轴上显示网格线（横纵方向）
# 显示图形
plt.show()











