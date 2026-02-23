import torch
import matplotlib.pyplot as plt
from define import CNNLSTMModel
from define import Loss
from define import CNNGRUAttentionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_path = 'best_model_weights_cnn_gru_att.pth'
print(f"Loaded model weights from {model_path}")

# 初始化模型
# model = CNNLSTMModel().to(device)
model=CNNGRUAttentionModel().to(device)
model.load_state_dict(torch.load(model_path))

file_path = 'generated_data_1000.pt'
# 加载数据
test_data = torch.load(file_path)

criterion = Loss()

model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 评估过程中不计算梯度
    # 随机选择数据并获取其序号
    # index, (x, signal_with_drift, true_signal, baseline_drift) = random.choice(list(enumerate(train_data)))
    index = 521  # 读取的样本索引，可以根据需求更改
    # 使用索引访问 train_data 中的样本
    x, signal_with_drift, true_signal, baseline_drift = test_data[index]

    # 输出序号和数据
    print(f"Selected index: {index}")

    signal_test = signal_with_drift.view(1, 1, -1).float().to(device)  # 将信号转换为张量，并调整为适应LSTM输入的形状
    estimated_baseline = model(signal_test)  # 使用模型预测基线漂移，并转换为NumPy数组
    baseline=baseline_drift.view(1, -1, 1).to(device)
    loss = criterion(estimated_baseline, baseline)
    print(f"Loss: {loss.item()}")
    estimated_baseline= estimated_baseline.flatten().cpu().numpy()
    corrected_signal = signal_with_drift.numpy() / estimated_baseline  # 从带有漂移的信号中减去估计的漂移得到校正后的信号

    # 可视化基线漂移和校正后的信号
    plt.plot(x.numpy(), signal_with_drift.numpy(), label='Signal with Drift', color='blue')  # 绘制带有基线漂移的信号
    plt.plot(x.numpy(), corrected_signal, label='Corrected Signal', color='red')  # 绘制校正后的信号
    plt.plot(x.numpy(), baseline_drift.numpy(), label='Baseline Drift', color='green', linestyle='--')  # 绘制基线漂移
    plt.plot(x.numpy(), estimated_baseline, label='Estimated Baseline Drift', color='red', linestyle='--')  # 绘制估计的基线漂移
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()  # 显示图形
