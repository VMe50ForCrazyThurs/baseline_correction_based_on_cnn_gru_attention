
from torch.utils.data import DataLoader, TensorDataset
from define import CNNGRUModel
from define import CNNLSTMModel
from define import CNNGRUAttentionModel
from define import Loss
import torch
import time  # 导入时间模块


batch_size=64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.backends.cudnn.enabled)  # 确保 cuDNN 已启用

# model_path = input("Please provide the path to the saved model file (.pth): ").strip()
model_path = 'best_model_weights265.pth'
print(f"Loaded model weights from {model_path}")

# 初始化模型
# model = CNNGRUModel().to(device)
model = CNNLSTMModel().to(device)
# model = CNNGRUAttentionModel().to(device)
model.load_state_dict(torch.load(model_path))

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

file_path = 'generated_data_1000.pt'
# 加载数据
test_data = torch.load(file_path)
criterion = Loss()

test_data = [(signal_with_drift.clone().detach(), baseline_drift.clone().detach()) for (_, signal_with_drift, _, baseline_drift) in test_data]
# 创建 DataLoader
signal_with_drift_list, baseline_drift_list = zip(*test_data)
test_dataset = TensorDataset(torch.stack(signal_with_drift_list), torch.stack(baseline_drift_list))
# 使用 num_workers 参数来启用多线程读取数据，pin_memory=True 可以帮助加速数据传输到 GPU
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

model.eval()  # 设置模型为评估模式
test_loss = 0.0
total_time = 0.0  # 用于记录总处理时间
num_samples = 0  # 用于记录总样本数量
with torch.no_grad():  # 评估过程中不计算梯度
    for batch_data in test_loader:
        signal_with_drift, baseline_drift = batch_data
        signal_test = signal_with_drift.view(batch_size, 1, -1).float().to(device)  # 将信号转换为张量，并调整为适应LSTM输入的形状
        drift_test = baseline_drift.view(batch_size, -1, 1).float().to(device)  # 将基线漂移转换为张量
        outputs = model(signal_test)  # 预测基线漂移
        num_samples += signal_test.size(0)  # 累计样本数量

        loss = criterion(outputs, drift_test)  # 计算测试集上的损失
        test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)  # 计算测试集上的平均损失
    print(f'Test: {avg_test_loss:.8f}')
