# 适合 Conv1d 的形状: (batch_size, feature_size, sequence_length)
# 适合 LSTM 的形状，即 (batch_size, sequence_length, feature_size)
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from define import CNNGRUModel
from define import CNNLSTMModel
from define import CNNGRUAttentionModel
from define import Loss
import torch
import matplotlib.pyplot as plt

epochs = 5000  # 训练的轮数
batch_size = 64 # 每个batch的大小
lr = 0.02
early_stopping_patience = 15  # 早停的耐心值

# 数据文件的路径
train_file_path = 'generated_data_4000.pt'
test_file_path = 'generated_data_1000.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 初始化模型
# model = CNNLSTMModel().to(device)
model= CNNGRUAttentionModel().to(device)
model._initialize_weights()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total_params/1e6:.3f} M')

# weight_decay 相当于 L2 正则化的强度。值越大，正则化的效果越强，参数的更新受到的惩罚越大，通常可以防止模型过拟合。
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#指数衰减（Exponential Decay）：学习率随着训练的进行按指数衰减，可以有效地让学习率逐渐变小。
scheduler = ExponentialLR(optimizer, gamma=0.95)
# 混合损失函数，包含均方误差和L1正则化平滑项
criterion = Loss()  # 使用混合损失函数
criterion = criterion.to(device) # 如果想要把模型搬到GPU上跑，就要在定义优化器之前就完成.cuda( )这一步

train_data=torch.load(train_file_path)
test_data=torch.load(test_file_path)
# 更新train_data和test_data的解包以适应包含四个值的情况
train_data = [(signal_with_drift.clone().detach(), baseline_drift.clone().detach()) for (_, signal_with_drift, _, baseline_drift) in train_data]
test_data = [(signal_with_drift.clone().detach(), baseline_drift.clone().detach()) for (_, signal_with_drift, _, baseline_drift) in test_data]
# 创建 DataLoader
signal_with_drift_list, baseline_drift_list = zip(*train_data)
train_dataset = TensorDataset(torch.stack(signal_with_drift_list), torch.stack(baseline_drift_list))
signal_with_drift_list, baseline_drift_list = zip(*test_data)
test_dataset = TensorDataset(torch.stack(signal_with_drift_list), torch.stack(baseline_drift_list))

# 使用 num_workers 参数来启用多线程读取数据，pin_memory=True 可以帮助加速数据传输到 GPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

# 训练模型
loss_history = []  # 存储每个epoch的训练损失值
test_loss_history = []  # 存储每个epoch的测试损失值
best_loss = float('inf')  # 用于记录最佳模型的损失
no_improve_epoch = 0  # 用于记录连续未改善的epoch次数

for epoch in range(epochs):
    # start_time = time.time()  # 记录当前epoch开始时间
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    for batch_data in train_loader:
        signal_with_drift, baseline_drift = batch_data
        signal_train = signal_with_drift.view(batch_size, 1, -1).float().to(device)  # 将信号转换为张量，并调整为适应cnn输入的形状
        drift_train = baseline_drift.view(batch_size, -1, 1).float().to(device) # 将基线漂移转换为张量
        optimizer.zero_grad()  # 将梯度缓存清零
        outputs = model(signal_train)  # 前向传播，计算模型输出
        loss = criterion(outputs, drift_train)  # 计算损失值
        loss.backward()  # 反向传播，计算梯度
        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3000.0)  # 3/5
        optimizer.step()  # 更新模型参数
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)  # 计算每个epoch的平均损失
    print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}')  # 输出每个epoch的训练损失
    loss_history.append(avg_train_loss)  # 记录每个epoch的训练损失值

    # 如果当前损失比之前最佳损失好，则保存模型权重
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        torch.save(model.state_dict(), 'best_model_weights.pth')
        print(f'Best model saved at epoch {epoch + 1} with loss {best_loss:.6f}')
        no_improve_epoch = 0  # 重置未改善计数器
    else:
        no_improve_epoch += 1  # 增加未改善计数器

    scheduler.step()

    # 早停机制：如果连续若干epoch未改善，则停止训练
    if no_improve_epoch >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

    # 每5个epoch在测试集上测试一次
    if (epoch + 1) % 5 == 0:
        model.eval()  # 设置模型为评估模式
        test_loss = 0.0
        with torch.no_grad():  # 评估过程中不计算梯度
            for batch_data in test_loader:
                signal_with_drift, baseline_drift = batch_data
                signal_test = signal_with_drift.view(batch_size, 1, -1).float().to(device)  # 将信号转换为张量，并调整为适应LSTM输入的形状
                drift_test = baseline_drift.view(batch_size, -1, 1).float().to(device)  # 将基线漂移转换为张量

                outputs = model(signal_test)  # 预测基线漂移
                loss = criterion(outputs, drift_test)  # 计算测试集上的损失
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)  # 计算测试集上的平均损失
        test_loss_history.append(avg_test_loss)  # 记录每个epoch的测试损失值
        print(f'Epoch [{epoch + 1}/{epochs}], Test MSE: {avg_test_loss:.6f}')

# 绘制训练过程中每个epoch的训练和测试损失值折线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b', label='Training Loss')
plt.plot(range(5, len(test_loss_history) * 5 + 1, 5), test_loss_history, marker='o', linestyle='-', color='r', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()  # 显示图形

# 将损失记录写入文件
with open("train_loss_history.txt", "w") as train_file:
    for loss in loss_history:
        train_file.write(f"{loss}\n")

with open("test_loss_history.txt", "w") as test_file:
    for loss in test_loss_history:
        test_file.write(f"{loss}\n")

print("训练和测试损失已写入文件 train_loss_history.txt 和 test_loss_history.txt")