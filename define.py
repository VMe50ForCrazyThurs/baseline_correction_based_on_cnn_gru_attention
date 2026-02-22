# 定义损失函数及相关模型
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        rmse= torch.sqrt(mse)
        mae= self.mae_loss(output, target)
        return mse

class CNNGRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(CNNGRUModel, self).__init__()
        # CNN 层
        # 第一卷积层，输入通道数为1（假设是单通道信号），输出通道数为16，卷积核大小为3，使用padding=1以保持输入和输出的长度相同
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 将卷积输出展平为一维，输入维度为 32 * 64（假设输入序列长度为256，经过两次池化后长度变为64），输出维度为128
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 32, 256),
            nn.Dropout(0),
        )
        # 将 输出的特征映射到输出序列，输出维度为 256（输出序列长度为256）
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 第一层 GRU：输入尺寸：(32, 10, 1)。输出尺寸：(32, 10, 64)
        # 第二层 GRU：输入尺寸：(32, 10, 64)（第一层输出的结果），输出尺寸：(32, 10, 64)
        # 最终，经过两层 GRU 的处理，输出数据的尺寸为 (batch_size, sequence_length, hidden_size)，即 (32, 10, 64)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.fc2 = nn.Sequential(nn.Linear(256, 256))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        # 由于卷积和池化操作使得数据形状变为 (batch_size, channels, length)，需要展平成 (batch_size, -1) 进行全连接层计算
        x = x.view(x.size(0), -1)
        # 通过全连接层 1，激活函数为 ReLU
        x = self.fc1(x)
        # gru的输入需要是三维的 (batch_size, sequence_length, input_size)，这里增加一个时间维度
        x = x.unsqueeze(2)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        x, _ = self.gru(x, h0)
        # 应用层归一化
        x = self.layer_norm(x)
        # 对每个时间步的输出通过全连接层，得到长度为256的输出序列
        # (batch_size, l, h)--(b,l)
        x=torch.sum(x, dim=-1)
        x = self.fc2(x)
        return x.unsqueeze(-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用 Kaiming 正态分布初始化卷积层的权重，以适应 ReLU 激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果存在偏置，则将偏置初始化为 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用 Kaiming 均匀分布初始化全连接层的权重
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                # 如果存在偏置，则计算合适的边界范围并使用均匀分布进行初始化
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.GRU):
                # 对 GRU 层的权重进行初始化
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        # 使用 Xavier 均匀分布初始化输入到隐藏层的权重
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # 使用正交矩阵初始化隐藏到隐藏层的权重，确保梯度的稳定性
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        # 将偏置初始化为 0
                        nn.init.constant_(param.data, 0)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(CNNLSTMModel, self).__init__()
        # CNN 层
        # 第一卷积层，输入通道数为1（假设是单通道信号），输出通道数为16，卷积核大小为3，使用padding=1以保持输入和输出的长度相同
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 将卷积输出展平为一维，输入维度为 32 * 64（假设输入序列长度为256，经过两次池化后长度变为64），输出维度为128
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 32, 256),
            nn.Dropout(0),
        )
        # LSTM 层
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.fc2 = nn.Sequential(nn.Linear(256,256))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        # 由于卷积和池化操作使得数据形状变为 (batch_size, channels, length)，展平成 (batch_size, -1) 进行全连接层计算
        x = x.view(x.size(0), -1)
        # 通过全连接层 1，激活函数为 ReLU
        x = self.fc1(x)
        # LSTM 的输入需要是三维的 (batch_size, sequence_length, input_size)，这里增加一个时间维度
        x = x.unsqueeze(2)
        # 初始化 LSTM 的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        # 应用层归一化
        x = self.layer_norm(x)
        # 对每个时间步的输出通过全连接层，得到长度为256的输出序列
        # (batch_size, l, h)--(b,l)
        x = torch.sum(x, dim=-1)
        x = self.fc2(x)
        return x.unsqueeze(-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用 Kaiming 正态分布初始化卷积层的权重，以适应 ReLU 激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果存在偏置，则将偏置初始化为 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用 Kaiming 均匀分布初始化全连接层的权重
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                # 如果存在偏置，则计算合适的边界范围并使用均匀分布进行初始化
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.LSTM):
                # 对 LSTM 层的权重进行初始化
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        # 使用 Xavier 均匀分布初始化输入到隐藏层的权重
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # 使用正交矩阵初始化隐藏到隐藏层的权重，确保梯度的稳定性
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        # 将偏置初始化为 0
                        nn.init.constant_(param.data, 0)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        """
        参数:
            hidden_size: 编码器输出和解码器隐藏状态的维度
            attention_size: 注意力机制中间表示的维度
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, attention_size)  # 用于编码器输出的全连接层
        self.W2 = nn.Linear(hidden_size, attention_size)  # 用于解码器隐藏状态的全连接层
        self.V = nn.Linear(attention_size, 1)             # 输出最终得分

    def forward(self, features, hidden):
        """
        参数:
            x---features:  shape = (batch_size, seq_len, hidden_size)
            q---hidden:  shape = (batch_size, hidden_size)
        返回:
            context_vector: 加权求和后的上下文向量, shape = (batch_size, hidden_size)
            attention_weights: 注意力权重, shape = (batch_size, seq_len)
        """
        # 扩展维度以便广播: hidden -> (batch_size, 1, hidden_size)
        hidden_with_time_axis = hidden.unsqueeze(1)

        # 计算注意力得分：
        # W1(features): (batch_size, seq_len, attention_size)
        # W2(hidden_with_time_axis): (batch_size, 1, attention_size)，广播到 (batch_size, seq_len, attention_size)
        score = torch.tanh(self.W1(features) + self.W2(hidden_with_time_axis))  # (batch_size, seq_len, attention_size)

        # 计算注意力权重：
        # V(score): (batch_size, seq_len, 1) → squeeze(-1): (batch_size, seq_len)
        attention_weights = F.softmax(self.V(score).squeeze(-1), dim=1)  # (batch_size, seq_len)

        # 计算上下文向量：
        # features: (batch_size, seq_len, hidden_size)
        # attention_weights.unsqueeze(-1): (batch_size, seq_len, 1)
        # 广播相乘后：context_vector = (batch_size, seq_len, hidden_size)
        # 再 sum(dim=1) 得到 (batch_size, seq_len)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * features, dim=2)
        # context_vector = attention_weights.unsqueeze(-1) * features
        # (batch_size, seq_len, 1)
        return context_vector.unsqueeze(-1)
        # return context_vector

class CNNGRUAttentionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(CNNGRUAttentionModel, self).__init__()
        # CNN 层
        # 第一卷积层，输入通道数为1（假设是单通道信号），输出通道数为16，卷积核大小为3，使用padding=1以保持输入和输出的长度相同
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 将卷积输出展平为一维，输入维度为 32 * 64（假设输入序列长度为256，经过两次池化后长度变为64），输出维度为128
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 32, 256),
            nn.Dropout(0),
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 第一层 GRU：输入尺寸：(32, 10, 1)。输出尺寸：(32, 10, 64)
        # 第二层 GRU：输入尺寸：(32, 10, 64)（第一层输出的结果），输出尺寸：(32, 10, 64)
        # 最终，经过两层 GRU 的处理，输出数据的尺寸为 (batch_size, sequence_length, hidden_size)，即 (32, 10, 64)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)  # 添加层归一化

        # Attention 机制
        attention_size = int(hidden_size/2)  # 或根据需要调整
        self.bahdanau_attention = BahdanauAttention(hidden_size, attention_size)
        self.fc2 = nn.Sequential(nn.Linear(256, 256))
        # 初始化模型参数
        self._initialize_weights()

    def forward(self, x):
        # 适合Conv1d的形状: (batch_size, feature_size, sequence_length)
        # query = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        # 由于卷积和池化操作使得数据形状变为 (batch_size, channels, length)，这里需要展平成 (batch_size, -1) 进行全连接层计算
        x = x.view(x.size(0), -1)
        # 通过全连接层 1，激活函数为 ReLU
        x = self.fc1(x)
        # LSTM 的输入需要是三维的 (batch_size, sequence_length, input_size)，这里增加一个时间维度
        x = x.unsqueeze(2)
        # 初始化 LSTM 的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # 前向传播 GRU
        x, h_n = self.gru(x, h0)
        # 应用层归一化
        x = self.layer_norm(x)
        # (batch_size, hidden_size)
        query = h_n[-1]

        # 加性注意力
        # x ：[batch_size, seq_len, hidden_size]
        context = self.bahdanau_attention(x, query)

        # ---- 全连接输出 ----
        out = self.fc2(context.transpose(1, 2))  # [batch_size, output_size]
        return out.transpose(1, 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用 Kaiming 正态分布初始化卷积层的权重，以适应 ReLU 激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果存在偏置，则将偏置初始化为 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用 Kaiming 均匀分布初始化全连接层的权重
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                # 如果存在偏置，则计算合适的边界范围并使用均匀分布进行初始化
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.GRU):
                # 对 GRU 层的权重进行初始化
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        # 使用 Xavier 均匀分布初始化输入到隐藏层的权重
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # 使用正交矩阵初始化隐藏到隐藏层的权重，确保梯度的稳定性
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        # 将偏置初始化为 0
                        nn.init.constant_(param.data, 0)


