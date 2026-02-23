# baseline_correction_based_on_cnn_gru_attention
Baseline correction for biphoton Hong–Ou–Mandel interference curves using a CNN–GRU–Attention model trained on synthetic drift/noise patterns and validated on experimental coincidence counts.

data_generate.py：用于生成仿真数据集

define.py：定义了损失函数和三种模型结构，cnn-gru, cnn-lstm, cnn-gru-attention

train&test.py：加载训练集及测试集，训练并每5个epoch测试一次，保存最佳权重及损失曲线

test.py：加载最佳权重及测试集，对测试集上的loss进行计算

simple_visualization.py：加载最佳权重及测试集，对测试集上的指定样本进行预测并画图展示

generated_data_4000.pt：生成的训练集，包含4000个样本

generated_data_1000.pt：生成的测试集，包含1000个样本

best_model_weights_cnn_gru.pth、best_model_weights_cnn_lstm.pth、best_model_weights_cnn_gru_att.pth：训练过程中在训练集上loss最小的一组权重

train_loss_history_cnn_gru.txt、train_loss_history_cnn_lstm.txt、train_loss_history_cnn_gru_att.txt：训练过程中，每个epoch在训练集上的loss

test_loss_history_cnn_gru.txt、test_loss_history_cnn_lstm.txt、test_loss_history_cnn_gru_att.txt：训练过程中，每5个epoch在测试集上的loss




