import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pickle


class CausalGate(nn.Module):
    def __init__(self, input_size):
        super(CausalGate, self).__init__()
        # 定义因果门结构的层
        self.gate_vector = nn.Parameter(torch.ones(input_size))

    def forward(self, x):
        # 因果门结构的前向传播
        softmax_vector = F.softmax(self.gate_vector, dim=0)
        return x * softmax_vector


class CGTSTModel(nn.Module):
    def __init__(self, input_size, output_size, in_window_size, out_window_size):
        super(CGTSTModel, self).__init__()
        self.causal_gate = CausalGate(input_size)
        self.linear_transform = nn.Linear(input_size, input_size)
        self.positional_encoding = PositionalEncoding(max_len=in_window_size, d_model=input_size)
        self.embedding = nn.Linear(input_size, 76)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=76, nhead=4), num_layers=3)
        self.fc = nn.Linear(76 * in_window_size, output_size * out_window_size)

    def forward(self, x):
        # 运行因果门结构
        x = self.causal_gate(x)

        # 线性变换和位置编码
        x = self.linear_transform(x)
        x = self.positional_encoding(x)
        x = self.embedding(x)

        # Transformer 编码器层
        x = self.transformer_encoder(x)

        # 平展成一维向量
        x = x.flatten(start_dim=1)

        # 最终的线性层输出
        x = self.fc(x).unsqueeze(2)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pe[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.unsqueeze(0)
        return x


# 创建数据窗口
def create_sequences(data, in_window_size, out_window_size):
    train_sequences = []
    train_targets = []
    test_sequences = []
    test_targets = []
    for i in range(len(data) - in_window_size - out_window_size + 1 - math.ceil(len(data) * 0.2)):
        sequence = data[i:i + in_window_size]
        target = data[i + in_window_size:i + in_window_size + out_window_size]
        train_sequences.append(sequence)
        train_targets.append(target)
    for i in range(len(data) - in_window_size - out_window_size + 1 - math.ceil(len(data) * 0.2),
                   len(data) - in_window_size - out_window_size + 1):
        sequence = data[i:i + in_window_size]
        target = data[i + in_window_size:i + in_window_size + out_window_size]
        test_sequences.append(sequence)
        test_targets.append(target)
    return (torch.stack(train_sequences), torch.stack(train_targets),
            torch.stack(test_sequences), torch.stack(test_targets))


def test_create_sequences(data, in_window_size, out_window_size):
    train_sequences = []
    train_targets = []
    for i in range(len(data) - in_window_size - out_window_size + 1):
        sequence = data[i:i + in_window_size]
        target = data[i + in_window_size:i + in_window_size + out_window_size]
        train_sequences.append(sequence)
        train_targets.append(target)
    return torch.stack(train_sequences), torch.stack(train_targets)


if __name__ == '__main__':
    # 读取化学反应时间序列数据
    file_path = '2.1 new.xlsx'
    xls = pd.ExcelFile(file_path)
    df_data = pd.read_excel(file_path, xls.sheet_names[2])

    # 转换数据为Tensor
    data = torch.tensor(df_data.values, dtype=torch.float32)

    # 划分数据窗口
    in_window_size = 10
    out_window_size = 5
    train_sequences, train_targets_all, test_sequences, test_targets_all = create_sequences(data, in_window_size,
                                                                                            out_window_size)

    test_data = data[len(data) - in_window_size - out_window_size + 1 - math.ceil(len(data) * 0.2):]

    # 拆分单个变量
    train_targets_ones = torch.split(train_targets_all, split_size_or_sections=1, dim=2)

    # 拆分单个变量
    test_targets_ones = torch.split(test_targets_all, split_size_or_sections=1, dim=2)

    # 定义模型和数据路径
    input_size = df_data.shape[1]
    output_size = 1

    # 判断是否可以使用cuda，如果可以就使用cuda，否则使用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'当前使用设备:{device}')
    model = CGTSTModel(input_size, output_size, in_window_size, out_window_size).to(device)

    # 定义训练循环
    epochs = 100
    batch_size = 256

    # 因果门矩阵与潜在邻接矩阵

    Causality_gate_values_over_epochs = []

    for indx in range(1):
        causality_gate_matrix = []
        potential_adjacency_matrix = []
        for idx, train_targets in enumerate(train_targets_ones):
            # 克隆模型，确保每次循环使用独立的模型实例
            current_model = copy.deepcopy(model)
            current_model = current_model.to(device)
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = optim.Adam(current_model.parameters(), lr=0.001)

            # 定义 DataLoader
            train_dataset = TensorDataset(train_sequences.to(device), train_targets.to(device))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

            Causality_gate_values = []

            for epoch in range(epochs):
                total_loss = 0.0
                for batch in train_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()

                    # 前向传播
                    outputs = current_model(inputs)

                    # 反向传播和优化
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                Causality_gate_values.append(current_model.causal_gate.gate_vector.data.cpu().numpy())
                average_loss = total_loss / len(train_loader)
                if epoch % 50 == 0:
                    print(f'[{idx + 1}] Epoch [{epoch + 1}/{epochs}], Loss: {average_loss}')

            # 将因果门矩阵数据转换为 DataFrame
            Causality_gate_values_df = pd.DataFrame(np.array(Causality_gate_values))

            # 保存因果门矩阵数据为 CSV 文件
            if indx == 0:
                Causality_gate_values_df.to_csv(f'Causality_gate_values_{idx + 1}.csv', index=False, header=False)

            Causality_gate_values_over_epochs.append(Causality_gate_values)
            # 在测试集上评估模型
            test_dataset = TensorDataset(test_sequences.to(device), test_targets_ones[idx].to(device))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

            current_model.eval()  # 将模型切换到评估模式

            total_test_loss = 0.0
            with torch.no_grad():
                for batch in train_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)

                    # 前向传播
                    outputs = current_model(inputs)

                    # 计算测试集上的损失
                    test_loss = criterion(outputs, targets)
                    total_test_loss += test_loss.item()

            # 计算平均测试集损失
            average_test_loss = total_test_loss / len(train_loader)
            print(f'Average_test_Loss: {average_test_loss}')

            # 单变量潜在邻接矩阵
            one_potential_adjacency_matrix = []

            for feature_idx in range(input_size):
                temp_data = test_data.clone()
                # 获取第 n 列的数据
                column_to_shuffle = temp_data[:, feature_idx]
                # 生成随机排列的索引
                indices = torch.randperm(len(column_to_shuffle))
                # 使用索引重新排序第 n 列的数据
                shuffled_column = column_to_shuffle[indices]
                # 将打乱后的列放回原始张量中
                temp_data[:, feature_idx] = shuffled_column
                # 划分数据窗口
                temp_sequences, temp_targets_all = test_create_sequences(temp_data, in_window_size,
                                                                               out_window_size)
                # 拆分单个变量
                temp_targets_ones = torch.split(temp_targets_all, split_size_or_sections=1, dim=2)
                # 定义 DataLoader
                temp_dataset = TensorDataset(temp_sequences.to(device), temp_targets_ones[idx].to(device))
                temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

                temp_test_loss = 0.0
                with torch.no_grad():
                    for batch in temp_loader:
                        inputs, targets = batch[0].to(device), batch[1].to(device)

                        # 前向传播
                        outputs = current_model(inputs)

                        # 计算测试集上的损失
                        temp_loss = criterion(outputs, targets)
                        temp_test_loss += temp_loss.item()

                # 计算平均测试集损失
                average_temp_loss = temp_test_loss / len(temp_loader)
                print(f'Average_temp_Loss: {average_temp_loss}')

                # 计算PFI
                potential_adjacency = average_test_loss / average_temp_loss

                # 单行伪因果关系矩阵
                one_potential_adjacency_matrix.append(potential_adjacency)

            # 因果门矩阵
            causality_gate_matrix.append(current_model.causal_gate.gate_vector.data.cpu().numpy())
            # 伪因果关系矩阵
            potential_adjacency_matrix.append(one_potential_adjacency_matrix)

        # 保存变量到文件
        # with open('Causality_gate_values_over_epochs.pkl', 'wb') as file:
        #     pickle.dump(Causality_gate_values_over_epochs, file)

        # 将因果门矩阵数据转换为 DataFrame
        causality_gate_df = pd.DataFrame(np.array(causality_gate_matrix))

        # 保存因果门矩阵数据为 CSV 文件
        causality_gate_df.to_csv(f'{indx + 1}-causality_gate_matrix.csv', index=False, header=False)

        # 将因果门矩阵数据转换为 DataFrame
        potential_adjacency_matrix_df = pd.DataFrame(np.array(potential_adjacency_matrix))

        # 保存因果门矩阵数据为 CSV 文件
        potential_adjacency_matrix_df.to_csv(f'{indx + 1}-PFI.csv', index=False, header=False)

    # # 可视化矩阵
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(causality_gate_matrix, cmap="viridis", annot=True, fmt=".2f", cbar=True)
    # plt.title("Causality Gate Matrix")
    # plt.xlabel("Features")
    # plt.ylabel("Features")
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(potential_adjacency_matrix, cmap="viridis", annot=True, fmt=".2f", cbar=True)
    # plt.title("PFI")
    # plt.xlabel("Features")
    # plt.ylabel("Features")
    # plt.show()
