# CGTST-CausalGate: 因果门控时间序列神经网络
 以化学工业反应为例的因果门控时间序列神经网络

## 简介 (Introduction)

- CGTST (Causal Gate Time Series Transformer) 因果门控时间序列转换器是一种新型的最相关性分析模型。CGTST模型通过因果门控向量值对输入的 N 维变量的重要程度进行评估。这些向量值会随着因果模型的迭代在负无穷到正无穷的区间逐渐收敛。其中，向量值越接近 0 的变量对因果模型的影响越小，而向量值绝对值越大的变量对因果模型的影响越大。

## 安装指南 (Installation Guide)

- 详细可见requirement.txt文档

## 使用说明 (Usage Instructions)

1. 导入所需库和模块：
- `torch`：用于实现神经网络的计算。
- `nn`：提供神经网络构建块。
- `optim`：提供优化算法。
- `matplotlib.pyplot`、`numpy`、`pandas`、`seaborn`：用于数据可视化和分析。
- `pickle`：用于保存和加载模型数据。

2. 定义模型和数据路径：
- 设置数据文件路径和模型参数，如输入大小、输出大小、窗口大小等。

3. 创建数据窗口：
- `create_sequences`函数用于将数据划分为训练和测试窗口。

4. 定义模型架构：
- `CGTSTModel`类继承自`nn.Module`，包含因果门控结构、线性变换、位置编码、Transformer编码器层和输出层。
- `CausalGate`类实现因果门控结构。

5. 训练和测试模型：
- 模型训练分为多个 epoch，每个 epoch 包含多个批次。
- 使用`MSELoss`作为损失函数，`Adam`作为优化器。
- 每个批次的数据通过网络前向传播，计算损失，并进行反向传播和优化。
- 在测试集上评估模型性能。

6. 计算因果关系：
- 对于每个特征，从原始数据中克隆一份数据，用于打乱该特征。
- 生成一个随机的索引排列，将当前特征的列数据按照随机索引重新排序。
- 将打乱后的列数据放回原始数据张量中，形成一个新的打乱后的数据序列。
- 划分数据窗口，得到打乱后的输入序列和目标序列。
- 使用打乱后的输入序列和目标序列进行测试，计算平均测试集损失（average_temp_loss）。
- 计算潜在邻接矩阵（potential_adjacency）：将平均测试集损失（average_test_loss）除以平均打乱后序列的测试集损失（average_temp_loss）。
- 将潜在邻接矩阵值存储到一个列表中。
- 这个过程通过循环遍历每个特征来重复执行，最终得到一个包含所有特征的潜在邻接矩阵。

7. 数据保存和可视化：
- 将训练过程中的因果门矩阵和潜在邻接矩阵保存为 CSV 文件。
- 可视化因果门矩阵和潜在邻接矩阵。

## 功能特性 (Features)

- 详细介绍模型的重要特性和优势。
- 强调因果关系在化学工业反应预测中的应用。

## 数据说明 (Data Description)

- 工业化学反应的时间序列文件，具体格式见test.xlsx文件

## 模型架构 (Model Architecture)

1. CausalGate(nn.Module)：
- 该类继承自nn.Module父类，表示因果门结构的类。
- 初始化函数`__init__`接受输入大小（input_size）作为参数，并调用父类的初始化函数。
- 定义了一个可训练的参数`gate_vector`，它是一个形状为input_size的张量，初始值为全1。
- 前向传播函数`forward`接受输入x，并通过对`gate_vector`进行softmax操作，计算得到softmax_vector。
- 返回输入x与softmax_vector的点乘结果。

2. CGTSTModel(nn.Module)：
- 该类继承自nn.Module父类，表示CGTST-CausalGate模型的类。
- 初始化函数`__init__`接受输入大小（input_size）、输出大小（output_size）、输入窗口大小（in_window_size）和输出窗口大小（out_window_size）作为参数，并调用父类的初始化函数。
- 定义了模型中的各个层和模块：
- `causal_gate`：CausalGate类的实例，用于运行因果门结构。
- `linear_transform`：线性变换层，将输入进行线性变换。
- `positional_encoding`：位置编码层，用于将输入序列进行位置编码。
- `embedding`：线性变换层，将输入进行嵌入操作。
- `transformer_encoder`：Transformer编码器层，利用多层TransformerEncoderLayer对输入进行编码。
- `fc`：线性层，将编码后的输入进行平展并输出结果。
- 前向传播函数`forward`接受输入x，并按照模型的架构依次进行计算和转换操作，最终返回模型的输出。

3. PositionalEncoding(nn.Module)：
- 该类继承自nn.Module父类，表示位置编码层的类。
- 初始化函数`__init__`接受最大序列长度（max_len）和每个位置的编码维度（d_model）作为参数，并调用父类的初始化函数。
- 创建一个形状为max_len × d_model的全零张量pe。
- 通过两层for循环，计算每个位置上的编码值，根据位置的奇偶性使用sin和cos函数进行编码。
- 使用`register_buffer`方法将计算得到的pe作为位置编码张量进行注册。
- 前向传播函数`forward`接受输入x，并将位置编码张量与输入相加，实现位置编码的操作。
- 返回结果。

## 实验结果 (Experimental Results)

- 略（作者懒得找文件了，可以自己跑一遍）

## 引用 (Citation)

- Large-scale chemical process causal discovery from big data with transformer-based deep learning
---
