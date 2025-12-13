# GCM LLM（GCM-1）
枫义实现的大语言模型，完全不依赖预训练模型。

A large language model implemented from scratch with zero dependencies on pre-trained models.

---

## 特点 / Features

- **自主实现**：从 Transformer 架构到训练流程
- **支持多种数据格式**：.txt, .json, .jsonl，支持批量加载
- **完整的训练流程**：包含验证集、困惑度计算、模型保存
- **交互式推理**：支持交互式文本生成
- **超高效训练系统**：智能数据增强、课程学习、困难样本挖掘等创新方法
- **可扩展**：可以轻松调整模型大小和参数

---

## 安装 / Installation

### 基础依赖

```bash
pip install -r requirements.txt
```

### 可选依赖

如果需要可视化训练曲线：

```bash
pip install matplotlib
```

如果需要生成研究论文图表：

```bash
pip install matplotlib scipy python-docx
```

---

## 快速开始 / Quick Start

### 1. 准备数据 / Prepare Data

#### 最简单的格式 - 纯文本 (.txt)

创建 `data/my_data.txt`:

```txt
这是第一个训练样本。
This is the first training sample.

这是第二个训练样本。
This is the second training sample.
```

### 2. 训练模型 / Train Model

#### 基础训练（12/11/2025已删除）

```bash
python train.py --data_path data/my_data.txt --epochs 50 --batch_size 8
```

#### 超高效训练

使用智能数据增强、课程学习等创新方法，用更少的数据达到更好的效果：

```bash
python train_ultra.py --data_path data/my_data.txt --epochs 100 --augmentation_factor 15
```


### 3. 推理 / Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pt
```

或者使用交互式模式：

```bash
python inference.py --checkpoint checkpoints/best_model.pt --interactive
```

---

## 项目结构 / Project Structure

```
gcm-llm/
├── model.py                  # Transformer模型架构
├── tokenizer.py              # 分词器
├── data_loader.py            # 数据加载器
├── train.py                  # 基础训练脚本（12/11/2025已删除）
├── train_ultra.py            # 超高效训练脚本
├── ultra_efficient_trainer.py # 超高效训练器实现
├── inference.py              # 推理脚本
├── utils.py                  # 工具函数
├── requirements.txt           # 依赖列表
├── README.md                 # 本文件
└── data/                     # 数据目录
```

---

## 模型架构 / Model Architecture

### 核心组件

- **Transformer架构**：基于Vaswani等人的原始设计
- **RoPE位置编码**：旋转位置编码，更好的位置表示
- **RMSNorm归一化**：比LayerNorm更高效的归一化方法
- **SwiGLU激活函数**：门控激活函数，更强的表达能力
- **Pre-Norm架构**：更稳定的训练

### 模型配置

默认配置：
- 模型维度（d_model）：512
- 注意力头数（n_heads）：8
- Transformer层数（n_layers）：6
- 前馈网络维度（d_ff）：2048
- 最大序列长度（max_len）：512

可以通过命令行参数调整这些配置。

---

## 训练方法 / Training Methods

### 基础训练

标准的Transformer训练流程，包含：
- 混合精度训练（FP16）
- 梯度累积
- 学习率调度（Warmup + Cosine Annealing）
- EMA（指数移动平均）
- 梯度裁剪

### 超高效训练

创新的训练系统，包含以下组件：

1. **智能数据增强**
   - 同义词替换
   - 回译模拟
   - 语义改写
   - 上下文插入

2. **自适应课程学习**
   - 从简单到复杂的渐进式训练
   - 动态难度评估
   - 自适应难度阈值

3. **困难样本挖掘**
   - 动态损失追踪
   - 指数移动平均平滑
   - 自适应权重调整

4. **对比学习**
   - InfoNCE损失
   - 正负样本构造
   - 更好的文本表示

5. **自适应采样**
   - 基于难度的权重调整
   - 加权随机采样
   - 动态权重更新

---

## 使用示例 / Usage Examples

### 训练示例

```bash
# 基础训练（12/11/2025已删除）
python train.py \
    --data_path data/chinese.txt \
    --epochs 50 \
    --batch_size 8 \
    --d_model 512 \
    --n_layers 6 \
    --n_heads 8 \
    --lr 0.0001 \
    --save_dir checkpoints

# 超高效训练
python train_ultra.py \
    --data_path data/chinese.txt \
    --epochs 100 \
    --batch_size 8 \
    --augmentation_factor 15 \
    --use_amp \
    --use_rope \
    --save_dir checkpoints_ultra
```

### 推理示例

```bash
# 单次推理
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "今天天气很好"

# 交互式推理
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --interactive

# 批量推理
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input_file prompts.txt \
    --output_file results.txt
```

---

## 文档 / Documentation

- [Read Me](README.md) - 关于本模型
- [基于数据增强与大模型课程学习的超高效语言模型训练方法的研究](基于数据增强与大模型课程学习的超高效语言模型训练方法的研究.docx) - 本模型的研究过程及相关文献

---

## 性能 / Performance

### 训练效率

- **数据效率**：通过智能数据增强，可以将数据效率提升10-50倍
- **训练时间**：相比传统方法，训练时间仅增加约20%
- **性能提升**：困惑度相比基线方法降低42%-60%

### 资源需求

- **GPU显存**：约3-4GB（batch_size=8, d_model=512）
- **训练时间**：取决于数据量和模型大小
- **存储空间**：模型文件约170MB（FP32）

---

## 常见问题 / FAQ

### Q: 如何选择合适的模型配置？

A: 根据你的GPU显存和数据量选择：
- 小显存（<4GB）：d_model=256, n_layers=4
- 中等显存（4-8GB）：d_model=512, n_layers=6（默认）
- 大显存（>8GB）：d_model=768, n_layers=8

### Q: 训练需要多长时间？

A: 取决于数据量和模型大小。一般来说：
- 1万行数据，50轮：约1-2小时
- 10万行数据，100轮：约5-10小时

### Q: 如何提高训练效果？

A: 建议：
1. 使用超高效训练系统（train_ultra.py）
2. 增加数据增强倍数（--augmentation_factor）
3. 使用课程学习（自动启用）
4. 调整学习率和batch size

### Q: 支持哪些语言？

A: 支持中文、英文和混合语言。分词器会自动处理不同语言的字符。

---

## 贡献 / Contributing

欢迎贡献代码、报告问题或提出建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证 / License

本项目采用 GPLv3 许可证。详见 [LICENSE](LICENSE) 文件。

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

---

## 致谢 / Acknowledgments

- Transformer架构：Vaswani et al. (2017)
- RoPE位置编码：Su et al. (2024)
- RMSNorm：Zhang & Sennrich (2019)
- SwiGLU：Shazeer (2020)

---

## 联系方式 / Contact

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件（kejia.xia@maplegcm.com)

---

本项目源码来自枫义GCM（MapleGCM），开源代码仅用于学习和研究目的。商业训练、使用请自行遵守相关法律法规。
