# PCNet 统一模型

PCNet是一个统一的时序预测模型框架，集成了多种模型变体，通过配置参数即可切换不同的模型模式。

## 支持的模型模式

| 模式 | 全称 | 特点 |
|------|------|------|
| DPCF | Dynamic Periodic Cross Forecasting | 动态周期模板 + 交叉注意力 + 拼接预测 |
| DPRF | Dynamic Periodic Residual Forecasting | 动态周期模板 + 交叉注意力 + 残差预测 |
| FPCF | Fixed Periodic Cross Forecasting | 固定周期模板 + 拼接预测 |
| FPRF | Fixed Periodic Residual Forecasting | 固定周期模板 + 残差预测 |
| MLP | Multi-Layer Perceptron | 简单MLP基线模型 |

## 使用方法

### 基本用法

```bash
python -u run.py \
  --is_training 1 \
  --model PCNet \
  --pcnet_mode DPCF \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --cycle 24 \
  --template_num 8 \
  --d_model 512 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 0.001
```

### 切换模型模式

只需修改 `--pcnet_mode` 参数：

```bash
# 使用DPRF模式
--pcnet_mode DPRF

# 使用FPCF模式
--pcnet_mode FPCF

# 使用FPRF模式
--pcnet_mode FPRF

# 使用MLP模式
--pcnet_mode MLP
```

## 模型架构差异

### 1. 模板生成方式

- **Dynamic (DPCF, DPRF)**: 使用SimpleCrossAttention动态生成模板
- **Fixed (FPCF, FPRF, MLP)**: 使用固定的可学习模板参数

### 2. 预测方式

- **Cross Forecasting (DPCF, FPCF)**: 输入与模板拼接后预测
  ```
  input = concat(x_input, template)
  output = MLP(input)
  ```

- **Residual Forecasting (DPRF, FPRF)**: 预测残差后加上模板
  ```
  input = x_input - template[:seq_len]
  output = MLP(input) + template[seq_len:]
  ```

- **MLP**: 直接预测
  ```
  input = x_input
  output = MLP(input)
  ```

## 脚本示例

参考 `scripts/PCNet/etth1.sh` 查看完整的实验脚本示例。

## 向后兼容

原有的独立模型文件仍然可用：
- `models/DPCF.py` - 可通过 `--model DPCF` 使用
- `models/DPRF.py` - 可通过 `--model DPRF` 使用
- `models/FPCF.py` - 可通过 `--model FPCF` 使用
- `models/FPRF.py` - 可通过 `--model FPRF` 使用
- `models/MLP.py` - 可通过 `--model MLP` 使用