# TS100 iTransformer 运行说明

## 目录说明

- `train/train.csv`：训练数据，形状为 `20000 x 100`。
- `test_demo/hist_96.npy`：默认测试输入，形状为 `250 x 96 x 100`。
- `model/`、`layers/`、`utils/`：从 iTransformer 参考工程复制的模型代码。
- `data_provider/data_loader_ts.py`：本任务专用数据读取和标准化代码。
- `train_ts.py`：训练四个预测长度模型。
- `predict_ts.py`：生成四个提交预测文件。
- `design.md`：一页设计思路。

## 环境依赖

建议使用 Python 3.8 或以上版本，并安装：

```bash
pip install torch numpy pandas scikit-learn
```

## 训练

在 `TS` 目录下执行：

```bash
python train_ts.py
```

默认配置面向 A6000 这类 48GB 显存显卡：

- `batch_size=128`
- `eval_batch_size=256`
- `train_epochs=40`
- `patience=8`
- `d_model=768`
- `n_heads=12`
- `e_layers=3`
- `d_ff=3072`
- 默认开启 AMP 混合精度

默认会训练四个模型：

- `pred_len=96`
- `pred_len=192`
- `pred_len=336`
- `pred_len=720`

模型会保存到：

```text
checkpoints/itransformer_pl96/
checkpoints/itransformer_pl192/
checkpoints/itransformer_pl336/
checkpoints/itransformer_pl720/
```

每个目录包含：

- `checkpoint.pth`：验证集 MSE 最低的模型权重。
- `scaler.npz`：训练集标准化参数。
- `config.json`：模型配置。
- `history.json`：训练和验证 MSE 记录。

显存不足时可以使用较小模型：

```bash
python train_ts.py --d_model 256 --d_ff 1024 --batch_size 16 --no_amp
```

如果 A6000 显存仍有较多空余，可以继续增大 batch：

```bash
python train_ts.py --batch_size 192 --eval_batch_size 384
```

只训练某个预测长度：

```bash
python train_ts.py --pred_lens 96
```

## 生成预测文件

训练完成后执行：

```bash
python predict_ts.py
```

默认读取：

```text
test_demo/hist_96.npy
```

并输出到：

```text
predictions/pred_96.npy
predictions/pred_192.npy
predictions/pred_336.npy
predictions/pred_720.npy
```

如果现场测试文件路径不同：

```bash
python predict_ts.py --test_npy path/to/test.npy --output_dir predictions
```

## 验证集评估

训练完成后可以重新读取四个 checkpoint，在本地验证集上统一计算 MSE：

```bash
python eval_ts.py
```

输出格式类似：

```text
MSE 96: ...
MSE 192: ...
MSE 336: ...
MSE 720: ...
MSE Avg: ...
```

现场评分重点看四个长度的平均 MSE。根据验收标准，平均 MSE 小于 `0.01` 是主要得分线，小于 `0.006` 和 `0.005` 有额外加分。

## 打包提交

先确认 `predictions` 目录下已有四个预测文件，然后执行：

```bash
python make_submission.py --name "时间序列-组员1(姓名+学号)-组员2(姓名+学号)"
```

会生成：

```text
时间序列-组员1(姓名+学号)-组员2(姓名+学号).zip
```

压缩包中包含四个预测结果、完整代码、设计思路、训练配置和 checkpoint 目录。

## 评价方式

评价方式与 `test_demo/eval.ipynb` 一致：

```python
import numpy as np

def MSE(pred, true):
    return np.mean(np.square(true - pred))

score = (mse_96 + mse_192 + mse_336 + mse_720) / 4
```
