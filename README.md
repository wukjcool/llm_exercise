# LLM Exercise

大模型核心研发学习实验。当前仅包含第一节：用 PyTorch 训练线性模型，学习自动微分与梯度下降。

## 快速运行

在本仓库目录运行；本机已经配置好 `.venv`，可直接执行：

```bash
.venv/bin/python train.py
```

或激活环境后运行：

```bash
source .venv/bin/activate
python train.py
```

## VS Code：F5 运行

在 VS Code 中单独打开 `llm_exercise` 文件夹，按 **F5**，使用“第一节：训练线性模型”配置。程序固定运行 `train.py`，使用本目录的 `.venv/bin/python`，输出显示在集成终端；可在代码行旁设置断点。

需要 Microsoft Python 和 Python Debugger 扩展，本仓库已添加扩展推荐。调试配置只进入自己的代码，不单步进入 PyTorch 内部实现。配置方式见 [VS Code 官方文档](https://code.visualstudio.com/docs/python/debugging)。

## 重建环境

当前验证环境为 Apple Silicon macOS、Python 3.12.14、PyTorch 2.15.0。依赖清单固定了本机安装的版本，未验证其他平台。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r lessons/01-linear-model/requirements-lock.txt
.venv/bin/python lessons/01-linear-model/check_environment.py
```

虚拟环境不适合直接复制到其他路径或提交 Git；移动或克隆仓库后，按上述命令重建。

## 实验与记录

- [训练代码](train.py)：默认执行 101 次更新，逐步记录参数和损失。
- [第一节笔记](lessons/01-linear-model/README.md)：数学推导、实验结果、复现命令与待解问题。
- [协作约定与学习路线](LEARNING.md)：目标、学习方式与当前进度。
- `lessons/01-linear-model/results/`：CSV、配置摘要、运行日志和曲线。

训练、对照和绘图可分别运行：

```bash
.venv/bin/python train.py
.venv/bin/python lessons/01-linear-model/experiments.py
.venv/bin/python lessons/01-linear-model/plot.py
```

这些命令会更新对应的结果文件。本地 `.venv/`、Python 缓存和 `.DS_Store` 已被 Git 忽略；代码、锁定依赖和实验记录纳入版本管理。
