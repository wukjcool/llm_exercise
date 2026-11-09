# LLM Exercise

大模型核心研发学习实验。目前完成了八节 PyTorch 练习，从线性模型与梯度下降走到下一字符预测和自注意力。

## 当前状态

- [学习笔记](notes/learning-notes.md)收录了 Lesson 1–8 的代码入口、公式和流程图。
- [协作约定与进度](LEARNING.md)记录学习方式和阶段路线。

## VS Code：F5 运行

在 VS Code 中单独打开 `llm_exercise` 文件夹，打开要运行的 Python 文件，保存后按 **F5**。Python 调试器使用本目录的 `.venv/bin/python` 运行当前文件（`${file}`），自动打开 **Debug Console（调试控制台）** 显示 `print()` 输出，支持断点调试。

使用 VS Code 默认的 F5 调试快捷键，之前为本项目添加的 Code Runner F5 绑定已移除。配置见 `.vscode/launch.json`，其中 `console` 为 `internalConsole`；Python 和 Python Debugger 已加入扩展推荐。

需要选择配置时，使用“Python：运行当前文件”。调试控制台不支持程序的 `input()` 交互输入；之后遇到此类程序时再切换到集成终端。

也可以在本仓库目录运行：

```bash
.venv/bin/python lessons/08-attention/train.py
```

## 重建环境

当前验证环境为 Apple Silicon macOS、Python 3.12.14、PyTorch 2.15.0。依赖清单固定了本机安装的版本，未验证其他平台。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r lessons/01-linear-model/requirements-lock.txt
```

本地 `.venv/`、Python 缓存和 `.DS_Store` 已被 Git 忽略。移动或克隆仓库后，按上述命令重建虚拟环境。
