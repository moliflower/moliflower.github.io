---
layout: page
title: TorchMorph
description: 面向 PyTorch 的 CUDA 加速批量并行形态学变换库。
importance: 1
category: 开源项目
github: https://github.com/intcomp/torchmorph
related_publications: false
---

## 项目简介

[TorchMorph](https://github.com/intcomp/torchmorph) 是一个轻量、可扩展的 PyTorch 形态学操作库。项目通过自定义 CUDA kernel 将形态学变换引入 PyTorch，面向实时和大规模视觉任务提供 GPU 加速与批量并行能力。

我为该项目主要算子贡献者，主要关注 CUDA 算子的实现、性能优化和测试验证。

## 主要特点

- **CUDA 加速**：核心操作由原生 CUDA kernel 实现，以提高计算吞吐。
- **PyTorch 集成**：输入和输出均为 `torch.Tensor`，兼容 CUDA stream 与 PyTorch 工作流。
- **批量并行**：面向大 batch 和多维输入进行并行处理。
- **模块化设计**：不同形态学操作使用相对独立的实现，便于继续扩展。
- **轻量依赖**：核心运行环境主要依赖 PyTorch 与可用的 CUDA 工具链。

## 技术栈

`Python` · `PyTorch` · `C++` · `CUDA` · `GPU Kernel Optimization`

## 项目链接

- [GitHub 仓库](https://github.com/intcomp/torchmorph)
- [查看源代码与最新进展](https://github.com/intcomp/torchmorph/commits/main/)

> 项目仍在持续开发中，后续将根据算子实现和性能测试进展继续更新本页面。
