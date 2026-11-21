---
layout: page
title: TorchMorph
description: 面向 PyTorch 的 CUDA 加速批量并行形态学变换库。
importance: 1
category: 开源项目
github: https://github.com/intcomp/torchmorph
docs: https://intcomp.github.io/tm/
related_publications: false
---

## 项目简介

TorchMorph 是一个 PyTorch 扩展，用自定义 CUDA 算子批量处理图像和掩码。我是论文第一作者和项目的主要算子贡献者，负责 CUDA 算子实现、性能优化和测试验证。

## 论文

Congrats! 🎆 [TorchMorph: CUDA-accelerated morphological transforms](https://doi.org/10.1016/j.patrec.2026.11.008) 已在 _Pattern Recognition Letters_ 在线发表。

## 算子与安装

TorchMorph 提供二值与灰度形态学、距离变换和 Sinkhorn 最优传输算子。形态学与距离变换可直接处理 `(B, C, Spatial...)` 格式的 CUDA 张量，支持最多 8 个空间维度。

从 [GitHub 仓库]({{ page.github }}) 获取源码，按 [README]({{ page.github }}/blob/main/README.md) 编译 CUDA 扩展。请使用与 PyTorch CUDA 版本兼容的 `nvcc`。算子和参数见 [API 文档]({{ page.docs }})。
