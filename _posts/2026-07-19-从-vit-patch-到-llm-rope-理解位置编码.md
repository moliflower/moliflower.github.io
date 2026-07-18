---
layout: post
title: 从 ViT 的 Patch 到 LLM 的 RoPE：理解位置编码
date: 2026-07-19 00:31:29 +0800
description: 从注意力的置换等变性出发，理解绝对位置、相对位置、ViT 的 patch 位置嵌入与大语言模型中的旋转位置编码 RoPE。
tags: [Transformer, ViT, 位置编码, RoPE]
categories: [技术笔记]
related_posts: false
---

最近看 Vision Transformer（ViT）时，我又遇到了 patch 的位置编码；而之前看大语言模型时，也经常看到 RoPE。它们一个处理图像，一个处理文本，写法看起来也不一样，但解决的其实是同一个问题：**Transformer 只看内容之间是否匹配，并不会天然理解这些内容原本排在什么位置。**

这篇文章从 Self-Attention 的性质出发，依次整理绝对位置编码、相对位置编码、ViT 中的二维 patch 位置，以及 LLM 中常用的旋转位置编码 RoPE。重点不是记住每一种公式，而是理解：位置信息究竟在注意力计算的哪一步发挥作用。

## 1. 为什么 Self-Attention 不认识顺序

对输入序列 $X$，Self-Attention 的核心计算是：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中每个 token 都使用同一组 $W_Q$、$W_K$ 和 $W_V$。如果同时打乱输入 token 的排列，输出也只会跟着做相同的排列。换句话说，Self-Attention 对输入顺序具有**置换等变性**：它擅长比较“谁和谁相关”，但公式本身没有“第几个”的概念。

这会带来两个直观问题：

- 在文本中，“人咬狗”和“狗咬人”包含相同的字，但语义完全不同；
- 在图像中，蓝色 patch 出现在上方可能是天空，出现在下方则可能是水面。

因此，送入 Transformer 的表示不能只有内容，还需要某种位置信号。可以先用一句话记住：

> token embedding 回答“这是什么”，position embedding 回答“它在哪里”。

对于带因果掩码的 LLM，mask 已经区分了“过去”和“未来”，但它仍没有充分表达两个历史 token 相隔多远，所以通常仍然需要专门的位置编码。

## 2. 位置信息可以加在哪里

位置编码并不只有一种固定形式。按照它进入模型的位置，可以先分成三类：

| 类型          | 典型形式                      | 直观理解                          |
| ------------- | ----------------------------- | --------------------------------- |
| 绝对位置      | $z_i=x_i+p_i$                 | 给每个位置发一个座位号            |
| 相对位置      | $s_{ij}=q_i^\top k_j+b_{i-j}$ | 直接告诉注意力两者相隔多远        |
| 旋转位置 RoPE | $q_i'=R_iq_i,\ k_j'=R_jk_j$   | 按位置旋转 Q、K，用夹角差表示距离 |

绝对位置主要修改 Transformer 的输入；相对位置通常直接修改注意力分数；RoPE 则处在两者之间：它用绝对位置旋转每个向量，却让 Q、K 的点积呈现出相对位置。

## 3. 绝对位置编码：给每个 token 一个坐标

### 3.1 可学习的位置嵌入

最直接的方式是准备一个可训练矩阵：

$$
P\in\mathbb{R}^{L_{\max}\times d}
$$

第 $i$ 个 token 的输入变为：

$$
z_i=x_i+P_i
$$

模型在训练中自己学习每个位置应该是什么向量。这种方法简单、有效，BERT 和原始 ViT 都采用过类似思路。不过，位置表的大小与最大长度绑定；遇到训练范围以外的位置时，不能直接查到新的向量，通常需要插值、扩充或重新训练。

### 3.2 正弦与余弦位置编码

原始 Transformer 使用固定的正弦、余弦函数：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

不同维度对应不同频率：高频维度对短距离变化敏感，低频维度变化更慢，可以覆盖更长的尺度。这很像用许多快慢不同的指针共同描述一个位置。

正弦编码还有一个重要性质。根据三角函数的和角公式，$PE(pos+\Delta)$ 可以由 $PE(pos)$ 经过一个只与位移 $\Delta$ 有关的线性变换得到。因此它虽然写入的是绝对位置，却给模型比较相对位移提供了方便的结构。

固定公式能够计算训练长度之外的位置，但这只表示“编码有定义”，并不保证模型一定能在更长序列上正常泛化。模型是否学会长度外推，还与训练分布、注意力模式和具体任务有关。

## 4. ViT：Patch 有内容，但它原本在图像哪里

ViT 先把一张大小为 $H\times W\times C$ 的图像切成 $P\times P$ 的小块，patch 数量为：

$$
N=\frac{HW}{P^2}
$$

每个 patch 被展平并线性投影成一个 token。加入分类 token 后，原始 ViT 的输入可以写成：

$$
z_0=
\left[x_{\mathrm{class}};
x_p^1E;
x_p^2E;
\cdots;
x_p^NE\right]+E_{pos}
$$

其中：

- $x_p^iE$ 是第 $i$ 个 patch 的内容表示；
- $E_{pos}\in\mathbb{R}^{(N+1)\times d}$ 是可学习的位置嵌入；
- 额外的一行位置向量对应分类 token。

这里最容易混淆的是：**patch embedding 能编码 patch 内部的像素内容，但不会自动说明这个 patch 位于整张图的左上角还是右下角。**把 patch 按行展平成序列只是数据排布方式；如果没有位置编码，后面的标准 Self-Attention 仍然无法把序号理解为空间坐标。

原始 ViT 使用的是可学习的一维位置嵌入。虽然形式上只是把二维网格按行编号，论文中的可视化发现，训练后的位置向量会呈现出二维结构：距离较近的 patch 往往具有更相似的位置向量，同一行或同一列也会表现出相似性。也就是说，模型可以从任务中重新学出图像网格的拓扑关系。

### 分辨率变化时为什么要插值

假设训练时图像被切成 $14\times14$ 个 patch，位置表中就有 $196$ 个 patch 位置。微调时如果分辨率变大，网格可能变成 $24\times24$，原来的位置表不够用了。

常见做法不是修改 patch 的内容投影，而是把 patch 对应的位置向量恢复成二维网格，对它做双线性或双三次插值，再展平成新的序列。分类 token 的位置向量通常单独保留。这里被插值的是**坐标表**，不是原始图片，也不是 patch token 本身。

## 5. 相对位置编码：很多关系只与距离有关

在语言中，“前一个词”“相隔三步”“同一句中的附近词”往往比“绝对位于第 137 个 token”更有意义。在图像中，物体的局部结构也常常与整体平移无关。

一种直接的方法是在注意力分数中加入相对位置偏置：

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}+b_{i-j}
$$

$b_{i-j}$ 只由两个 token 的相对位移决定。这样，同一种局部关系可以出现在序列的不同位置，模型不必分别学习“位置 5 到位置 6”和“位置 105 到位置 106”。

实际模型中，相对距离可能被裁剪或分桶。例如很远的距离可以共享一个桶，因为模型通常不需要精确区分“相隔 1001”和“相隔 1002”。二维视觉模型还可以把行位移与列位移分开编码。

绝对位置与相对位置并不是非此即彼。绝对位置能表达“位于开头”“靠近图像边缘”，相对位置擅长表达局部结构；不少模型会通过不同方式同时保留两类信息。

## 6. RoPE：把相对距离变成旋转角之差

RoPE（Rotary Position Embedding）的核心想法非常漂亮：把向量的维度两两配对，在每个二维平面中按照位置旋转不同角度。

对位置 $m$ 的 query 和位置 $n$ 的 key，有：

$$
q_m'=R_mq_m,\qquad k_n'=R_nk_n
$$

其中 $R_m$、$R_n$ 是旋转矩阵。计算注意力点积时：

$$
\begin{aligned}
(q_m')^\top k_n'
&=(R_mq_m)^\top(R_nk_n)\\
&=q_m^\top R_m^\top R_nk_n\\
&=q_m^\top R_{n-m}k_n
\end{aligned}
$$

最后只剩下 $n-m$。这就是 RoPE 最关键的性质：**对单个 Q、K 来说，它们按照各自的绝对位置旋转；进入点积后，真正影响注意力关系的却是相对位置。**

RoPE 通常只作用于 Q 和 K，而不作用于 V，因为位置主要需要影响“应该关注谁”的权重。旋转还会保持向量范数不变，因此不会仅仅因为位置变大就直接放大或缩小 Q、K。

### 6.1 为什么要有许多旋转频率

如果所有二维子空间都使用同一个角速度，位置表示很快就会周期性重复。RoPE 因此让不同维度对使用不同频率：

$$
\theta_i=10000^{-2i/d}
$$

位置 $m$ 对应的旋转角为 $m\theta_i$。前面的维度旋转较快，容易分辨邻近位置；后面的维度旋转较慢，能够表示更长尺度的变化。这与正弦位置编码的多频率思想是一脉相承的，只是 RoPE 没有把正弦、余弦向量直接加到 token 上，而是用它们旋转 Q、K。

### 6.2 一个简化的 PyTorch 实现

下面的代码采用相邻两维组成一对的写法。实际模型还需要处理不同的张量布局、缓存、缩放策略以及部分维度旋转等细节。

```python
import torch


def build_rope_cache(seq_len, head_dim, device):
    assert head_dim % 2 == 0

    pair_index = torch.arange(0, head_dim, 2, device=device)
    inv_freq = 10000 ** (-pair_index / head_dim)
    position = torch.arange(seq_len, device=device)
    angles = torch.outer(position, inv_freq)

    # 每一对相邻维度共享同一个旋转角
    angles = torch.repeat_interleave(angles, 2, dim=-1)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return cos, sin


def rotate_pairs(x):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    return x * cos + rotate_pairs(x) * sin


# q, k 的形状为 [batch, heads, seq_len, head_dim]
cos, sin = build_rope_cache(q.shape[-2], q.shape[-1], q.device)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

代码中的 `x * cos + rotate_pairs(x) * sin` 正是二维旋转：若一对维度是 $(x_1,x_2)$，旋转后就是
$(x_1\cos\theta-x_2\sin\theta,\ x_2\cos\theta+x_1\sin\theta)$。

## 7. ViT 的位置嵌入和 LLM 的 RoPE 有什么不同

原始 ViT 与现代 LLM 的常见选择可以这样对照：

| 问题             | 原始 ViT             | 使用 RoPE 的 LLM               |
| ---------------- | -------------------- | ------------------------------ |
| token 是什么     | 图像 patch           | 文本 token                     |
| 原始几何         | 二维网格             | 一维序列                       |
| 典型位置方案     | 可学习的绝对位置向量 | 对 Q、K 做多频率旋转           |
| 位置信息影响哪里 | Transformer 输入     | 注意力点积                     |
| 长度发生变化     | 对二维位置网格插值   | 计算新角度，但仍有长度外推问题 |

这张表描述的是原始 ViT 和一类常见 LLM，并不表示“视觉模型只能使用绝对位置，语言模型只能使用 RoPE”。现代视觉 Transformer 也会使用二维相对位置、窗口偏置或二维 RoPE；语言模型同样存在可学习绝对位置、相对位置偏置和 ALiBi 等方案。

选择不同，主要来自任务的归纳偏置：

- 图像 patch 天然构成规则二维网格，分类时输入分辨率通常相对稳定；
- 语言中的依赖经常跨位置复用，而且推理时上下文长度可能不断变化；
- 因此，显式进入注意力关系、计算又高效的 RoPE 很适合自回归语言模型。

## 8. RoPE 能否自动解决无限长上下文

不能。RoPE 可以为任意位置计算旋转角，不需要固定长度的位置表，但“公式能计算”不等于“模型见过并理解这种位置分布”。超过训练长度后，模型会遇到未见过的相位组合，注意力模式也可能发生偏移。

此外，旋转本身具有周期性。不同频率共同作用会延缓整体重复，却不会让长度外推问题凭空消失。因此，长上下文模型还会使用位置插值、频率缩放、NTK-aware scaling、YaRN 等方法，并配合长序列继续训练或微调。

这与 ViT 中的位置插值有一个有趣的共同点：当输入尺度改变时，模型需要把训练阶段学到的“坐标系”平滑地迁移到新的尺度，而不是简单假设原坐标系可以无限延长。

## 9. 我的最终理解

现在再看各种位置编码，我会把它们理解成三个层次：

1. **没有位置编码时，注意力主要知道“内容是否匹配”。**
2. **绝对位置给内容附上坐标，相对位置直接描述两个坐标的差。**
3. **RoPE 用绝对位置旋转 Q、K，却在点积中自然得到相对位移。**

ViT 的 patch 位置编码和 LLM 的 RoPE 并不是两个孤立知识点。前者让我看到位置编码如何把一组图像小块重新组织成二维空间，后者则展示了如何把相对距离嵌入注意力的几何结构。

如果只记一个比喻：绝对位置像在座位上贴编号，相对位置像直接量两张座位之间的距离，而 RoPE 像让每个位置拥有一根按不同速度旋转的指针；比较两根指针时，角度差自然就暴露了它们相隔多远。

## 参考资料

- [位置编码有什么用？简单讲解位置编码原理 + 源码解读（绝对 / 相对 / RoPE）](https://www.bilibili.com/video/BV1xR1RY9ECm/)
- [让研究人员绞尽脑汁的 Transformer 位置编码](https://www.kexue.fm/archives/8130)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
