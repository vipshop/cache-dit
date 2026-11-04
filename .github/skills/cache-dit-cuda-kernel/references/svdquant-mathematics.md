# 从 SmoothQuant 到 SVDQuant：数学原理、实现对应与相关方法借鉴

> 状态：Reference ｜ 日期：2026-11-04
> 关联：`svdq-nvfp4-cute.md` 记录当前 CuTe 实现与性能；`rfc-cuda-optimizations.md` 记录优化项。本文解释数学与量化语义，不替代数值契约和实测验收。

## 0. 阅读路线与核心结论

本文按以下因果链展开：

1. **SmoothQuant**：用逐输入通道的对角重参数化，把激活的量化难度迁移到权重。
2. **W4A4 的新矛盾**：平滑只搬运异常值，不消除异常值；当权重也只有 4 bit 时，权重侧误差会显著恶化。这是论文给出的机制解释和实验结论，不是一个无条件数学定理。
3. **SVDQuant**：把平滑后权重拆成 16-bit 低秩主成分与 4-bit 残差，让高精度分支吸收大幅值方向，低比特分支只量化较小残差。
4. **TurboQuant**：研究随机旋转向量量化、MSE 与内积失真的区别，以及用 1-bit QJL 残差获得无偏内积估计。其定理依赖特定随机量化器，不能直接套到 SVDQ。
5. **DVD-Quant**：提供权重网格精修、在线旋转缩放和时间步位宽切换三条思路；可借鉴的是具体部件，不是把 Video DiT 的论文收益直接当成 SVDQ 收益。

一句话概括 SVDQuant：

$$
\boxed{\text{先平滑，再把“难量化的主成分”交给高精度低秩分支，最后只量化残差。}}
$$

## 1. 符号、矩阵方向与量化器

### 1.1 全文统一采用 PyTorch 权重方向

设一个 Linear 层有 $M$ 个 token、 $K$ 个输入通道、 $N$ 个输出通道：

$$
X\in\mathbb{R}^{M\times K},\qquad
W\in\mathbb{R}^{N\times K},\qquad
Y=XW^\top\in\mathbb{R}^{M\times N}.
$$

SVDQuant 原文使用 $W\in\mathbb{R}^{m\times n}$ 和 $Y=XW$。本文统一写成其转置等价形式，便于对照 `nn.Linear.weight[N,K]`。原文的 $L_1$ 是 down 因子、 $L_2$ 是 up 因子；转置后本文的 $L_1$ 是 up 因子、 $L_2$ 是 down 因子，二者角色相反，交叉阅读时不能只换矩阵形状。原文平滑式写成 $W\operatorname{diag}(\lambda)$，与其 $W\in\mathbb{R}^{m\times n}$、 $\lambda\in\mathbb{R}^m$ 在维度上不一致；按逐输入通道缩放定义应为 $\operatorname{diag}(\lambda)W$，转置后即本文的 $WS$。

令 $s\in\mathbb{R}_{>0}^{K}$， $S=\operatorname{diag}(s)$。SmoothQuant 和 SVDQuant 的平滑变换在本文记号下始终是

$$
\widehat X=XS^{-1},\qquad \widehat W=WS,
$$

因为

$$
\widehat X\widehat W^\top
=XS^{-1}(WS)^\top
=XS^{-1}SW^\top
=XW^\top.
$$

逐元素看，第 $j$ 个输入通道满足

$$
\widehat X_{ij}=\frac{X_{ij}}{s_j},\qquad
\widehat W_{nj}=s_jW_{nj}.
$$

这组展开式可用于检查所有转置和缩放方向。

### 1.2 区分量化码、scale 与反量化张量

对对称均匀量化，记：

- $\bar q$：存储的低比特码；
- $\Delta$：量化步长；
- $\mathcal Q(X)=\Delta\bar q$：反量化后的张量，仍与 $X$ 同形。

典型的 signed $b$-bit absmax 量化为

$$
q_{\max}=2^{b-1}-1,\qquad
\Delta_X=\frac{\max|X|}{q_{\max}},\qquad
\bar q_X=\operatorname{round}\left(\frac{X}{\Delta_X}\right),\qquad
\mathcal Q(X)=\Delta_X\bar q_X.
$$

实际 kernel 还可能有 clamp、非均匀 FP4 码本、group scale 和不同的舍入规则。不要把离线平滑因子 $s_j$ 与运行时量化步长 $\Delta$ 混为一谈：前者是逐输入通道的离线重参数化，后者是量化器的动态或静态参数。

### 1.3 量化粒度为什么也是数学问题

在传统整数 GEMM 中，若激活按 token、权重按输出通道设置 scale，则可以把 scale 提到矩阵乘外：

$$
Y\approx
\operatorname{diag}(\Delta_X)
(\bar q_X\bar q_W^\top)
\operatorname{diag}(\Delta_W).
$$

但激活的逐输入通道 scale 位于归约轴 $K$：

$$
Y_{mn}=\sum_k X_{mk}W_{nk},
$$

不能在标准 GEMM 结束后统一提出。SmoothQuant 的时代背景是传统 INT8 kernel 不支持这种内部 scale；现代 group-wise W4A4 kernel 能在归约过程中按组乘 scale，但组大小、布局和指令仍受硬件约束。因此“数学上可分解”不等于“现有 kernel 可直接执行”。

## 2. SmoothQuant：逐通道等价迁移

> 出处：SmoothQuant §3–4，PDF pp. 3–4；α 消融见 §5.2 与 Figure 10。

### 2.1 异常值如何耗尽有效量化级数

SmoothQuant 观察到：激活异常值往往持续出现在固定输入通道，而不是随机出现在不同 token。对 per-tensor INT8 量化，设第 $i$ 个通道最大幅值为 $m_i$，整张量最大幅值为 $m$，原文用下式描述该通道的有效量化级数：

$$
\text{effective levels}_i\approx 2^8\frac{m_i}{m}.
$$

异常通道抬高 $m$ 后，普通通道可能只剩很少有效级数。per-token scale 不能消除同一 token 内部的通道差异；逐输入通道 scale 虽能解决问题，却又落在传统 GEMM 的归约轴上。

该式是 SmoothQuant 对 INT8 的直观估算，不应不加说明地当成任意位宽、任意非均匀量化器的严格定理。

### 2.2 等价重参数化

SmoothQuant 选择逐输入通道的 $S$，把计算改写为

$$
Y=XW^\top
=(XS^{-1})(WS)^\top
=\widehat X\widehat W^\top.
$$

变换本身在全精度代数中严格等价；误差只来自随后分别量化 $\widehat X$ 和 $\widehat W$。工程上， $S^{-1}$ 通常可折入前一层，残差分支则必须同步处理，否则等价性会被破坏。

### 2.3 平滑因子与迁移强度 $\alpha$

定义校准数据上的逐输入通道幅值：

$$
m_j^X=\max_{\text{calib},i}|X_{ij}|,\qquad
m_j^W=\max_n|W_{nj}|.
$$

SmoothQuant 取

$$
\boxed{
s_j=\frac{(m_j^X)^\alpha}{(m_j^W)^{1-\alpha}}
},\qquad 0\leq\alpha\leq1.
$$

代入 $\widehat X=XS^{-1}$ 和 $\widehat W=WS$，得到

$$
\max|\widehat X_j|
=\frac{m_j^X}{s_j}
=(m_j^Xm_j^W)^{1-\alpha},
$$

以及

$$
\max|\widehat W_j|
=s_jm_j^W
=(m_j^Xm_j^W)^\alpha.
$$

因此

$$
\max|\widehat X_j|\cdot\max|\widehat W_j|=m_j^Xm_j^W,
$$

与 $\alpha$ 无关； $\alpha$ 只是在对数域重新分配两侧幅值。特别地，

$$
\alpha=\frac12
\quad\Longrightarrow\quad
\max|\widehat X_j|
=\max|\widehat W_j|
=\sqrt{m_j^Xm_j^W}.
$$

所以 $\alpha=0.5$ 的精确含义是**两侧逐通道最大幅值相等**。它不是“量化误差各减少一半”，也不是理论最优点。两个端点可用于检查公式方向：

- $\alpha=1$： $s_j=m_j^X$，激活通道被归一化，更多难度迁到权重；
- $\alpha=0$： $s_j=1/m_j^W$，更多难度留在激活。

SmoothQuant 原文报告多数模型在约 0.4–0.6 区间表现较稳，但不同模型可使用更大值，例如 GLM-130B 的 0.75。正确做法是按模型和层校准，而不是把 0.5 当成普适常数。

## 3. 从 SmoothQuant 到 SVDQuant 的误差视角

> 出处：SVDQuant §4.1–4.2，PDF p. 5；Proposition 4.1 证明见 Appendix A.1，PDF p. 17。

### 3.1 线性层量化误差分解

定义

$$
E(X,W)=
\left\|XW^\top-\mathcal Q(X)\mathcal Q(W)^\top\right\|_F.
$$

SVDQuant 的 Proposition 4.1 给出

$$
\boxed{
E(X,W)\leq
\|X\|_F\|W-\mathcal Q(W)\|_F
+\|X-\mathcal Q(X)\|_F
\left(\|W\|_F+\|W-\mathcal Q(W)\|_F\right).
}
$$

证明只需插入中间项并使用三角不等式与 Frobenius 范数次乘性：

$$
E(X,W)=\left\|X(W-\mathcal Q(W))^\top+(X-\mathcal Q(X))\mathcal Q(W)^\top\right\|_F.
$$

$$
E(X,W)\leq\|X\|_F\|W-\mathcal Q(W)\|_F+\|X-\mathcal Q(X)\|_F\|\mathcal Q(W)\|_F.
$$

$$
E(X,W)\leq\|X\|_F\|W-\mathcal Q(W)\|_F+\|X-\mathcal Q(X)\|_F\left(\|W\|_F+\|W-\mathcal Q(W)\|_F\right).
$$

最后一步使用

$$
\|\mathcal Q(W)\|_F
=\|W- (W-\mathcal Q(W))\|_F
\leq\|W\|_F+\|W-\mathcal Q(W)\|_F.
$$

**成立条件**：该确定性上界对任意 $X,W$ 和任意反量化映射 $\mathcal Q$ 成立，不需要高斯或异常值假设。

**解释边界**：它是弱上界，不是误差等式，也不能保证按上界排序就一定按真实端到端误差排序。SmoothQuant 原文的出发点是有效量化级数均衡；用这个 Frobenius 上界统一解释 SmoothQuant，是 SVDQuant 及二级解读材料提供的后验视角，不应倒写成 SmoothQuant 原始推导。

### 3.2 为什么平滑到 4 bit 后仍不够

平滑后，激活侧 $\widehat X$ 的异常值减少，但权重侧 $\widehat W=WS$ 继承了更大的通道幅值。上界中，激活量化项可能下降，权重项

$$
\|\widehat X\|_F\|\widehat W-\mathcal Q(\widehat W)\|_F
$$

却可能上升。W8A8 中权重通常还有足够表示能力吸收这种迁移；W4A4 中，权重也只有很少码点，权重异常值同样会拉大量化步长。

这解释了 SVDQuant 的下一步：不是继续无限增大 $\alpha$，而是把平滑后权重中难以由 4 bit 表达的部分转移到高精度分支。

## 4. SVDQuant：高精度低秩分支加低比特残差

### 4.1 精确分解与唯一的量化近似

对平滑后权重做秩 $r$ 分解：

$$
\widehat W=L_1L_2+R,
$$

其中

$$
L_1\in\mathbb{R}^{N\times r},\qquad
L_2\in\mathbb{R}^{r\times K},\qquad
R\in\mathbb{R}^{N\times K},\qquad
r\ll\min(N,K).
$$

于是，在全精度低秩分支下，

$$
XW^\top=\widehat X\widehat W^\top.
$$

$$
XW^\top=\widehat X(L_1L_2+R)^\top.
$$

$$
XW^\top=\underbrace{\widehat XL_2^\top L_1^\top}_{\text{16-bit 低秩分支}}+\underbrace{\widehat XR^\top}_{\text{残差分支}}.
$$

SVDQuant 只把残差分支替换为 W4A4：

$$
\boxed{
Y\approx
\widehat XL_2^\top L_1^\top
+\mathcal Q(\widehat X)\mathcal Q(R)^\top.
}
$$

前两行是代数恒等式，只有最后的 $\mathcal Q$ 是近似。论文模型忽略 16-bit 低秩分支自身的存储和累加舍入；实现验收不能据此把它当成数学无损。

量化输出误差变为

$$
\left\|\widehat X\widehat W^\top-\widehat XL_2^\top L_1^\top-\mathcal Q(\widehat X)\mathcal Q(R)^\top\right\|_F=\left\|\widehat XR^\top-\mathcal Q(\widehat X)\mathcal Q(R)^\top\right\|_F.
$$

$$
\left\|\widehat XR^\top-\mathcal Q(\widehat X)\mathcal Q(R)^\top\right\|_F=E(\widehat X,R).
$$

由于 $\widehat X$ 已平滑，SVDQuant 进一步集中降低 $\|R\|_F$ 和 $\|R-\mathcal Q(R)\|_F$。

### 4.2 低秩分支的算力与参数比例

两次低秩 GEMM 的乘加量相对原完整 GEMM 的比例为

$$
\frac{MKr+MrN}{MKN}
=\frac{r(K+N)}{KN}.
$$

低秩参数相对原权重参数数量的比例同样是

$$
\frac{r(K+N)}{KN}.
$$

例如 $K=N=3072,r=32$ 时约为 2.08%。这只是乘加量或未计位宽的参数比例，不是打包后的字节比例，更不是延迟比例。小秩 GEMM 的额外输入读取、输出写入和 kernel launch 可能主导实际开销。

### 4.3 Proposition 4.2：残差范数如何进入量化误差上界

> 出处：SVDQuant §4.2，PDF p. 6；完整证明见 Appendix A.2，PDF pp. 17–18。

先固定定理的真实适用场景：

- 对含 $\mathcal N$ 个元素的单张量 $R$ 使用**单一 absmax scale**；
- $\Delta_R=\max|R|/q_{\max}$；
- 使用 round-to-nearest，论文采用每元素误差不超过一个 LSB 的宽松估计；
- 存在常数 $c$，使分布满足

$$
\mathbb E[\max|R|]
\leq c\,\mathbb E[\|R\|_F].
$$

则

$$
\boxed{
\mathbb E\|R-\mathcal Q(R)\|_F
\leq
\frac{c\sqrt{\mathcal N}}{q_{\max}}
\mathbb E\|R\|_F.
}
$$

推导如下。由 $\mathcal Q(R)=\Delta_R\operatorname{round}(R/\Delta_R)$，

$$
\|R-\mathcal Q(R)\|_F
\leq \Delta_R\sqrt{\mathcal N}
=\frac{\sqrt{\mathcal N}}{q_{\max}}\max|R|.
$$

两边取期望，再代入 regularity condition：

$$
\mathbb E\|R-\mathcal Q(R)\|_F\leq\frac{\sqrt{\mathcal N}}{q_{\max}}\mathbb E[\max|R|].
$$

$$
\mathbb E\|R-\mathcal Q(R)\|_F\leq\frac{c\sqrt{\mathcal N}}{q_{\max}}\mathbb E\|R\|_F.
$$

若元素为独立同分布高斯，论文用高斯最大值不等式，并配合 Cauchy–Schwarz 与半正态一阶矩得到

$$
c=\sqrt{\frac{\pi\log\mathcal N}{\mathcal N}},
$$

从而

$$
\mathbb E\|R-\mathcal Q(R)\|_F
\leq
\frac{\sqrt{\pi\log\mathcal N}}{q_{\max}}
\mathbb E\|R\|_F.
$$

这个系数并不紧。例如 $\mathcal N=4096$ 且 INT4 的 $q_{\max}=7$ 时约为 0.73。它提供的是“残差范数变小有助于压低误差上界”的动机，不是精确误差预测器。

**适用边界必须保留：**

1. per-group scale 需要按组重新分析，不能把单张量证明原样当成 group-wise kernel 的严格定理。
2. 命题形式上针对 Eq. (1) 的量化器，但证明第一步依赖逐元素舍入和每元素不超过一个 LSB 的误差界。GPTQ 的误差反馈使 $\mathcal Q(R)$ 不再是对原 $R$ 的逐元素舍入结果，因此不能直接套用该证明。
3. $c$ 依赖分布；改变 rank 会同时改变 $\|R\|_F$ 和残差分布，不保证整个上界严格单调下降。
4. 论文使用一个 LSB 的宽松 rounding 界；普通最近舍入可得到更紧的半个 LSB，但这不改变上述适用条件。

### 4.4 Eckart–Young：最优的是低秩近似，不是端到端量化误差

> 出处：SVDQuant §4.2，PDF p. 6；最优性边界为本文基于目标函数的分析。

对精确 SVD

$$
\widehat W=U\Sigma V^\top,
$$

取前 $r$ 个奇异三元组：

$$
L_1=U_r\Sigma_r,\qquad L_2=V_r^\top.
$$

Eckart–Young–Mirsky 定理给出

$$
L_1L_2
=\arg\min_{\operatorname{rank}(L)\leq r}
\|\widehat W-L\|_F,
$$

并且

$$
\boxed{
\|R\|_F
=\|\widehat W-L_1L_2\|_F
=\sqrt{\sum_{i=r+1}^{\min(N,K)}\sigma_i^2}.
}
$$

这里是**尾奇异值平方和开根**，不是“奇异值的几何平均”。

不能进一步声称“SVD 使量化误差最小”，原因有三层：

1. Eckart–Young 优化无权重的 $\|\widehat W-L\|_F$，而真实层输出误差含激活加权，直接目标更接近 $\min_{\operatorname{rank}(L)\leq r}\|\widehat X(\widehat W-L)^\top\|_F$。

2. Proposition 4.2 只是上界，且常数依赖残差分布。
3. 实际还有 group scale、GPTQ/RTN、低精度累加和后续非线性，均不在 Eckart–Young 目标内。

此外，尾谱恒等式只对**精确截断 SVD 的初始分解**成立。随机化近似 SVD 或后续 refine 后，不能继续把当前 $\|R\|_F$ 写成原矩阵的尾奇异值和。

### 4.5 迭代 refine：把部分量化误差重新吸收到低秩分支

> 出处：SVDQuant §4.2，PDF p. 6；迭代式为本文按原文与当前实现语义展开的等价写法。

论文在初始 SVD 后继续迭代。记 $\mathcal P_r(Z)$ 为对 $Z$ 做秩 $r$ 截断分解，一种等价表述是

$$
R^{(t)}=\widehat W-L^{(t)},
$$

$$
Z^{(t)}=\widehat W-\mathcal Q(R^{(t)}),
$$

$$
L^{(t+1)}=\mathcal P_r(Z^{(t)}),
\qquad
R^{(t+1)}=\widehat W-L^{(t+1)}.
$$

关键是最后的 residual 始终相对原始 $\widehat W$ 重新锚定，而不是相对 $Z^{(t)}$。运行时计算的是

$$
L^{(t+1)}+\mathcal Q(R^{(t+1)}),
$$

重新锚定可避免重复扣除上一次校正。论文称迭代后选择误差最小的结果；具体实现是否保存最优轮次要以其代码为准。

这也说明“闭式 SVD 就是完整 SVDQuant”并不准确：初始 SVD 提供主成分，refine 则让低秩分支进一步吸收残差量化留下的误差结构。

### 4.6 为什么顺序是先平滑、再 SVD、最后量化

- **先平滑**：把激活中的持续通道异常值迁入权重，使高精度低秩分支有机会同时承接两侧量化难度。
- **再 SVD**：提取平滑后权重中占主导的大奇异方向，得到范围更小的残差。
- **最后量化**：只对 $R$ 和 $\widehat X$ 做低比特量化，可再叠加 GPTQ 或 refine。

这与“先量化，再用低秩分支补偿量化误差”的 LoRC 路线不同。SVDQuant 论文指出，量化误差的奇异值谱较平，低秩近似难以抓住；平滑后权重的主奇异方向则更集中。低秩分支也不等于 LoRA：前者服务于量化重构，后者服务于任务适配，虽然二者可共享融合 kernel 或合并存储。

## 5. 论文配置、性能含义与当前实现对应

### 5.1 论文的离线与在线流程

> 出处：SVDQuant §4.2–4.3，PDF pp. 5–7；具体配置见 Appendix D，PDF p. 19。

**离线：**

1. 用校准数据统计激活通道幅值，逐层搜索 $\alpha$，目标为 SVD 后的层输出 MSE。
2. 计算 $\widehat W=WS$，做截断 SVD 和可选迭代 refine。
3. 量化残差权重。论文 4-bit 主结果使用 GPTQ，而不是纯 RTN。
4. 保存 4-bit 权重码、scale、平滑因子和 16-bit 低秩因子。

**在线：**

1. producer 同时完成激活平滑、动态量化和低秩 down projection。
2. GEMM 同时完成低比特残差积、低秩 up projection 和 bias 累加。

论文 4-bit 配置为 rank 32；INT4 使用 group size 64 和 16-bit scale，NVFP4 使用 group size 16 和 FP8 scale。FLUX 的 adaptive normalization 相关输入保留 16 bit，其他部分模型的 cross-attention K/V 投影也保留高精度。讨论“全模型 W4A4”时必须带上这些例外。

### 5.2 为什么低秩分支必须融合

> 出处：SVDQuant §4.3 与 Figure 6，PDF pp. 6–7。

论文 Fig. 6 中，朴素 rank-32 低秩分支在 QKV projection 上带来 57% 延迟开销；正文概括为约 50%。融合后论文报告典型净开销降至 5–10%。两组数字的层次和口径不同，不能混写。

融合原则是：

- down projection 与 activation quantize 共享同一份输入，避免重读；
- up projection 与 4-bit GEMM 共享同一份输出，避免重写。

因此“低秩 FLOPs 很少”只说明算力占比小，不能推出“低秩免费”。当前 CuTe/native 的具体性能仍以 `svdq-nvfp4-cute.md` 的实测为准。

### 5.3 数学量到 cache-dit 代码的映射

以下路径均相对 cache-dit 仓库根目录：

| 数学量或步骤 | 当前代码对应 | 需要注意的语义 |
|---|---|---|
| $s$ | `compute_smooth_scale`、`smooth_factor` | 默认 $\alpha=0.5$；实际存储还经过 kernel 所需的打包排列 |
| $\widehat W=WS$ | `quantization/svdquant/quantizer.py` 中 `smoothed_weight` | 沿输入通道列乘 $s$ |
| $L_1,L_2,R$ | `lowrank.py::decompose_lowrank_residual` | `proj_up` 对应 $L_1$，原始 `lowrank_down` 对应 $L_2$ |
| 初始 SVD | `calibrate_precision` 路由 | 默认 `low` 使用随机化 `torch.svd_lowrank`，不保证精确 Eckart–Young 最优 |
| refine | `_refine_lowrank_split` | 当前默认 `svd_refine_iters=0`；开启后残差相对 $\widehat W$ 重锚定 |
| 残差量化 | `_fake_quantize_dequantize_residual` 与 packing | 当前实现是 RTN 模拟加可选 refine；论文主结果的 GPTQ 不能归到当前实现名下 |
| INT4 权重 scale | `_compute_group_scales` | 每输出通道、每 64 输入通道一组， $\Delta=\max|R|/7$，码 clamp 到 $[-8,7]$ |
| NVFP4 权重 scale | `_compute_nvfp4_channel_and_group_scales` | E2M1 码本，per-output-channel scale 乘 per-16-group FP8 E4M3 scale |
| 激活 scale | native/CuTe producer | 每次 forward 按 token 与 64/16 通道组动态产生 |
| 低秩运行时输入 | `packing.py` 中 `lora_down /= smooth` 与 producer 的 raw input | 存储的是 $(L_2S^{-1})^\top$，所以低秩分支直接消费原始 $X$ |

最后一项可写成

$$
\widetilde L_2=L_2S^{-1},
\qquad
D=X\widetilde L_2^\top
=X(L_2S^{-1})^\top
=XS^{-1}L_2^\top
=\widehat XL_2^\top.
$$

随后计算 $DL_1^\top$，与理论式完全一致。这是实现中最容易把 $S$ 重复乘除的一处。

### 5.4 论文结果的正确读法

> 出处：SVDQuant §5.2 与 Figure 8，PDF pp. 8–9；Table 4 见 Appendix E.4，PDF p. 27。

SVDQuant 论文 Table 4 的 FLUX 单步桌面 RTX 4090 结果为：BF16 657 ms、naive INT4 212 ms、未融合 SVDQuant 250 ms、SVDQuant+Nunchaku 218 ms。它们支持“融合使低秩分支接近 naive INT4”的结论，不支持脱离硬件和模型推广固定加速比。

论文报告的 FLUX 模型大小约从 22.2 GiB 降至 6.1 GiB，低秩分支额外约 0.3 GiB；另一组 DiT 推理显存口径为 24.4 GiB 降至 6.9 GiB。摘要和图中的 3.5×/3.6×、相对 BF16/W4A16 的 8.7×/3.0×属于不同基准，引用时必须带基线。

主表 4-bit 精度结果包含 GPTQ。把当前 RTN、不同 rank 或关闭 refine 的结果与论文主表直接比较，属于配置不一致。

## 6. TurboQuant：随机旋转、Lloyd–Max 与 QJL 残差

### 6.1 问题设定与 SVDQ 不同

> 出处：TurboQuant §1–2，PDF pp. 1–8。TurboQuant 对旋转后的笛卡尔坐标做 Lloyd–Max；PolarQuant 是另一项极坐标变换工作，在本文中仅作为 KV cache 基线，二者不能混称。

TurboQuant 处理单位向量 $x\in\mathbb S^{d-1}$ 的存储与内积估计，主要面向 KV cache 和最近邻检索。它区分

$$
D_{mse}=\mathbb E\|x-\widetilde x\|_2^2
$$

和

$$
D_{prod}=\mathbb E\left(\langle y,x\rangle-
\langle y,\widetilde x\rangle\right)^2.
$$

其内积任务中，查询 $y$ 可以保持全精度；SVDQ W4A4 则要求激活和权重都变成 tensor core 可消费的低比特码。两者的最优目标和执行约束不同。

### 6.2 随机旋转使坐标分布可预先求解

> 出处：TurboQuant §3.1 与 Theorem 1，PDF pp. 9–10；信息论下界见 §3.3。

取 Haar 随机正交矩阵 $\Pi$，令

$$
z=\Pi x.
$$

由于 $x$ 在单位球上， $z$ 在单位球上均匀分布。单坐标密度为

$$
f_Z(z)=
\frac{\Gamma(d/2)}
{\sqrt{\pi}\Gamma((d-1)/2)}
(1-z^2)^{(d-3)/2},
\qquad -1\leq z\leq1.
$$

高维时

$$
Z_j\approx\mathcal N(0,1/d),
$$

不同坐标渐近近独立。于是可对每个坐标使用相同的一维 Lloyd–Max 码本，而不需要校准具体数据集。这里的“近独立”是高维近似，不是有限维严格独立。

对码点 $c_1<\cdots<c_{2^b}$，最近邻边界为

$$
\tau_i=\frac{c_i+c_{i+1}}2,
$$

MSE 最优码点满足条件均值方程

$$
c_i=\mathbb E[Z\mid Z\in(\tau_{i-1},\tau_i)].
$$

量化与反量化为

$$
idx_j=\arg\min_k|z_j-c_k|,
\qquad
\widetilde x=\Pi^\top\widetilde z,
\qquad
\widetilde z_j=c_{idx_j}.
$$

正交变换保持范数，因此旋转域的坐标 MSE 就是原空间的向量 MSE。论文 Theorem 1 给出 $D_{mse}\leq(\sqrt{3\pi}/2)\cdot4^{-b}$，下界为 $D_{mse}\geq4^{-b}$；内积失真的对应下界为 $D_{prod}\geq(\|y\|_2^2/d)\cdot4^{-b}$。这里的“近优”指只相差常数因子，不是说任意任务都达到相同端到端质量。

### 6.3 MSE 最优不等于内积无偏

在 1 bit、高维极限下，码点约为

$$
c_\pm=\pm\sqrt{\frac{2}{\pi d}}.
$$

对任意固定 $y$，该重建给出

$$
\mathbb E\langle y,\widetilde x\rangle
=\frac{2}{\pi}\langle y,x\rangle,
$$

存在 $2/\pi$ 的乘性偏置。这是 TurboQuant 引入第二阶段残差修正的原因。

### 6.4 QJL 的无偏性与方差

> 出处：TurboQuant §2.2、Lemma 4，PDF p. 8；该引理转述 QJL 的既有结果，并非 TurboQuant 首创。

令 $S\in\mathbb R^{d\times d}$，元素独立服从 $\mathcal N(0,1)$。对单位向量 $u$，定义

$$
Q_{qjl}(u)=\operatorname{sign}(Su),
$$

$$
Q_{qjl}^{-1}(v)=
\frac{\sqrt{\pi/2}}{d}S^\top v.
$$

对任意一行 $s_i\sim\mathcal N(0,I)$，有

$$
\mathbb E\left[s_i\operatorname{sign}(s_i^\top u)\right]
=\sqrt{\frac{2}{\pi}}u.
$$

因此

$$
\mathbb E[Q_{qjl}^{-1}(Q_{qjl}(u))]
=\frac{\sqrt{\pi/2}}{d}
\cdot d\sqrt{\frac{2}{\pi}}u
=u.
$$

于是对任意 $y$，内积估计无偏；论文同时给出

$$
\operatorname{Var}
\left(\langle y,Q_{qjl}^{-1}Q_{qjl}(u)\rangle\right)
\leq\frac{\pi}{2d}\|y\|_2^2.
$$

无偏性是对随机矩阵取期望，不是每次实际查询都没有误差。

### 6.5 两阶段 TurboQuant 与内积误差界

> 出处：TurboQuant §3.2、Algorithm 2 与 Theorem 2，PDF pp. 12–13。

总预算为每坐标 $b$ bit 时，先用 $b-1$ bit 做 MSE 量化：

$$
\widetilde x_{mse}=\mathcal Q_{mse}^{(b-1)}(x),
\qquad
r=x-\widetilde x_{mse},
\qquad
\gamma=\|r\|_2.
$$

再对残差方向做 1-bit QJL：

$$
\widetilde x
=\widetilde x_{mse}
+\gamma\frac{\sqrt{\pi/2}}{d}
S^\top\operatorname{sign}(Sr).
$$

由于 $\operatorname{sign}(Sr)=\operatorname{sign}(S(r/\gamma))$，以 $\widetilde x_{mse}$ 为条件，

$$
\mathbb E_S[\widetilde x\mid\widetilde x_{mse}]
=\widetilde x_{mse}+r=x.
$$

并且

$$
D_{prod}=\mathbb E_{\widetilde x_{mse}}\operatorname{Var}_S\left(\langle y,\widetilde x\rangle\mid\widetilde x_{mse}\right).
$$

$$
D_{prod}\leq\frac{\pi}{2d}\|y\|_2^2\mathbb E\|r\|_2^2.
$$

$$
D_{prod}\leq\frac{\pi}{2d}\|y\|_2^2D_{mse}^{(b-1)}.
$$

**这个界不能套到普通 SVDQ。** 它依赖独立的 QJL 随机投影、条件无偏性和浮点残差范数 $\gamma$；普通 RTN、INT4 或 NVFP4 不满足这些条件。实际存储还要计入 $\gamma$，非单位向量还需保存原始范数；稠密 $\Pi,S$ 还涉及 $O(d^2)$ 变换成本。

### 6.6 对 SVDQ 有什么可借鉴

- **可借鉴思想**：区分重构 MSE 与下游内积/层输出误差；评价量化器时不能只看权重 rel-L2。
- **可借鉴异常值处理视角**：正交旋转重新分配坐标能量，对角平滑只改变各通道尺度，两者不是同一种操作。
- **不宜直接搬算子**：Lloyd–Max 非均匀码本不能直接替换 INT4/E2M1 tensor-core 语义；QJL 还会增加稠密随机投影、符号位和范数侧信息。
- **不能搬精度结论**：TurboQuant 的 3.5-bit KV cache“质量中性”不是 W4A4 Linear 的精度保证。
- **不能搬性能结论**：论文的索引或 KV 结果不包含 SVDQ producer、低秩分支和 packed GEMM 的成本模型。

## 7. DVD-Quant：BGR、ARQ 与时间步位宽切换

### 7.1 “Data-free”的准确含义

DVD-Quant 的 data-free 主要指**不依赖离线校准数据集**，不是推理时不需要数据统计：

- BGR 只使用权重，离线精修量化网格；
- ARQ 在 forward 中计算旋转后激活的通道最大值；
- $\delta$-GBS 在生成过程中读取相邻时间步特征并累计变化量。

它仍有离线搜索成本、在线统计成本以及需要调节的超参数。

### 7.2 BGR（Bounded-init Grid Refinement）：裁剪初始化与网格精修

> 出处：DVD-Quant §3.1 与 Algorithm 1，PDF pp. 4–5。

对每输出通道的非对称均匀量化，MinMax 使用

$$
\Delta=\frac{\max W-\min W}{2^b-1},
\qquad
z=-\operatorname{round}\left(\frac{\min W}{\Delta}\right),
$$

$$
q=\operatorname{clamp}
\left(\operatorname{round}(W/\Delta)+z,0,2^b-1\right),
\qquad
\widetilde W=\Delta(q-z).
$$

BGR 的目标是

$$
\min_{\Delta,z}\|W-\Delta(q-z)\|_F.
$$

固定 $q,z$ 时，对某一行， $\Delta$ 的最小二乘闭式解为

$$
\boxed{
\Delta'=
\frac{\langle q-z,W\rangle}
{\langle q-z,q-z\rangle}.
}
$$

因为

$$
\frac{\partial}{\partial\Delta}
\|W-\Delta(q-z)\|_2^2=0
$$

恰好得到该式。随后更新 $z$，重新计算 $q$，可迭代多轮。完整 BGR 不只是一次交替最小二乘：外层还通过逐步 clamp 权重边界进行 bounded search，为内层精修提供更好的初始网格。

论文报告权重重构误差显著下降，但这不等于端到端生成质量同比例提升；其消融也表明 BGR 不能单独解决 W4A4 激活问题。

### 7.3 对称、无 zero-point 时如何借到 SVDQ

对 SVDQ 的 INT4 残差，令 $z=0$，固定当前码 $q$，则每组 scale 的 LS 更新为

$$
\Delta'
=\frac{\sum_{i\in g}q_iR_i}{\sum_{i\in g}q_i^2}.
$$

随后按当前量化规则重新取整、clamp，并只在实际误差下降时接受。这是对现有 absmax 初始化最直接的数学补充，但必须满足：

1. 在 SVDQ 的 group 粒度上做，而不是照搬 DVD-Quant 的 per-row 粒度。
2. 残差是低秩分解后的 $R$，不是原始 $W$；仍需评估层输出和最终生成质量。
3. NVFP4 的 $q_i$ 是 E2M1 码值，且存在 per-channel 与 per-group 两级 scale。
4. NVFP4 候选 group scale 必须先模拟 FP8 E4M3 舍入再计算接受准则，否则连续 LS 的收益可能在存储 scale 时消失。
5. 推理期码和 scale 格式不变时才可能保持零运行时开销；离线精修本身仍有成本。

因此它是**高优先级原型验证项**，不是已经证明有效的精度收益。

### 7.4 ARQ（Auto-scaling Rotated Quantization）：Hadamard 旋转加在线通道缩放

> 出处：DVD-Quant §3.2，PDF pp. 5–6。

在本文矩阵方向下，取归一化 Hadamard 矩阵 $H$，使 $HH^\top=I$；若使用未归一化的 $\pm1$ Hadamard 矩阵，应写成 $H/\sqrt K$，否则会多出整体因子 $K$。此时有

$$
XW^\top=(XH)(WH)^\top.
$$

DVD-Quant 令

$$
\widetilde X=XH,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_K),
\qquad
\lambda_j=\max_i|\widetilde X_{ij}|,
$$

再计算

$$
\widehat X=\mathcal Q(\widetilde X\Lambda^{-1}),
\qquad
\widehat W=\operatorname{BGR}(WH),
$$

并将 $\Lambda$ 折回激活反量化侧，即

$$
Y\approx(\widehat X\Lambda)\widehat W^\top.
$$

快速 Hadamard 变换可把单个向量的旋转从稠密 $O(K^2)$ 降为 $O(K\log K)$，但在线逐通道最大值、额外数据移动和 kernel 融合仍有成本。直接 FHT 通常要求维度为 2 的幂，其他维度需要块对角拼接等扩展。

更关键的是， $\Lambda$ 位于归约轴 $K$。不能仅凭代数恒等式就宣称现有 SVDQ GEMM 可直接消费；需要明确它在反量化和 MMA 中的位置。SVDQuant 原文指出，adaptive normalization 的运行时权重会阻止某些离线旋转折叠，而在线同时旋转激活与权重又有显著开销。DVD-Quant 说明在线旋转是可能的工程路线，但并没有消除这些成本。

### 7.5 δ-GBS（δ-Guided Bit Switching）：按特征演化切换激活位宽

> 出处：DVD-Quant §3.3，PDF p. 6；速度与显存结果见 Table 5。

定义相邻时间步的归一化变化量，记为 $\ell_1$ 以区别于低秩因子 $L_1$：

$$
\ell_1(F,t)=
\frac{\|F_t-F_{t-1}\|_1}{\|F_{t-1}\|_1}.
$$

从上次重置点 $t_p$ 累计：

- 累计值小于 $\delta$ 时使用低位宽；
- 达到阈值时切到高位宽并重置计数器。

DVD-Quant 在 W4A4 与 W4A8 路径间切换，平均形成 W4A6，而不是为每个时间步重新生成一套 4-bit 权重。其 50 步 HunyuanVideo、RTX 4090 实验中，论文报告 W4A4 约 2.12× 延迟改善、W4A6 约 1.93×，显存降幅均约 3.68×。这些数字依赖其模型、视频设置和 kernel，不能当成 SVDQ 预期收益。

对 cache-dit 的启示是：累积相对变化量可作为时间步精度调度信号，但需要两套位宽 kernel、重新标定阈值，并与缓存复用策略分别验收。现有缓存统计形式相近，不代表阈值可直接复用，也不代表“特征变化小”同时证明“可跳过计算”和“可降低精度”。

## 8. 面向 SVDQ 的借鉴优先级

| 方向 | 可能收益 | 必须保持或验证的条件 | 失败判据 | 判定 |
|---|---|---|---|---|
| 对称 BGR 式 scale LS 精修 | 离线降低残差权重量化误差，运行时格式不变 | per-group、无 zero-point；重新量化后接受；测层输出与 e2e | FP8 scale 舍入后无改善，或 rel-L2 改善但生成质量不改善 | 优先原型验证 |
| NVFP4 两级 scale 联合精修 | 比单纯 absmax 更合理分配动态范围 | E2M1 码本、E4M3 group scale、per-channel scale 一起模拟 | 连续优化改善无法保留到实际存储格式 | 优先研究，复杂度高于 INT4 |
| 块 Hadamard/其他快速旋转 | 分散持续异常值，降低对静态校准的依赖 | 离线权重变换与在线 producer 同步；低秩支路同步；不破坏 fused MLP | 新增访存或打断融合后 e2e 更慢，或精度无稳定提升 | 需改造验证 |
| MSE 与内积/输出误差分开评估 | 避免只看权重重构误差 | 使用相同输入、相同层和下游指标 | 只报告权重 MSE，无法说明模型收益 | 可直接采纳为评估原则 |
| QJL 无偏残差 | 理论上改善随机内积估计 | 必须真的引入 QJL、随机矩阵和浮点范数 | 需要稠密投影或额外访存，且现有 16-bit 低秩分支已承担残差校正 | 不宜作为当前 SVDQ 主线 |
| Lloyd–Max 替代 INT4/E2M1 | 特定分布下降低标量 MSE | 需要 LUT/自定义 GEMM，不再等价于现有 tensor-core 格式 | 无法进入 MMA 或反量化开销吞掉收益 | 不宜直接替换 |
| $\delta$-GBS 时间步调度 | 在冗余步用低位宽、关键步用高位宽 | 双位宽 kernel、阈值重标定、与缓存分别 A/B | 切换开销或高位宽步占比使 e2e 收益消失 | 中长期验证 |

建议的实验顺序是：先做**离线、无 ABI 变化**的 INT4 scale 精修，再研究 NVFP4 两级 scale；旋转和时间步调度都应在独立 kernel 与完整 e2e 两层验收之后才能形成性能结论。

## 9. 常见误读与二手材料纠错

| 说法 | 正确理解 |
|---|---|
| $\alpha=0.5$ 理论最优 | 只表示在特定公式下两侧通道最大幅值相等；实际需按模型或层校准 |
| SmoothQuant 最小化了 SVDQuant 的 Frobenius 上界 | SmoothQuant 原文主要用有效量化级数解释；统一上界是后验分析框架 |
| SVD 使端到端量化误差最小 | 精确 SVD 只保证无权 Frobenius 意义下的最优 rank- $r$ 近似 |
| $\sqrt{\sum_{i>r}\sigma_i^2}$ 是奇异值几何平均 | 它是尾奇异值平方和开根，即截断残差的 Frobenius 范数 |
| refine 后仍可用原矩阵尾谱表示当前残差 | 尾谱恒等式只属于精确初始截断；refine 或随机化 SVD 后不能直接使用 |
| Proposition 4.2 证明所有 SVDQuant 实现的误差界 | 它针对单 scale absmax RTN 和分布正则条件，不直接覆盖 group-wise 或 GPTQ |
| 低秩分支几乎没有开销 | FLOPs 占比小不等于延迟小；未融合时论文报告显著访存开销 |
| 论文主结果就是纯 RTN | 论文 4-bit 主结果包含 GPTQ；当前 cache-dit 默认路径应单独说明 |
| 当前默认 SVD 一定满足精确 Eckart–Young 最优 | 默认 `calibrate_precision=low` 是随机化近似 SVD |
| TurboQuant 的误差界可直接用于 SVDQ | 该界来自 QJL 条件无偏性，普通 INT4/NVFP4 不满足 |
| DVD-Quant 的 data-free 表示运行时无统计 | ARQ 和 $\delta$-GBS 都使用运行时数据统计 |
| “显存降低 3.6×、加速 3×”可脱离基线引用 | 3.6×对应模型体积，3.5×对应另一组 DiT 推理显存；3.0×常对应 W4A16 基线，8.7×是另一组相对 BF16 的 e2e 口径 |

中文二级解读《From SmoothQuant to SVDQuant》适合建立直观路线，但其中低秩矩阵形状与计算占比存在记号混用，“尾奇异值几何平均”属于术语错误，“8-bit 下对 SmoothQuant 无显著优势”不能由 SVDQuant 正文直接支持。正式引用应以原论文为准。

## 10. 公式与实现验收清单

### 数学自检

- 所有 Linear 公式是否统一使用 $W[N,K]$ 和 $Y=XW^\top$？
- 平滑是否始终为 $X S^{-1}$、 $WS$，而不是把逆乘到错误一侧？
- 是否区分低比特码 $\bar q$ 与反量化张量 $\mathcal Q(X)$？
- 每条定理是否同时写出 scale 粒度、分布、随机性和舍入前提？
- 是否把 Eckart–Young 的目标误写成激活加权输出误差？
- 是否把上界当成等式，或把期望无偏当成单次无误？
- TurboQuant 的 $D_{mse}$ 是否明确为 $b-1$ bit 阶段？

### 实现与实验自检

- 论文的逐层 $\alpha$ 搜索、GPTQ、refine 是否与当前实现配置区分？
- 默认随机化 SVD 是否被误写成精确 SVD？
- `proj_down` 是否已包含 $S^{-1}$，低秩分支是否因此消费原始 $X$？
- INT4/NVFP4 的 group size、scale dtype 和码本是否进入模拟？
- 新 scale 是否按实际存储 dtype 舍入后再计算误差？
- 新旋转是否同时修改主残差分支和低秩分支？
- 性能结论是否区分 producer、独立 GEMM、完整 layer 和 e2e？
- 精度结论是否至少包含层输出误差与同配置生成指标，而不只看权重 MSE？

## 参考文献

1. Guangxuan Xiao et al. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**. ICML 2023；本文核对 arXiv:2211.10438v7。重点：§2–4、Eq. (1)–(4)。
2. Muyang Li et al. **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**. ICLR 2025；本文核对 arXiv:2411.05007v3。重点：§3–4、Appendix A/D/E。
3. Amir Zandieh et al. **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**. arXiv:2504.19874v1, 2025。重点：§2.2、§3.1–3.2、Theorem 1–2。
4. Zhiteng Li et al. **DVD-Quant: Data-free Video Diffusion Transformers Quantization**. arXiv:2505.18663v2, 2025。重点：§3.1–3.3、Table 1/3/5。
5. **From SmoothQuant to SVDQuant**，2025-12-15，中文二级解读材料。用于交叉检查直观解释，不作为数学命题或性能数字的第一来源。
