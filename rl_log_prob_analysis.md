# verl 重要性采样（Importance Sampling）中三个 Log Prob 的作用分析

> **文档说明**：本文档详细分析 verl 框架中开启重要性采样时，`rollout_log_prob`、`old_log_prob`、`current_log_prob` 这三个变量如何参与 loss 计算。
>
> **参考文档**：
> - [Rollout Correction Usage Guide](verl/docs/algo/rollout_corr.md)
> - [Rollout Correction Mathematical Formulations](verl/docs/algo/rollout_corr_math.md)

---

## 目录

1. [概述](#概述)
2. [三个 Log Prob 的定义](#三个-log-prob-的定义)
3. [IS 权重计算流程](#is-权重计算流程)
4. [Bypass 模式：两策略框架](#bypass-模式两策略框架)
5. [REINFORCE Loss 计算](#reinforce-loss-计算)
6. [PPO Loss 计算](#ppo-loss-计算)
7. [Decoupled 模式：三策略框架](#decoupled-模式三策略框架)
8. [完整流程图](#完整流程图)
9. [关键代码位置](#关键代码位置)
10. [总结](#总结)

---

## 概述

在强化学习中，当**数据收集策略**（behavior policy）与**训练策略**（training policy）不一致时，需要使用**重要性采样**（Importance Sampling, IS）来修正这种分布偏差。

verl 框架通过 **Rollout Correction** 机制实现了这一修正，核心是计算三个策略的 log 概率：

```
π_rollout (rollout_log_prob)  →  数据收集策略
π_old (old_log_prob)          →  近端策略（PPO 锚点）
π_θ (current_log_prob)        →  当前训练策略
```

这三个 log prob 通过不同的组合方式，计算出 **IS 权重** 和 **PPO ratio**，从而实现对两种分布漂移的修正：
- **Drift 1**: π_rollout → π_old（off-policy 修正）
- **Drift 2**: π_old → π_θ（PPO 更新修正）

---

## 三个 Log Prob 的定义

### 1. **rollout_log_prob** (π_rollout)

**定义**：行为策略（Behavior Policy），数据收集时使用的策略。

**来源**：
- Rollout 阶段生成轨迹时计算
- 通常使用不同的实现（如 vLLM BF16、SGLang）
- 可能是过时的模型检查点（异步训练）
- 可能来自专家演示或辅助策略（DAPO）

**用途**：
- 作为 IS 权重计算的**分母**：`π_old / π_rollout` 或 `π_θ / π_rollout`
- 计算 off-policy 指标（KL 散度、困惑度）

**代码位置**：`verl/verl/workers/rollout/schemas.py`

```python
# 在 rollout 阶段生成
rollout_log_prob = rollout_worker.generate(prompts, calculate_log_probs=True)
```

---

### 2. **old_log_prob** (π_old)

**定义**：近端策略（Proximal Policy），PPO 裁剪的锚点策略。

**两种模式**：

#### **Decoupled 模式**（三策略）
- π_old 在训练开始时通过 `actor.compute_log_prob()` 独立计算
- 用于实现批次大小不变性（batch size invariance）
- 可以高效利用过时数据

#### **Bypass 模式**（两策略）
- **π_old = π_rollout**（直接使用 rollout 策略）
- 跳过额外的 forward pass，计算更快
- 不实现批次大小不变性

**用途**：
- **Decoupled 模式**：IS 权重分子 `π_old / π_rollout`，PPO ratio 分母 `π_θ / π_old`
- **Bypass 模式**：PPO ratio 分母 `π_θ / π_rollout`（π_old = π_rollout）

**代码位置**：`verl/verl/trainer/ppo/ray_trainer.py`

```python
# Decoupled 模式
old_log_prob = actor.compute_log_prob(batch)

# Bypass 模式
old_log_prob = rollout_log_prob  # 直接赋值
```

---

### 3. **current_log_prob** (π_θ)

**定义**：当前训练策略，每次梯度更新后的策略。

**来源**：
- Actor model 的 forward pass
- 每个 PPO epoch 中多次计算

**用途**：
- PPO ratio 的**分子**：`π_θ / π_old`
- Bypass 模式下 IS 权重的**分子**：`π_θ / π_rollout`
- 计算策略梯度：`∇log π_θ(a|s)`

**代码位置**：`verl/verl/workers/actor/dp_actor.py`

```python
# 每次 forward 计算
log_prob = actor_model(batch.sequences).log_probs
```

---

## IS 权重计算流程

### 核心函数：`compute_rollout_correction_weights`

**位置**：`verl/verl/trainer/ppo/rollout_corr_helper.py:481-594`

### 输入参数

```python
log_ratio = old_log_prob - rollout_log_prob  # log(π_old / π_rollout)
```

- **Decoupled 模式**：log_ratio = log(π_old / π_rollout)
- **Bypass 模式**：log_ratio = log(π_θ / π_rollout)（因为 old_log_prob = π_θ in bypass）

### Token 级别 IS 权重

```python
if rollout_is == "token":
    # 每个 token 独立计算权重
    # ρ_t = π_old(a_t|s_t) / π_rollout(a_t|s_t)
    log_ratio_safe = torch.clamp(log_ratio, min=-20, max=20)  # 防止数值溢出
    rollout_is_weights = torch.exp(log_ratio_safe)

    # 截断极端权重（TIS: Truncated Importance Sampling）
    rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)  # 通常 2.0

    # 零化 padding 位置
    rollout_is_weights = rollout_is_weights * response_mask
```

**特点**：
- 每个 token 独立截断
- 方差较低，但有偏差（bias）
- 适用于轻度 off-policy 场景
- 典型阈值：1.5 - 5.0

---

### Sequence 级别 IS 权重

```python
elif rollout_is == "sequence":
    # 整个序列共享一个权重
    # ρ_seq = ∏_t (π_old(a_t|s_t) / π_rollout(a_t|s_t))
    #       = exp(∑_t log(π_old / π_rollout))
    log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1)  # (batch_size,)
    log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-20, max=20)

    # 广播到所有 token（整个序列用同一个权重）
    rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)

    # 截断
    rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)  # 通常 2.0-10.0

    # 零化 padding
    rollout_is_weights = rollout_is_weights * response_mask
```

**特点**：
- 整个序列的概率乘积
- 无偏（unbiased），但方差更高
- 对异常值更敏感
- 典型阈值：2.0 - 10.0

---

### 梯度阻断（stopgrad）

```python
# 关键：阻断梯度流
rollout_is_weights = rollout_is_weights.detach()
```

**为什么必须 detach？**

这是**重要性采样理论的数学要求**，不是实现细节：

1. **IS 的本质**：改变采样分布的测度（reweight samples），而不是优化重加权函数本身

2. **正确的策略梯度**：
   ```
   ∇_θ J(θ) = E_π_rollout[w(θ) * ∇_θ log π_θ(a|s) * A]
   ```
   其中 `w(θ)` 出现是因为测度变换，不应对其求导

3. **不 detach 的后果**：
   ```python
   # 错误：会产生额外的偏差项
   ∇_θ [w(θ) * log π_θ] = log π_θ * ∇_θ w(θ)  ← 错误的偏差
                          + w(θ) * ∇_θ log π_θ  ← 正确的梯度
   ```

4. **PyTorch 实现**：
   ```python
   # 正确：stopgrad on weights
   loss = -advantages * log_prob * rollout_is_weights.detach()
   ```

参考：`verl/docs/algo/rollout_corr_math.md` §3.2.2

---

## Bypass 模式：两策略框架

**核心思想**：设置 **π_old = π_rollout**，跳过单独计算 old_log_prob 的 forward pass。

### 代码实现

**位置**：`verl/verl/trainer/ppo/core_algos.py:2065-2200`

```python
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,  # 在 Bypass 模式下实际是 rollout_log_prob
    log_prob: torch.Tensor,      # π_θ (当前策略)
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: ActorConfig,
):
    # Step 1: 确认在 Bypass 模式下 old_log_prob = rollout_log_prob
    rollout_log_prob = old_log_prob

    # Step 2: 计算 IS 权重和拒绝采样 mask
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,              # ← 注意：传入 current policy!
                rollout_log_prob=rollout_log_prob,  # ← rollout policy
                response_mask=response_mask,
                rollout_is="sequence",              # 或 "token"
                rollout_is_threshold=2.0,
                rollout_rs="seq_mean_k1",           # 可选：拒绝采样
                rollout_rs_threshold="0.999_1.001",
            )
        )

    # Step 3: 提取 IS 权重
    # IS 权重 = exp(log_prob - rollout_log_prob) = π_θ / π_rollout
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"]

    # Step 4: 根据 loss_type 选择损失函数
    loss_type = config.policy_loss.rollout_correction.loss_type

    if loss_type == "reinforce":
        # ===== REINFORCE 模式：显式使用 IS 权重 =====
        pg_loss, metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=modified_response_mask,
            rollout_is_weights=computed_is_weights,  # ← 应用 IS 权重
        )

    elif loss_type == "ppo_clip":
        # ===== PPO-clip 模式：ratio 已包含 IS，不需要额外权重 =====
        pg_loss, metrics = compute_policy_loss_vanilla(
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=modified_response_mask,
            rollout_is_weights=None,  # ← 明确设为 None，避免双重计算
        )

    return pg_loss, metrics
```

### 两种 Loss Type 的区别

| 特性 | `loss_type="reinforce"` | `loss_type="ppo_clip"` |
|------|-------------------------|------------------------|
| **Loss 函数** | L = -E[w · log π_θ · A] | L = -E[min(r·A, clip(r)·A)] |
| **IS 处理** | 显式应用 IS 权重 w = π_θ/π_rollout | ratio r = π_θ/π_rollout 隐式包含 |
| **Trust Region** | 仅通过 IS 截断 | PPO clipping (1-ε, 1+ε) |
| **IS 权重参数** | 必须传入 `rollout_is_weights` | 必须传入 `None`（避免双重计算）|
| **适用场景** | 需要精细 IS 控制 | 标准 PPO 训练 |

**关键警告**：在 PPO-clip 模式下，**不能同时应用 IS 权重**，因为 ratio 本身已经是 π_θ/π_rollout，会导致双重修正！

---

## REINFORCE Loss 计算

**位置**：`verl/verl/trainer/ppo/core_algos.py:1984-2061`

### 数学公式

**无 IS 权重**：
```
L = -E[log π_θ(a|s) * A(s,a)]
∇_θ L = -E[∇log π_θ(a|s) * A]  (标准 REINFORCE)
```

**有 IS 权重**：
```
L = -E_π_rollout[w * log π_θ(a|s) * A(s,a)]
其中 w = π_θ / π_rollout (truncated IS weight)
∇_θ L = -E[stopgrad(w) * ∇log π_θ(a|s) * A]  (IS-corrected policy gradient)
```

### 代码实现

```python
def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,  # π_rollout
    log_prob: torch.Tensor,          # π_θ
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_weights: Optional[torch.Tensor] = None,  # π_θ / π_rollout
    loss_agg_mode: str = "seq-mean-token-sum",
    config: ActorConfig = None,
) -> tuple[torch.Tensor, dict[str, Any]]:

    # 计算策略梯度 loss
    if rollout_is_weights is not None:
        # ===== IS 修正的策略梯度 =====
        # L = -E[stopgrad(w) · log π_θ · A]
        # w 已经 detach，作为常数系数
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # ===== 标准 REINFORCE =====
        # L = -E[log π_θ · A]
        pg_losses = -advantages * log_prob

    # 聚合 loss（按 token 或 sequence）
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # 计算 KL 散度（用于监控策略漂移）
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics
```

### 关键点

1. **IS 权重作为常数系数**：
   ```python
   pg_losses = -advantages * log_prob * rollout_is_weights
   #                                     ^^^^^^^^^^^^^^^^^^^ 已经 detach
   ```

2. **梯度只对 log_prob 求导**：
   ```python
   ∇_θ pg_loss = ∇_θ (-advantages * log_prob * rollout_is_weights)
               = -advantages * ∇_θ log_prob * rollout_is_weights  (w 作为常数)
   ```

3. **无 PPO clipping**：纯策略梯度，依赖 IS 截断控制更新步长

---

## PPO Loss 计算

**位置**：`verl/verl/trainer/ppo/core_algos.py:1160-1250`

### 数学公式

**标准 PPO**：
```
L_PPO(θ) = -E[min(r_t * A, clip(r_t, 1-ε, 1+ε) * A)]
其中 r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
```

**带 IS 权重的 PPO**（Decoupled 模式）：
```
L_PPO(θ) = -E[w_t * min(r_t * A, clip(r_t, 1-ε, 1+ε) * A)]
其中：
  w_t = π_old / π_rollout  (修正 Drift 1)
  r_t = π_θ / π_old        (修正 Drift 2，通过 clipping)
```

**Bypass 模式的 PPO**：
```
L_PPO(θ) = -E[min(r_t * A, clip(r_t, 1-ε, 1+ε) * A)]
其中 r_t = π_θ / π_rollout  (π_old = π_rollout，ratio 隐式包含 IS)
```

### 代码实现

```python
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,  # π_old (在 Bypass 下 = π_rollout)
    log_prob: torch.Tensor,      # π_θ
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: ActorConfig = None,
    rollout_is_weights: Optional[torch.Tensor] = None,  # Bypass 下为 None
) -> tuple[torch.Tensor, dict[str, Any]]:

    # Step 1: 计算 PPO ratio = π_θ / π_old
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)  # ratio = π_θ / π_old

    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    # Step 2: PPO clipping
    clip_ratio = config.clip_ratio  # 通常 0.2

    # 未 clip 的 loss
    pg_losses1 = -advantages * ratio

    # Clip 后的 loss
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - clip_ratio, 1 + clip_ratio
    )

    # 取两者的最大值（更保守的更新）
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # Step 3: Dual-clip（可选，防止过度惩罚）
    clip_ratio_c = config.get("clip_ratio_c", 3.0)  # 下界
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    # Step 4: 根据 advantage 符号选择
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Step 5: 应用 IS 权重（仅在 Decoupled 模式下）
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Step 6: 聚合 loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Step 7: 计算 clip 比例（用于监控）
    pg_clipfrac = masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }

    return pg_loss, pg_metrics
```

### Bypass 模式下的简化

在 Bypass 模式（`loss_type="ppo_clip"`）下：

```python
# old_log_prob = rollout_log_prob
ratio = exp(log_prob - old_log_prob)
      = exp(log_prob - rollout_log_prob)
      = π_θ / π_rollout  ← ratio 已经包含了 IS 修正！

# 因此不需要额外的 IS 权重
rollout_is_weights = None  # 必须是 None，避免双重修正
```

**为什么不能再应用 IS 权重？**

如果再乘以 `w = π_θ / π_rollout`，会得到：
```
loss = w * ratio * A
     = (π_θ / π_rollout) * (π_θ / π_rollout) * A
     = (π_θ / π_rollout)² * A  ← 错误！双重修正
```

---

## Decoupled 模式：三策略框架

**核心思想**：单独计算 π_old，实现批次大小不变性（batch size invariance）。

### 策略角色

1. **π_rollout**：行为策略（数据收集）
2. **π_old**：近端策略（PPO 锚点，训练开始时计算）
3. **π_θ**：当前策略（正在更新）

### 两种分布漂移

**Drift 1: π_rollout → π_old**（Off-Policy Gap）
- **来源**：rollout 实现与训练实现不同、模型过时、回放缓冲区
- **修正方式**：IS 权重 `w = π_old / π_rollout`

**Drift 2: π_old → π_θ**（Policy Update Drift）
- **来源**：梯度下降更新 θ
- **修正方式**：PPO clipping on ratio `r = π_θ / π_old`

### Loss 公式

```
L_DecoupledPPO(θ) = -E_π_rollout [w_t * min(r_t * A, clip(r_t, 1-ε, 1+ε) * A)]

其中：
  w_t = π_old(a_t|s_t) / π_rollout(a_t|s_t)  ← IS 权重（修正 Drift 1）
  r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)       ← PPO ratio（修正 Drift 2）
```

### 代码流程

```python
# Step 1: Rollout 阶段（数据收集）
rollout_log_prob = vllm_worker.generate(prompts)  # π_rollout

# Step 2: 训练开始时计算 π_old
old_log_prob = actor.compute_log_prob(batch)  # π_old (额外 forward pass)

# Step 3: 计算 IS 权重（修正 Drift 1）
log_ratio = old_log_prob - rollout_log_prob  # log(π_old / π_rollout)
rollout_is_weights = exp(clamp(log_ratio, -20, 20)).clamp(max=2.0)
rollout_is_weights = rollout_is_weights.detach()

# Step 4: PPO 更新循环
for epoch in range(ppo_epochs):
    # 计算当前策略的 log prob
    log_prob = actor(batch)  # π_θ

    # 计算 PPO ratio（修正 Drift 2）
    ratio = exp(log_prob - old_log_prob)  # π_θ / π_old

    # PPO clipping
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clamp(ratio, 1-ε, 1+ε)
    pg_losses = maximum(pg_losses1, pg_losses2)

    # 应用 IS 权重
    pg_losses = pg_losses * rollout_is_weights  # w_t * loss

    # 聚合并反向传播
    loss = agg_loss(pg_losses, response_mask)
    loss.backward()
    optimizer.step()
```

### 与 Bypass 模式的对比

| 特性 | Decoupled 模式 | Bypass 模式 |
|------|---------------|------------|
| **策略数量** | 3（π_rollout, π_old, π_θ）| 2（π_rollout = π_old, π_θ）|
| **π_old 计算** | 独立计算（额外 forward pass）| π_old = π_rollout |
| **IS 权重** | w = π_old / π_rollout | w = π_θ / π_rollout (仅 REINFORCE) |
| **PPO ratio** | r = π_θ / π_old | r = π_θ / π_rollout |
| **批次大小不变性** | ✅ 是 | ❌ 否 |
| **计算成本** | 较高（额外 forward）| 较低 |
| **适用场景** | 严重 off-policy、回放缓冲区 | 轻度 off-policy、在线训练 |

---

## 完整流程图

### 1. Bypass + REINFORCE 模式

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据收集阶段 (Rollout)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    rollout_log_prob (π_rollout)
                                │
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段 (Training)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    current_log_prob (π_θ)
                                │
                                │
                    ┌───────────┴───────────┐
                    │                       │
         rollout_log_prob          current_log_prob
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                log_ratio = log_prob - rollout_log_prob
                          = log(π_θ / π_rollout)
                                │
                                ▼
                    ┌───────────────────────┐
                    │  计算 IS 权重          │
                    │  w = exp(log_ratio)   │
                    │    = π_θ / π_rollout  │
                    └───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
        ┌───────────▼───────────┐  ┌───────▼────────┐
        │  安全边界裁剪          │  │  响应掩码       │
        │  clamp(-20, 20)       │  │  response_mask │
        └───────────┬───────────┘  └───────┬────────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  截断极端权重          │
                    │  clamp(max=threshold) │
                    │  (TIS)                │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  梯度阻断              │
                    │  w = w.detach()       │
                    │  (stopgrad)           │
                    └───────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  REINFORCE Loss               │
                │  L = -E[w * log π_θ * A]      │
                │                               │
                │  w: 常数系数 (stopgrad)        │
                │  log π_θ: 参与梯度计算         │
                └───────────────────────────────┘
                                │
                                ▼
                        ∇_θ L = -E[w * ∇log π_θ * A]
```

---

### 2. Bypass + PPO-clip 模式

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据收集阶段 (Rollout)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        rollout_log_prob (π_rollout) = old_log_prob
                                │
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段 (Training)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    current_log_prob (π_θ)
                                │
                                │
                    ┌───────────┴───────────┐
                    │                       │
              old_log_prob          current_log_prob
           (= rollout_log_prob)            │
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                log_ratio = log_prob - old_log_prob
                          = log_prob - rollout_log_prob
                          = log(π_θ / π_rollout)
                                │
                                ▼
                    ┌───────────────────────┐
                    │  计算 PPO ratio        │
                    │  ratio = exp(log_ratio)│
                    │        = π_θ / π_rollout│
                    │                       │
                    │  ⚠️ ratio 已经包含 IS!  │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  PPO Clipping         │
                    │  pg1 = -A * ratio     │
                    │  pg2 = -A * clip(ratio)│
                    │  loss = max(pg1, pg2) │
                    └───────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  ⚠️ 不应用额外的 IS 权重!      │
                │  rollout_is_weights = None    │
                │                               │
                │  理由：ratio 已经是 π_θ/π_rollout│
                │  再乘 IS 权重会导致双重修正!     │
                └───────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  PPO Loss                     │
                │  L = -E[min(ratio*A,          │
                │            clip(ratio)*A)]    │
                │                               │
                │  ratio 隐式包含 IS 修正        │
                └───────────────────────────────┘
```

---

### 3. Decoupled 模式（三策略）

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据收集阶段 (Rollout)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    rollout_log_prob (π_rollout)
                                │
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    训练开始时计算 π_old                          │
│                 (额外的 forward pass)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                old_log_prob (π_old) = actor.compute_log_prob(batch)
                                │
                                │
                ┌───────────────┴───────────────┐
                │   计算 IS 权重（修正 Drift 1）  │
                │   log_ratio = old_log_prob -   │
                │               rollout_log_prob │
                │   w = exp(log_ratio).detach()  │
                │     = π_old / π_rollout        │
                └───────────────┬───────────────┘
                                │
                                ▼
                    rollout_is_weights (w)
                                │
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     PPO 更新循环 (多个 epoch)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    current_log_prob (π_θ)
                                │
                                │
                ┌───────────────┴───────────────┐
                │   计算 PPO ratio（修正 Drift 2）│
                │   ratio = exp(log_prob -       │
                │               old_log_prob)    │
                │         = π_θ / π_old          │
                └───────────────┬───────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  PPO Clipping         │
                    │  pg1 = -A * ratio     │
                    │  pg2 = -A * clip(ratio)│
                    │  loss = max(pg1, pg2) │
                    └───────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  应用 IS 权重                  │
                │  loss = loss * w              │
                │                               │
                │  w = π_old / π_rollout        │
                └───────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  Decoupled PPO Loss           │
                │  L = -E[w * min(ratio*A,      │
                │              clip(ratio)*A)]  │
                │                               │
                │  w: 修正 Drift 1 (rollout→old)│
                │  ratio: 修正 Drift 2 (old→θ)  │
                └───────────────────────────────┘
                                │
                                ▼
                        反向传播并更新参数
```

### 关键区别总结

| 模式 | 策略数量 | IS 权重 | PPO Ratio | 修正内容 |
|------|---------|---------|-----------|---------|
| **Bypass + REINFORCE** | 2 | π_θ / π_rollout (显式) | 无 | 单一漂移 |
| **Bypass + PPO-clip** | 2 | 无（隐式在 ratio 中）| π_θ / π_rollout | 单一漂移 |
| **Decoupled** | 3 | π_old / π_rollout | π_θ / π_old | 双重漂移 |

---

## 关键代码位置

### 1. IS 权重计算

**主函数**：`verl/verl/trainer/ppo/rollout_corr_helper.py`

```python
# Line 481-594: 核心 IS 权重计算
def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...

# Line 726-831: 统一入口（IS + RS + Metrics）
def compute_rollout_correction_and_rejection_mask(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: float = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[str | float] = None,
    rollout_is_batch_normalize: bool = False,
) -> tuple[Optional[DataProto], torch.Tensor, dict[str, float]]:
    ...
```

---

### 2. Bypass 模式损失函数

**主函数**：`verl/verl/trainer/ppo/core_algos.py`

```python
# Line 2065-2200: Bypass 模式入口
@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,  # = rollout_log_prob
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ...
```

---

### 3. REINFORCE 损失函数

```python
# Line 1984-2061: REINFORCE 损失计算
def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """REINFORCE-style policy gradient loss with optional IS correction."""

    if rollout_is_weights is not None:
        # Line 2040: IS-corrected policy gradient
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Line 2043: Standard REINFORCE
        pg_losses = -advantages * log_prob
    ...
```

---

### 4. PPO 损失函数

```python
# Line 1160-1250: PPO clipped objective
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy objective and related metrics for PPO."""

    # Line 1210-1213: 计算 ratio
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # Line 1216-1235: PPO clipping
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    ...

    # Line 1238-1239: 应用 IS 权重（仅 Decoupled 模式）
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    ...
```

---

### 5. Trainer 主流程

**Decoupled 模式**：`verl/verl/trainer/ppo/ray_trainer.py`

```python
# 计算 old_log_prob（额外 forward pass）
old_log_prob_output = actor_rollout_ref.compute_log_prob(data)
old_log_prob = old_log_prob_output.batch['old_log_prob']

# PPO 更新循环
for _ in range(ppo_epochs):
    actor_output = actor_rollout_ref.update_actor(data)
```

**Bypass 模式**：`verl/verl/trainer/ppo/ray_trainer.py`

```python
# 跳过 old_log_prob 计算
if config.algorithm.rollout_correction.bypass_mode:
    # 直接使用 rollout_log_prob
    data.batch['old_log_prob'] = data.batch['rollout_log_prob']
```

---

### 6. GRPO 优势估计

**位置**：`verl/verl/trainer/ppo/core_algos.py:267-330`

```python
# Line 267-330: GRPO 优势函数计算
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    GRPO 使用组内归一化：
    A_i = (R_i - mean(R_group)) / std(R_group)
    """
    scores = token_level_rewards.sum(dim=-1)

    # 按 group 计算均值和标准差
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx] = torch.std(scores_tensor)

    # 归一化
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]

    return scores, scores
```

---

### 7. 配置文件

**算法配置**：`verl/verl/trainer/config/algorithm.py`

```python
@dataclass
class RolloutCorrectionConfig:
    """Rollout correction configuration for IS/RS."""

    rollout_is: Optional[str] = None  # "token", "sequence", or None
    rollout_is_threshold: float = 2.0
    rollout_is_batch_normalize: bool = False
    rollout_rs: Optional[str] = None  # e.g., "seq_mean_k1", "token_k1"
    rollout_rs_threshold: Optional[Union[str, float]] = None
    bypass_mode: bool = False  # Skip old_log_prob computation
    loss_type: str = "ppo_clip"  # "ppo_clip" or "reinforce"

    @classmethod
    def bypass_pg_is(cls, threshold: float = 2.0):
        """Bypass mode with REINFORCE + Seq-TIS."""
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_ppo_clip(cls):
        """Bypass mode with PPO-clip (no explicit IS weights)."""
        return cls(
            rollout_is=None,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="ppo_clip",
        )
```

---

## 总结

### 三个 Log Prob 的核心作用

| Log Prob | 策略 | 主要用途 |
|----------|------|---------|
| **rollout_log_prob** | π_rollout | IS 权重分母；off-policy 指标基准 |
| **old_log_prob** | π_old | IS 权重分子（Decoupled）；PPO ratio 分母 |
| **current_log_prob** | π_θ | PPO ratio 分子；策略梯度计算 |

---

### IS 权重公式

**Decoupled 模式**：
```
w = π_old(a_t|s_t) / π_rollout(a_t|s_t)
  = exp(old_log_prob - rollout_log_prob)
```

**Bypass 模式（REINFORCE）**：
```
w = π_θ(a_t|s_t) / π_rollout(a_t|s_t)
  = exp(current_log_prob - rollout_log_prob)
```

**Bypass 模式（PPO-clip）**：
```
不使用单独的 IS 权重，ratio 隐式包含：
ratio = π_θ(a_t|s_t) / π_rollout(a_t|s_t)
```

---

### 处理流程

1. **安全边界裁剪**：`clamp(log_ratio, -20, 20)` → 防止数值溢出
2. **截断**：`clamp(weights, max=threshold)` → TIS 方差缩减
3. **梯度阻断**：`weights.detach()` → IS 理论要求
4. **应用到 loss**：
   - REINFORCE: `loss = -w * log π_θ * A`
   - PPO-clip: `loss = -min(ratio*A, clip(ratio)*A)` (ratio 已包含 IS)

---

### 关键注意事项

1. **梯度阻断是必须的**：
   ```python
   rollout_is_weights = rollout_is_weights.detach()
   ```
   这是 IS 理论的数学要求，不是实现细节。

2. **Bypass + PPO-clip 不能再应用 IS 权重**：
   ```python
   # 错误示例
   loss = ratio * advantages * rollout_is_weights  # ❌ 双重修正

   # 正确做法
   loss = ratio * advantages  # ✅ ratio 已包含 IS
   ```

3. **Token vs Sequence 级别的选择**：
   - Token 级别：低方差，有偏差，适合轻度 off-policy
   - Sequence 级别：无偏，高方差，适合严重 off-policy

4. **Decoupled vs Bypass 的选择**：
   - Decoupled：批次大小不变性，适合回放缓冲区
   - Bypass：计算高效，适合在线训练

---

### 参考文档

- **实现指南**：[verl/docs/algo/rollout_corr.md](verl/docs/algo/rollout_corr.md)
- **数学理论**：[verl/docs/algo/rollout_corr_math.md](verl/docs/algo/rollout_corr_math.md)
- **GRPO 文档**：[verl/docs/algo/grpo.md](verl/docs/algo/grpo.md)
- **核心代码**：`verl/verl/trainer/ppo/core_algos.py`
- **IS Helper**：`verl/verl/trainer/ppo/rollout_corr_helper.py`

---

**最后更新**：2026-02-05
