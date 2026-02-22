# Verl Fully Async Agentic RL：推理、Reward、Ref Model 计算流程

## 整体架构

在 fully async agentic RL 中，计算分布在 **Rollout 阶段**和 **Training 阶段**两个不同时期，分别使用独立的 GPU 资源池。

## 完整流程图

```
═══════════════════════  ROLLOUT 阶段 (Rollouter, rollout_pool GPU)  ═══════════════════════

 数据集
   ↓
 ① LLM 推理 (多轮 agent 交互)
   AgentLoopWorker → 推理 Server (vLLM/SGLang)
   生成 → 工具调用 → 生成 → ... → 终止
   产出: prompt_ids, response_ids, response_mask, rollout_log_probs
   ↓
 ② Reward 计算 (异步，流式)
   _compute_score() → RewardLoopWorker
   产出: reward_score 附加到 AgentLoopOutput
   ↓
 ③ 后处理 (_postprocess)
   padding、拼装 DataProto、构造 rm_scores tensor
   ↓
 MessageQueue (带 reward 的完整 sample)

═══════════════════════  TRAINING 阶段 (Trainer, trainer_pool GPU)  ═══════════════════════

   ↓ 从 MessageQueue 拉取 batch
 ④ Reward 补充计算 (_fit_compute_reward)
   若 rollout 阶段未算完，或需要 reward model 打分
   ↓
 ⑤ Old Log Prob 计算 (_fit_compute_log_prob)
   用当前 actor 重新算 rollout 时的 log prob（处理 staleness）
   ↓
 ⑥ Ref Log Prob 计算 (_fit_compute_ref_log_prob)
   用 reference model 计算 ref_log_prob
   ↓
 ⑦ Critic Value 计算 (_fit_compute_critic)
   计算 V(s) 值函数
   ↓
 ⑧ Advantage 计算 (_fit_compute_advantage)
   KL penalty: reward -= β * KL(old_log_prob || ref_log_prob)
   GAE / GRPO advantage estimation
   ↓
 ⑨ 更新 Critic (_fit_update_critic)
   ↓
 ⑩ 更新 Actor (_fit_update_actor)
   PPO clipped objective
   ↓
 ⑪ 权重同步 (_fit_update_weights)
   暂停 Rollouter → NCCL 同步 → 恢复 Rollouter
```

## 各阶段详解

### ① LLM 推理（Rollout 阶段）

发生在 `AgentLoopWorker` 中，调用推理 server 进行多轮生成。

产出的关键数据：
- `prompt_ids` / `response_ids`：token 序列
- `response_mask`：区分 LLM 生成的 token（1）和工具返回的 token（0）
- `rollout_log_probs`：生成时每个 token 的 log probability

### ② Reward 计算（Rollout 阶段，异步流式）

**在 rollout 阶段就开始算 reward**，不等到 training 阶段。

代码位置：`agent_loop/agent_loop.py:690-718`，`_compute_score()` 方法：

```python
async def _compute_score(self, output, ...):
    if self.reward_loop_worker_handles is not None:
        # 选一个 reward worker，异步计算
        result = await reward_worker.compute_score.remote(...)
        output.reward_score = result["reward_score"]
```

`RewardLoopWorker`（`reward_loop/reward_loop.py:92-268`）支持两种模式：
- **自定义 reward 函数**：规则型打分（如数学题答案校验）
- **Reward Model**：判别式 reward model 推理打分

reward 计算完后，在后处理阶段构造 `rm_scores` tensor（`agent_loop.py:754-759`）：

```python
scores = [input.reward_score for input in inputs]
rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
rm_scores[..., response_length] = torch.tensor(scores)  # 放在最后一个 token 位置
batch["rm_scores"] = rm_scores
```

sample 进入 MessageQueue 时，**已经携带了 reward**。

### ③ Reward 补充计算（Training 阶段）

`fully_async_trainer.py` 的 `fit_step()` 中调用 `_fit_compute_reward()`。

如果 rollout 阶段已经算好了 reward（`rm_scores` 已存在），这一步可能只做格式转换。如果有独立的 reward model 部署在 trainer 侧，则在此阶段补充计算。

### ④ Old Log Prob 重计算（Training 阶段）

由于 fully async 模式下 sample 可能是 **stale** 的（生成时用的参数版本 ≠ 当前参数版本），需要特殊处理。

代码位置：`fully_async_trainer.py:427-447`：

```python
def _compute_old_log_prob(self, batch):
    if self.local_trigger_step == 1:
        # 第一步：直接用当前 actor 算
        old_log_prob = super()._compute_old_log_prob(batch)
    else:
        # 后续步：恢复到生成时的参数版本，算完再切回来
        self.actor_rollout_wg.save_model_to_cpu(current_version)
        self.actor_rollout_wg.restore_model_from_cpu(version_1)
        old_log_prob = super()._compute_old_log_prob(batch)
        self.actor_rollout_wg.restore_model_from_cpu(current_version)
```

两种模式：
- **Bypass 模式**：直接用 rollout 阶段记录的 `rollout_log_probs`（省去重计算）
- **Decoupled 模式**：用 actor 重新前向计算

### ⑤ Ref Log Prob 计算（Training 阶段）

**Ref model 的计算完全在 training 阶段**，由 trainer_pool 的 GPU 执行。

代码位置：`ray_trainer.py:1099-1124`，`_compute_ref_log_prob()`：

```python
def _compute_ref_log_prob(self, batch):
    if self.ref_in_actor:
        # LoRA 模式：actor 关闭 LoRA adapter = ref model
        metadata["no_lora_adapter"] = True
        output = self.actor_rollout_wg.compute_log_prob(batch)
    else:
        # 独立 ref model worker
        output = self.ref_policy_wg.compute_ref_log_prob(batch)
    return ref_log_prob
```

两种部署方式：
- **`ref_in_actor=True`**（LoRA 训练时）：actor 关闭 LoRA adapter 就是 ref model，不需要额外 GPU
- **`ref_in_actor=False`**：独立的 `ref_policy_wg` worker group，占用 trainer_pool 中的 GPU

### ⑥ Advantage 计算（Training 阶段）

将 reward、old_log_prob、ref_log_prob、critic value 综合起来：

```python
# KL penalty（ray_trainer.py:69-100）
kld = core_algos.kl_penalty(old_log_probs, ref_log_prob)
token_level_rewards = token_level_scores - β * kld

# GAE advantage estimation
advantages = core_algos.compute_advantage(token_level_rewards, values, ...)
```

## 资源分配总结

| 计算任务 | 执行阶段 | 执行位置 | 所用资源 |
|----------|----------|----------|----------|
| LLM 推理（多轮生成） | Rollout | 推理 Server (vLLM/SGLang) | rollout_pool GPU |
| Reward 计算 | Rollout（流式） | RewardLoopWorker | rollout_pool GPU 或 CPU |
| Old Log Prob | Training | actor_rollout_wg | trainer_pool GPU |
| **Ref Log Prob** | **Training** | ref_policy_wg 或 actor (关 LoRA) | **trainer_pool GPU** |
| Critic Value | Training | critic_wg | trainer_pool GPU |
| Advantage + KL | Training | Trainer CPU | CPU |
| Actor/Critic 更新 | Training | actor/critic_wg | trainer_pool GPU |

## 关键设计点

- **Reward 尽量前置到 rollout 阶段异步计算**（与推理并行），减少 training 阶段的等待
- **Ref model 计算留在 training 阶段**，因为 ref model 参数固定，不需要和 rollout 同步，放在 trainer 侧用训练 GPU 计算即可
- **Old log prob 需要处理 staleness**，fully async 模式下 sample 生成时的参数版本可能已过期

## Rollouter 架构

Rollouter 在整个训练任务中是**唯一的（singleton）**，是 rollout 侧的总控进程，不直接占用 GPU。

```
FullyAsyncRollouter (唯一，协调者，Ray actor)
  │
  ├── resource_pool_manager
  │     └── rollout_pool: 例如 8 块 GPU
  │
  └── async_rollout_manager: FullyAsyncAgentLoopManager
        │
        ├── rollout_replicas: N 个推理 server 实例
        │     例如 8 GPU ÷ 2 (TP=2) = 4 个 replica
        │     ├── SGLang/vLLM Replica 0  (GPU 0,1)
        │     ├── SGLang/vLLM Replica 1  (GPU 2,3)
        │     ├── SGLang/vLLM Replica 2  (GPU 4,5)
        │     └── SGLang/vLLM Replica 3  (GPU 6,7)
        │
        └── agent_loop_workers: M 个 AgentLoopWorker
              ├── Worker 0 → 可调用任意 replica
              ├── Worker 1 → 可调用任意 replica
              ├── ...
              └── Worker 7 → 可调用任意 replica
```

## AgentLoopWorker 状态机

AgentLoopWorker 驱动多轮 agent 交互的状态机：

```
PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED
              ↑              ↓
              └──────────────┘  (循环直到结束条件)
```

1. **PENDING**：初始化 prompt，编码为 token IDs
2. **GENERATING**：调用推理 server 生成 tokens，支持 partial generation（可中途取消）
3. **PROCESSING_TOOLS**：解析工具调用，并行执行工具，拼接结果回 prompt
4. **TERMINATED**：达到 max_turns 或 response_length 上限，输出 AgentLoopOutput

### Partial Rollout 机制

权重同步时的中断恢复：

1. **暂停**：Manager 设置 `cancellation_event`，正在生成的 sample 中断，保存 partial output 到 `cancel_queue`
2. **同步**：NCCL 将新参数同步到推理 server
3. **恢复**：从 `cancel_queue` 取出 partial sample，从断点继续生成

## 完整数据流

```
Dataset
  ↓
Rollouter._feed_samples()          # 连续读取数据
  ↓ (pending_queue)
Rollouter._processor_worker()      # 异步处理
  ↓
AgentLoopManager                   # round-robin 分发
  ↓
AgentLoopWorker                    # 多轮 agent 交互循环
  ↓ (远程调用)
vLLM/SGLang Server                 # 推理生成
  ↓
RewardLoopWorker                   # 异步 reward 计算
  ↓
AgentLoopOutput → DataProto        # 后处理：padding、reward 拼装
  ↓
MessageQueue                       # 异步缓冲队列
  ↓
Trainer                            # Old log prob → Ref log prob → Critic → Advantage → PPO 更新
  ↓
ParameterSynchronizer              # 权重同步回 Rollouter
```
