# DrMAS 多Agent场景优势值计算详解

## 1. 背景：单Agent GRPO的优势值计算

对于单agent场景，GRPO算法会将一个样本复制K份（一组），对这一组样本进行rollout，对rollout生成的一组结果计算reward分数，用一组的均值作为baseline，从而避免引入critic的复杂性。

DrMAS在多Agent强化学习场景中对这种算法进行了修正，核心是**按agent独立计算baseline**，实现per-agent的信用分配。

## 2. 数据收集：一个题目复制K份

和单agent GRPO一样，DrMAS也会把每个数学题复制K份（默认K=8，由 `env.rollout.n` 配置）。假设一个batch有N个题目，复制后就有 N×K 个样本。

每组K个样本共享同一个 `traj_uid`（标识它们来自同一道题）和同一个 `uid`（标识同一组）。

## 3. 多Agent协作执行：每个样本产生多条训练数据

以 Math Orchestra 为例（`math_orchestra.py:76-122`），对于每个样本，执行流程是：

```
循环 max_loop_num 次:
    1. Solver Agent 对未通过验证的样本生成解答
    2. Verifier Agent 对这些样本进行验证
    3. 验证通过的标记为 approved，不再参与后续循环
```

关键点：**每个agent每次调用都会产生一条独立的训练数据**（保存在 `multiagent_batch_buffer` 中）。因此对于一个样本，可能产生多条训练记录，每条带有不同的 `agent_id`（如 `"Solver Agent"` 或 `"Verifier Agent"`）。

在 `rollout_loop.py:470-481`，所有agent的输出被统一打上：
- `agent_id`：标识哪个agent产生的
- `uid`：标识所属的组
- `traj_uid`：标识所属的轨迹（同一道题的K个样本共享）
- `rewards`：**最终环境返回的reward**（所有agent共享同一个最终结果reward）

## 4. 核心差异：`group_by_agent_id` 参数

这是DrMAS对标准GRPO的关键修正，代码在 `ray_trainer.py:269-274` 和 `core_algos.py:497-569`。

### 标准GRPO（`group_by_agent_id=False`）

分组索引 = `uid`（仅按题目分组）

```python
group_index = data.non_tensor_batch["uid"]  # 如 "problem_0"
```

一组K个样本中，Solver和Verifier的数据**混在一起**计算均值/标准差。所有agent共享同一个baseline。

### DrMAS模式（`group_by_agent_id=True`）

分组索引 = `uid` + `agent_id`（按题目**和**agent联合分组）

```python
group_index = np.array([f"{uid}_{agent_id}" for uid, agent_id in
    zip(data.non_tensor_batch["uid"], data.non_tensor_batch["agent_id"])])
# 如 "problem_0_Solver Agent", "problem_0_Verifier Agent"
```

这意味着 **Solver 和 Verifier 各自独立计算 baseline**。

## 5. 优势值计算的具体数学过程

核心函数在 `core_algos.py:497-569`：

```python
def compute_grpo_outcome_advantage(..., group_by_agent_id=False):
    scores = token_level_rewards.sum(dim=-1)  # 每个样本的标量reward

    # Step 1: 按 (group_index, traj_uid) 聚合reward
    for i in range(bsz):
        traj_accumulator[(index[i], traj_index[i])].append(scores[i])

    # Step 2: 关键分支
    for (idx, t_idx), reward_list in traj_accumulator.items():
        if group_by_agent_id:
            # DrMAS: 直接收集每个agent的独立reward
            id2score[idx].extend(reward_list)
        else:
            # 标准GRPO: 先按轨迹取平均，再收集
            avg_score = torch.stack(reward_list).mean()
            id2score[idx].append(avg_score)

    # Step 3: 按组计算均值和标准差
    for idx in id2score:
        id2mean[idx] = mean(id2score[idx])
        id2std[idx] = std(id2score[idx])

    # Step 4: 计算优势值
    advantage_i = (score_i - mean_group) / (std_group + eps)
```

## 6. 具体例子说明

假设1道数学题，K=4（复制4份），Solver和Verifier各生成一次响应：

```
题目 P，4份样本的最终reward分别为: [1, 0, 1, 1]

标准GRPO (group_by_agent_id=False):
  所有样本的group_index = "P"
  先按traj_uid取平均(如果单轨迹就是自身)，然后统一计算:
    mean = 0.75, std = 0.5
    Solver样本1的 advantage = (1 - 0.75) / 0.5 = 0.5
    Verifier样本1的 advantage = (1 - 0.75) / 0.5 = 0.5
  → Solver和Verifier使用相同的baseline

DrMAS (group_by_agent_id=True):
  Solver的 group_index = "P_Solver Agent"
  Verifier的 group_index = "P_Verifier Agent"

  Solver组 (4个样本的reward): [1, 0, 1, 1]
    mean_solver = 0.75, std_solver = 0.5
    Solver样本1的 advantage = (1 - 0.75) / 0.5 = 0.5

  Verifier组 (4个样本的reward): [1, 0, 1, 1]
    mean_verifier = 0.75, std_verifier = 0.5
    Verifier样本1的 advantage = (1 - 0.75) / 0.5 = 0.5
```

在这个简化例子中结果一样，但实际场景中：
- **不同agent可能有不同数量的活跃样本**（通过 `active_mask` 控制）
- **在多轮循环中（max_loop_num > 1），Solver 会产生多次输出**，这些输出各有不同的reward，但都按 Solver 自己的分组来计算baseline
- 当agent不共享模型时，**每个agent用各自的baseline来更新各自的策略**，避免了另一个agent的表现波动影响自身的梯度信号

## 7. 总结修正要点

| 维度 | 标准单Agent GRPO | DrMAS 多Agent GRPO |
|------|------------------|-------------------|
| 分组键 | `uid`（仅按题目） | `uid_agentId`（按题目+agent） |
| Baseline | 所有agent共享 | 每个agent独立 |
| 优势值 | 相对于全局组均值 | 相对于本agent组均值 |
| 效果 | agent间互相干扰 | **独立信用分配** |

配置上只需设置 `group_by_agent_id=True`（在 `run_math.sh` 中），就能启用这种 per-agent 的优势值计算。

## 8. 关键代码文件索引

| 组件 | 文件 | 行号 |
|------|------|------|
| 多Agent轨迹收集 | `agent_system/multi_turn_rollout/rollout_loop.py` | 370-500 |
| 数学Orchestra | `agent_system/agent/orchestra/math/math_orchestra.py` | 48-122 |
| GRPO优势值计算 | `verl/trainer/ppo/core_algos.py` | 497-569 |
| 优势值计算调用入口 | `verl/trainer/ppo/ray_trainer.py` | 247-377 |
| group_by_agent_id配置读取 | `verl/trainer/ppo/ray_trainer.py` | 1355-1373 |
| 训练脚本配置 | `examples/drmas_trainer/run_math.sh` | - |
