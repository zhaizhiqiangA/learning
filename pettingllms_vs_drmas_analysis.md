# PettingLLMs vs DrMAS: Multi-Agent RL System 深度对比与最优架构设计

## 一、两个系统的核心设计哲学差异

| 维度 | PettingLLMs | DrMAS |
|------|-------------|-------|
| **设计哲学** | 从Multi-Agent环境出发，强调agent间交互和环境状态流转 | 从RL训练框架出发，强调agent-aware的PPO训练流程 |
| **Agent定义** | 继承 `BaseAgent` dataclass，agent是有状态实体（prompt/action/reward/done） | 继承 `BaseAgent` class，agent是无状态函数式调用（call接口） |
| **协调机制** | Execution Engine 控制 turn-based/graph/autoevol 三种范式 | Orchestra 模式：Math(Solver-Verifier迭代) / Search(Verifier-Router条件分支) |
| **模型架构** | L0/L1/L2/L3 四级specialization（prompt/LoRA/full model） | Worker Group 映射：model_sharing / non-sharing / heterogeneous |
| **verl集成** | 外部封装verl，自建trainer调用verl的worker group | fork verl并深度修改ray_trainer.py，在verl内部实现multi-agent逻辑 |
| **Advantage算法** | AT-GRPO: uid=(env_id, turn_id, agent_id) | group_by_agent_id: uid=f"{uid}_{agent_id}" + GiGPO step-level reward |

---

## 二、Rollout生成流程对比

### PettingLLMs: 环境驱动的多轮交互

```
MultiAgentsExecutionEngine.generate_single_rollout():
│
├─ for turn_idx in range(max_turns):
│   ├─ for agent in turn_order:              # 按预定义顺序
│   │   ├─ agent.update_from_env(env)        # 从env获取观察
│   │   ├─ agent.update_from_model(env)      # 构造prompt
│   │   ├─ llm_async_generate()              # vLLM异步生成
│   │   ├─ agent.step(env)                   # 执行action（可含tool call）
│   │   ├─ agent.calculate_reward(env)       # 计算per-agent reward
│   │   └─ output_dpr → trajectory_dict      # 每个agent-turn独立DataProto
│   └─ check env.done
│
└─ 所有agent-turn的DataProto concat为训练batch
```

**关键特点：**
- Agent是有状态的，维护 `current_prompt`, `current_action`, `agent_reward`, `success`, `done`
- 环境状态 (`EnvState`) 在agent间传递（如 `MathEnvState` 跟踪 reasoning_solution + code_solution + history）
- 支持Tree-Structured Sampling：每个agent step做Best-of-N分支
- 每个agent-turn生成独立的 `(prompt, response)` DataProto，最终flatten为训练batch

### DrMAS: Orchestra驱动的Step-Level协调

```
MultiAgentTrajectoryCollector.vanilla_multi_turn_loop():
│
├─ envs.reset() → obs
├─ for _step in range(max_steps):
│   ├─ orchestra.run(gen_batch, obs, actor_rollout_wgs, active_masks, step)
│   │   ├─ MathOrchestra:
│   │   │   ├─ for loop_i in range(max_loop_num):
│   │   │   │   ├─ SolverAgent.call() → 生成方案 (active on unapproved)
│   │   │   │   └─ VerifierAgent.call() → approve/reject
│   │   │   └─ 返回 text_actions + multiagent_batch_buffer
│   │   └─ SearchOrchestra:
│   │       ├─ VerifierAgent.call() → route决策
│   │       ├─ if insufficient → SearchAgent.call()
│   │       └─ if sufficient → AnswerAgent.call()
│   │
│   ├─ envs.step(text_actions) → rewards, dones
│   ├─ 为每个agent的batch打上 agent_id, uid, traj_uid, rewards, active_masks
│   └─ active items追加到 total_batch_list
│
└─ gather_rollout_data() → 组装最终DataProto
```

**关键特点：**
- Agent是无状态的函数式调用，每次 `call()` 接收完整上下文（env_obs + team_context）
- Orchestra内部可有多轮循环（Math的Solver-Verifier迭代最多max_loop_num次）
- 使用 `active_masks` 和 `approved_vector` 控制哪些batch item需要哪个agent处理
- Reward由环境统一返回（episode reward），非per-agent独立计算
- 所有agent在同一step内的输出存入 `multiagent_batch_buffer`，按 `wg_id` 标记

### 核心差异

```
PettingLLMs:  env_state ←→ agent（双向有状态交互）
              每个agent-turn独立生成DataProto
              turn_order固定，所有agent顺序执行

DrMAS:        orchestra(env_obs, team_context) → text_actions（函数式）
              orchestra内部可有复杂的条件/迭代逻辑
              batch_buffer收集所有agent输出统一处理
              active_mask控制条件执行（非所有agent每步都需要运行）
```

---

## 三、训练更新流程对比

### PettingLLMs 训练流程

```
fit() 主循环:
│
├─ 1. wake_up() vLLM engines
├─ 2. generate_multiple_rollouts_concurrent()  # 多rollout并发
├─ 3. sleep() vLLM engines
├─ 4. _assign_consistent_uids()  # AT-GRPO uid=(env_id//N, turn_id, agent_id)
├─ 5. _update_parameters(batch, ppo_trainer):
│   ├─ pad prompts/responses
│   ├─ 构造 token_level_rewards (reward放在最后一个valid token)
│   ├─ compute_log_prob (一次，所有agent数据)
│   ├─ compute_ref_log_prob
│   ├─ compute_values (if critic)
│   ├─ compute_advantage (GRPO with AT-GRPO uid)
│   ├─ if lora_differ_mode:  # L2
│   │   ├─ 按agent_name拆分sub_batch
│   │   └─ 逐agent调用 update_actor(agent_batch)  # 顺序独立更新
│   └─ else:  # L1
│       └─ update_actor(batch)  # 一次更新，所有agent数据混合
└─ 6. save checkpoint / validate
```

### DrMAS 训练流程

```
fit() 主循环:
│
├─ 1. traj_collector.multi_turn_loop()  # 多轮交互收集trajectory
├─ 2. split_batch_by_wg_ids()  # 按worker group拆分
├─ 3. adjust_batch() per wg  # 每个agent独立batch size调整
├─ 4. combine_batches()  # 合并回统一batch
├─ 5. compute_reward()  # EpisodeRewardManager
├─ 6. split → compute_log_prob per wg → combine  # 每个wg独立log_prob
├─ 7. split → compute_ref_log_prob per wg → combine
├─ 8. compute_values (if critic, 统一)
├─ 9. compute_advantage:
│   ├─ if group_by_agent_id:
│   │   group_index = f"{uid}_{agent_id}"  # agent-aware分组
│   ├─ GRPO: 按group_index分组normalize
│   ├─ GiGPO: step_rewards + episode_rewards + anchor similarity
│   └─ RLOO/REINFORCE++/REMAX等多种estimator
├─ 10. split → update_actor per wg → combine  # 每个wg独立更新
└─ 11. save checkpoint / validate
```

### 训练更新的关键差异

| 维度 | PettingLLMs | DrMAS |
|------|-------------|-------|
| **Batch拆分粒度** | L1: 不拆分；L2: 按agent_name拆分 | 始终按wg_id拆分（支持不同model） |
| **拆分-合并模式** | 仅在update_actor时拆分 | 反复 split→process→combine（log_prob/ref/update各一次） |
| **Advantage分组** | uid=(env_id//N, turn_id, agent_id) 静态构造 | uid_{agent_id} 动态拼接，可选开关 |
| **Advantage算法** | 仅GRPO | GRPO + GiGPO + RLOO + REINFORCE++ + GAE等 |
| **Step-level reward** | 无（仅outcome reward） | GiGPO支持step-level discounted returns |
| **Invalid action处理** | 无 | apply_invalid_action_penalty |
| **Reward计算** | per-agent `calculate_reward()` 在rollout阶段 | EpisodeRewardManager 在训练阶段统一计算 |
| **多model支持** | L3模式用多PPO trainer dict | 天然支持多wg，每个wg独立model |
| **Dynamic sampling** | 无 | DAPO-style filter_group_data去除同质reward |

---

## 四、Agent协调与可扩展性对比

### PettingLLMs 协调模型

```
三种Execution Engine:

1. Turn-Based (FSM):
   固定turn_order → agent顺序执行 → env状态传递
   适合：固定角色分工（Reasoning + Tool Agent）

2. Graph-Based (AutoGen):
   AutoGen workflow DAG → agent按图执行
   适合：复杂依赖关系的workflow

3. AutoEvol:
   LLM生成MAS代码 → 执行生成的代码 → 评估效果
   适合：自动搜索最优agent协作模式
```

### DrMAS 协调模型

```
Orchestra模式（插件式）:

1. MathOrchestra（迭代式）:
   Solver → Verifier → (reject → Solver → Verifier → ...) → approve
   active_mask控制：只处理尚未approved的items

2. SearchOrchestra（条件分支式）:
   Verifier(Router) → if sufficient: AnswerAgent
                     → if insufficient: SearchAgent
   verification_vector控制分支路由

3. 自定义Orchestra:
   继承BaseOrchestra，实现run()方法
   AgentRegistry动态注册agent
```

### 可扩展性对比

| 维度 | PettingLLMs | DrMAS |
|------|-------------|-------|
| **添加新agent** | 继承BaseAgent，实现5个抽象方法 | @AgentRegistry.register + 实现call() |
| **添加新协调模式** | 新建ExecutionEngine子类（复杂） | 新建Orchestra子类（轻量） |
| **添加新环境** | 继承BaseEnv + BaseEnvState + 各Agent | 实现envs接口 + projection函数 |
| **Agent间通信** | 通过共享EnvState | 通过team_context字符串 |
| **条件执行** | 有限（skip_current_turn） | 完善（active_mask + approved_vector） |
| **Agent注册** | 无（硬编码在config中） | AgentRegistry工厂模式 |
| **Memory** | Agent内部状态 + EnvState history | 独立Memory模块（SimpleMemory/SearchMemory） |

---

## 五、最优Multi-Agent RL System架构设计

结合两者优势，设计最优架构：

### 设计原则

1. **DrMAS的Orchestra模式** 用于agent协调（灵活、可插拔）
2. **PettingLLMs的多级specialization** 用于模型管理（prompt/LoRA/full model）
3. **DrMAS的wg_id分拆合并模式** 用于训练流程（天然支持多model）
4. **PettingLLMs的AT-GRPO分组** 结合 **DrMAS的GiGPO** step-level reward
5. **verl的异步训推能力** 用于性能优化

### 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Unified Multi-Agent RL System                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Layer 1: Agent Definition                      │   │
│  │                                                                  │   │
│  │  AgentRegistry  ←── @register decorator                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │BaseAgent │ │ToolAgent │ │Verifier  │ │Custom    │           │   │
│  │  │(stateful)│ │(tool use)│ │Agent     │ │Agent     │           │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │   │
│  │       │  call() + state management + Memory module               │   │
│  │       └─────────────┴─────────────┴────────────┘                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │               Layer 2: Orchestration Engine                      │   │
│  │                                                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │ Sequential  │  │ Conditional │  │  Graph/DAG  │             │   │
│  │  │ (turn-based)│  │ (Router)    │  │ (AutoGen)   │             │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │   │
│  │         │                │                │                     │   │
│  │         └────────────────┴────────────────┘                     │   │
│  │                         │                                       │   │
│  │              BaseOrchestra.run()                                 │   │
│  │              - active_mask管理                                   │   │
│  │              - team_context累积                                  │   │
│  │              - multiagent_batch_buffer收集                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │             Layer 3: Model Management (Specialization)           │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────┐       │   │
│  │  │         WorkerGroupManager                            │       │   │
│  │  │                                                      │       │   │
│  │  │  L1 Prompt:    All agents → 1 WG (shared base)      │       │   │
│  │  │  L2 LoRA:      All agents → 1 WG (shared base       │       │   │
│  │  │                 + per-agent LoRA adapter)             │       │   │
│  │  │  L3 Full:      Agent → dedicated WG (separate model) │       │   │
│  │  │  Hybrid:       混合模式（部分shared + 部分dedicated） │       │   │
│  │  │                                                      │       │   │
│  │  │  agent_to_wg_mapping: Dict[agent_name, wg_id]       │       │   │
│  │  │  wg_to_config: Dict[wg_id, AgentSpecificConfig]     │       │   │
│  │  └──────────────────────────────────────────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │           Layer 4: Async Trajectory Collection                   │   │
│  │                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────┐     │   │
│  │  │         AsyncMultiAgentTrajectoryCollector              │     │   │
│  │  │                                                        │     │   │
│  │  │  ┌──────────────┐    ┌──────────────┐                 │     │   │
│  │  │  │ RolloutPool  │    │ RolloutPool  │  ...            │     │   │
│  │  │  │ (WG-0 vLLM)  │    │ (WG-1 vLLM)  │                 │     │   │
│  │  │  └──────┬───────┘    └──────┬───────┘                 │     │   │
│  │  │         │                    │                         │     │   │
│  │  │    ┌────┴────────────────────┴────┐                   │     │   │
│  │  │    │   StreamingRolloutManager    │                   │     │   │
│  │  │    │                              │                   │     │   │
│  │  │    │  - 并发env rollout           │                   │     │   │
│  │  │    │  - 完成即yield（流式产出）    │                   │     │   │
│  │  │    │  - orchestra内部并行执行      │                   │     │   │
│  │  │    │  - Tree-search branching     │                   │     │   │
│  │  │    └──────────────┬───────────────┘                   │     │   │
│  │  │                   ↓                                   │     │   │
│  │  │          MessageQueue (Ray Actor)                      │     │   │
│  │  │          - staleness control                           │     │   │
│  │  │          - backpressure                                │     │   │
│  │  └────────────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │          Layer 5: Async Training Pipeline                        │   │
│  │                                                                  │   │
│  │       MessageQueue                                               │   │
│  │           │                                                      │   │
│  │           ↓                                                      │   │
│  │  ┌──────────────────────────────────────────────────────┐       │   │
│  │  │       AsyncMultiAgentTrainer                          │       │   │
│  │  │                                                      │       │   │
│  │  │  1. batch = queue.get()                              │       │   │
│  │  │  2. split_by_wg_ids()                                │       │   │
│  │  │  3. per-wg: compute_log_prob (并行多wg)              │       │   │
│  │  │  4. per-wg: compute_ref_log_prob (并行)              │       │   │
│  │  │  5. combine → compute_advantage:                     │       │   │
│  │  │     ├─ AT-GRPO uid=(env_id, turn_id, agent_id)      │       │   │
│  │  │     ├─ + group_by_agent_id for per-agent norm        │       │   │
│  │  │     ├─ + GiGPO step-level reward (optional)          │       │   │
│  │  │     └─ + Dynamic sampling filter                     │       │   │
│  │  │  6. split → per-wg update_actor (并行多wg)           │       │   │
│  │  │  7. ParameterSynchronizer → 异步sync weights to vLLM │       │   │
│  │  └──────────────────────────────────────────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │        Layer 6: Advantage & Reward Computation                   │   │
│  │                                                                  │   │
│  │  Reward:                                                         │   │
│  │  ├─ Per-agent reward (PettingLLMs style calculate_reward)       │   │
│  │  ├─ Episode reward (DrMAS style EpisodeRewardManager)           │   │
│  │  ├─ Invalid action penalty                                      │   │
│  │  └─ Reward shaping (hop_weighted, turn_discount, etc.)          │   │
│  │                                                                  │   │
│  │  Advantage Estimators:                                           │   │
│  │  ├─ AT-GRPO: uid=(env_id, turn_id, agent_id) + norm_by_std     │   │
│  │  ├─ Dr.GRPO: AT-GRPO without std normalization                  │   │
│  │  ├─ GiGPO: episode_adv + step_adv (anchor similarity)          │   │
│  │  ├─ RLOO: leave-one-out with agent grouping                     │   │
│  │  ├─ REINFORCE++: baseline with agent grouping                   │   │
│  │  └─ GAE: standard generalized advantage estimation              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 六、异步训推架构设计

### 当前两个系统的问题

PettingLLMs和DrMAS都采用**同步on-policy**训练：

```
[Rollout全部完成] ──barrier──> [Training全部完成] ──barrier──> [下一轮Rollout]
     GPU空闲(训练中)               GPU空闲(推理中)
```

Multi-agent场景下问题更严重：
1. **多agent顺序生成**：一个agent生成时其他agent的GPU idle
2. **环境交互延迟**：tool call（代码执行、搜索API）的IO等待
3. **agent间不平衡**：不同agent的response长度差异大，短的等长的
4. **多model场景**：不同model的GPU各自idle等待同步点

### 异步训推架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Async Multi-Agent Training Pipeline                   │
│                                                                          │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────┐  │
│  │     Rollout Side (Inference)      │  │    Training Side (Update)    │  │
│  │                                  │  │                              │  │
│  │  ┌───────────────────────────┐   │  │  ┌────────────────────────┐  │  │
│  │  │  EnvRolloutWorker-0      │   │  │  │  TrainerWorker (WG-A)  │  │  │
│  │  │  ┌─────────────────────┐ │   │  │  │                        │  │  │
│  │  │  │  Orchestra          │ │   │  │  │  while True:           │  │  │
│  │  │  │  Agent-A → vLLM-A   │ │   │  │  │    batch = queue.get() │  │  │
│  │  │  │  Agent-B → vLLM-B   │ │   │  │  │    split_by_wg()      │  │  │
│  │  │  │  env.step()         │ │   │  │  │    compute_advantage() │  │  │
│  │  │  └─────────────────────┘ │   │  │  │    update_actor()      │  │  │
│  │  │  完成 → push to queue    │   │  │  │    sync_weights()      │  │  │
│  │  └───────────────────────────┘   │  │  └────────────────────────┘  │  │
│  │                                  │  │                              │  │
│  │  ┌───────────────────────────┐   │  │  ┌────────────────────────┐  │  │
│  │  │  EnvRolloutWorker-1      │   │  │  │  TrainerWorker (WG-B)  │  │  │
│  │  │  (同上，独立env实例)       │   │  │  │  (同上，独立model)      │  │  │
│  │  └───────────────────────────┘   │  │  └────────────────────────┘  │  │
│  │                                  │  │                              │  │
│  │  ┌───────────────────────────┐   │  │                              │  │
│  │  │  EnvRolloutWorker-2 ...  │   │  │                              │  │
│  │  └───────────────────────────┘   │  │                              │  │
│  └──────────┬───────────────────────┘  └──────────┬───────────────────┘  │
│             │                                     │                      │
│             ▼                                     ▼                      │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    MessageQueue (Ray Actor)                       │    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │  Staleness Control:                                      │   │    │
│  │  │  - max_staleness: 允许的最大policy版本差                  │   │    │
│  │  │  - 超过则丢弃旧数据（multi-agent场景staleness更关键）     │   │    │
│  │  │                                                          │   │    │
│  │  │  Backpressure:                                           │   │    │
│  │  │  - max_queue_size: 防止rollout产出过快                    │   │    │
│  │  │  - per-wg queue: 不同model的数据分开排队                 │   │    │
│  │  │                                                          │   │    │
│  │  │  Batching Strategy:                                      │   │    │
│  │  │  - 等待凑够min_batch_size再dequeue                       │   │    │
│  │  │  - 或超时后dequeue当前已有数据                            │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                ParameterSynchronizer                              │    │
│  │                                                                  │    │
│  │  训练完成 → NCCL broadcast → vLLM workers                        │    │
│  │                                                                  │    │
│  │  Multi-Agent优化:                                                 │    │
│  │  ├─ L1 Prompt: 一次sync（所有agent共享）                          │    │
│  │  ├─ L2 LoRA: per-agent LoRA adapter增量sync                      │    │
│  │  │   (只传LoRA delta，远小于full model sync)                      │    │
│  │  └─ L3 Full: per-wg独立sync（不同model并行sync）                  │    │
│  │                                                                  │    │
│  │  异步sync策略:                                                    │    │
│  │  ├─ Eager: 每次update后立即sync（最新但开销大）                    │    │
│  │  ├─ Lazy: 累积N次update后sync（高效但staleness大）                │    │
│  │  └─ Adaptive: 根据queue depth动态调整sync频率                     │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Multi-Agent特有的异步优化

#### 1. Agent-Level Pipeline Parallelism

当前两个系统在orchestra内部都是顺序执行agent。可以引入pipeline并行：

```
时间 →
─────────────────────────────────────────────────────
Env-0:  [Agent-A gen] [Agent-B gen] [env.step]
Env-1:       [Agent-A gen] [Agent-B gen] [env.step]
Env-2:            [Agent-A gen] [Agent-B gen] [env.step]

优化后 (pipeline):
─────────────────────────────────────────────────────
Env-0:  [Agent-A gen] [Agent-B gen] [env.step]
Env-1:  [Agent-A gen] [Agent-B gen] [env.step]    ← A和Env-0的B并行
Env-2:       [Agent-A gen] [Agent-B gen] [env.step]
```

对于不同model的agent（L3模式），Agent-A和Agent-B可以完全并行（用不同GPU）。
对于相同model的agent（L1/L2），利用vLLM的continuous batching，不同env的同一agent请求可以batch。

#### 2. Rollout-Training Overlap (One-Step-Off)

借鉴verl的one_step_off_policy模式，但适配multi-agent：

```python
class AsyncMultiAgentTrainer:
    def fit(self):
        # 启动第一轮rollout
        future_batch = self.rollout_async(gen_batch_0)

        for step in range(total_steps):
            # 获取当前batch（上一轮rollout结果）
            batch = future_batch.get()

            # 立即启动下一轮rollout（用当前policy，稍有staleness）
            next_gen_batch = self.dataloader.next()
            future_batch = self.rollout_async(next_gen_batch)

            # 在下一轮rollout进行的同时，执行训练
            self.train_step(batch)  # compute_advantage + update_actor

            # 训练完成后sync weights（异步）
            self.param_sync.async_broadcast()
```

Multi-agent场景收益更大，因为：
- Rollout阶段更长（多turn多agent交互 + tool call IO等待）
- 训练阶段相对较短（标准PPO update）
- Overlap的时间窗口更充裕

#### 3. Per-Agent Async Training

不同agent可以独立异步训练，不需要等待所有agent数据凑齐：

```
时间 →
─────────────────────────────────────────────────────
同步模式:
[所有agent rollout完成] → [Agent-A train] [Agent-B train] → [sync]

异步模式:
Agent-A: [rollout] → [train] → [sync-A] → [rollout] → ...
Agent-B:     [rollout] → [train] → [sync-B] → [rollout] → ...

关键约束: advantage computation需要统一的reward信号
解决方案:
  - 使用per-wg queue，每个wg独立consume
  - advantage在combine后统一计算（短暂barrier）
  - 或使用RLOO/REINFORCE++等不需要跨agent分组的estimator
```

#### 4. Streaming Rollout for Multi-Turn

多轮交互中，每完成一个env的整个episode就立即push到queue，不等所有env完成：

```python
class StreamingMultiAgentRollouter:
    async def generate_rollouts(self, envs):
        tasks = [self.run_episode(env_idx) for env_idx in range(num_envs)]

        for completed in asyncio.as_completed(tasks):
            trajectory = await completed
            # 单个episode完成即push，不等待所有env
            self.message_queue.push(trajectory)
```

这对multi-agent尤其重要，因为不同env的episode长度差异可能很大（有的env第一轮就done，有的要跑满max_turns）。

---

## 七、从两个系统取长补短的具体建议

### 从DrMAS吸收

1. **Orchestra模式** → 替代PettingLLMs的固定turn_order，支持条件分支和迭代
2. **AgentRegistry** → 统一agent注册和创建机制
3. **wg_id split/combine模式** → 统一处理多model场景（替代PettingLLMs的ppo_trainer_dict）
4. **group_by_agent_id开关** → 灵活控制advantage分组粒度
5. **GiGPO step-level reward** → 更细粒度的credit assignment
6. **Dynamic sampling** → DAPO-style过滤同质reward样本
7. **agent_specific_parameters** → per-agent学习率、batch size等超参
8. **active_mask** → 条件执行，不需要所有agent每步都运行
9. **Memory模块** → 独立的记忆管理，支持不同任务的记忆格式

### 从PettingLLMs吸收

1. **有状态Agent** → agent维护内部状态（reward、success、done），比DrMAS的无状态call更适合复杂交互
2. **多级Specialization** → L0/L1/L2/L3的清晰分级，特别是L2 LoRA模式
3. **Tree-Structured Sampling** → Best-of-N branching在每个agent step
4. **Per-agent reward** → agent.calculate_reward() 支持每个agent独立reward函数
5. **AT-GRPO uid构造** → (env_id, turn_id, agent_id) 比 f"{uid}_{agent_id}" 更精确
6. **vLLM wake_up/sleep** → 推理引擎生命周期管理，释放显存给训练
7. **Graph/AutoEvol Engine** → 支持AutoGen workflow和自动搜索MAS结构
8. **多种reward shaping** → hop_weighted, turn_discount, binary, normalized等

### 不建议采用的设计

| 来源 | 不建议 | 原因 |
|------|--------|------|
| PettingLLMs | 外部封装verl的方式 | 深度修改更灵活，但维护成本高；建议用DrMAS的fork+patch方式 |
| PettingLLMs | 所有agent-turn flatten为一个DataProto | 丢失了trajectory结构信息；DrMAS的traj_uid保留更好 |
| DrMAS | 无LoRA支持 | 应补充L2 LoRA specialization模式 |
| DrMAS | Orchestra内硬编码max_loop_num | 应支持动态终止条件（如PettingLLMs的env.done） |
| 两者 | 同步on-policy训练 | 必须引入异步训推 |

---

## 八、性能预期

基于verl fully-async的benchmark数据和multi-agent特性分析：

| 优化项 | 单Agent加速比 (verl数据) | Multi-Agent预期加速比 | 原因 |
|--------|-------------------------|----------------------|------|
| Rollout-Training Overlap | 1.23-1.40x | **1.4-1.8x** | Multi-agent rollout更长，overlap窗口更大 |
| Streaming Rollout | 1.15-1.20x | **1.3-1.5x** | Multi-agent episode长度方差大，流式收益更高 |
| Agent Pipeline Parallel | N/A | **1.2-1.4x** | 多model场景下不同agent GPU并行 |
| LoRA Delta Sync | N/A | **1.1-1.2x** | 只sync LoRA权重，远小于full model |
| Dynamic Sampling | 1.05-1.10x | **1.1-1.2x** | 过滤同质reward数据，减少无效训练 |
| **综合** | **2.35-2.67x** | **预计3-4x** | Multi-agent场景异步收益叠加 |

注：以上数据为理论估计，实际效果取决于agent数量、turn数量、model大小、GPU配置等因素。
