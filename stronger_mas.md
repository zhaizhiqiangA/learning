# Multi-Agent RL Framework Architecture Design

## Part I: Current PettingLLMs Architecture Analysis

### 1.1 Overall System Architecture (As-Is)

The current PettingLLMs framework follows a **synchronous on-policy training loop** where rollout generation and model training are strictly sequential, with a global synchronization barrier between them.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MultiAgentsPPOTrainer (Driver Process)              │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Main Training Loop (fit())                   │  │
│  │                                                                   │  │
│  │   ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │  │
│  │   │  Phase 1:    │    │  Phase 2:    │    │  Phase 3:          │  │  │
│  │   │  ROLLOUT     │───►│  PREPARE     │───►│  TRAIN             │  │  │
│  │   │  GENERATION  │    │  TRAINING    │    │  (Update Params)   │  │  │
│  │   │  (all envs)  │    │  DATA        │    │                    │  │  │
│  │   └──────┬───────┘    └──────────────┘    └────────┬───────────┘  │  │
│  │          │                                         │              │  │
│  │          │◄────────────────────────────────────────┘              │  │
│  │          │         SYNCHRONIZATION BARRIER                        │  │
│  │          │         (must complete all phases                      │  │
│  │          │          before next rollout)                          │  │
│  └──────────┴────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────┐    ┌──────────────────────────────────────┐  │
│  │ ppo_trainer_dict      │    │ agent_execution_engine               │  │
│  │ ┌──────────────────┐ │    │ (MultiAgentsExecutionEngine)         │  │
│  │ │ Model_0: Trainer  │ │    │                                      │  │
│  │ │  ├─ actor_wg      │ │    │  ┌────────────────────────────────┐ │  │
│  │ │  ├─ critic_wg     │ │    │  │ rollout_engine_dict            │ │  │
│  │ │  ├─ ref_wg        │ │    │  │ ┌────────────────────────────┐│ │  │
│  │ │  └─ rollout_mgr   │ │    │  │ │ Model_0: AsyncLLMServer   ││ │  │
│  │ └──────────────────┘ │    │  │ ├────────────────────────────┤│ │  │
│  │ ┌──────────────────┐ │    │  │ │ Model_1: AsyncLLMServer   ││ │  │
│  │ │ Model_1: Trainer  │ │    │  │ └────────────────────────────┘│ │  │
│  │ │  ├─ actor_wg      │ │    │  └────────────────────────────────┘ │  │
│  │ │  ├─ critic_wg     │ │    │                                      │  │
│  │ │  └─ rollout_mgr   │ │    │  ┌────────────────────────────────┐ │  │
│  │ └──────────────────┘ │    │  │ env_workers (Ray Actor Pool)   │ │  │
│  └──────────────────────┘    │  └────────────────────────────────┘ │  │
│                               └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Rollout Generation Phase (Detailed)

The execution engine supports three MAS design paradigms (A: Graph, B: Turn-based FSM, C: AutoEvol), but shares the same synchronization model:

```
generate_multiple_rollouts_concurrent()
│
├─── Mode: "tree" (training)                    Mode: "flat" (validation)
│    generate_env_idx_rollout()                 generate_single_rollout()
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│  For each env_idx (concurrent via asyncio):                            │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ For each turn (0 → max_turns):                                   │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │ For each agent in turn_order:                                │ │  │
│  │  │                                                              │ │  │
│  │  │  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐ │ │  │
│  │  │  │ 1. agent.    │     │ 2. llm_async │    │ 3. agent.    │ │ │  │
│  │  │  │ update_from  │────►│ _generate()  │───►│ update_from  │ │ │  │
│  │  │  │ _env()       │     │ (vLLM API)   │    │ _model()     │ │ │  │
│  │  │  └──────────────┘     └──────────────┘    └──────┬───────┘ │ │  │
│  │  │                                                   │         │ │  │
│  │  │                        ┌──────────────┐    ┌──────▼───────┐ │ │  │
│  │  │                        │ 5. agent.    │◄───│ 4. agent.    │ │ │  │
│  │  │                        │ calculate    │    │ step(env,    │ │ │  │
│  │  │                        │ _reward()    │    │  env_worker) │ │ │  │
│  │  │                        └──────────────┘    └──────────────┘ │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  │                                                                   │  │
│  │  [Tree mode only] Select best rollout, deepcopy state to all     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ══════════════ asyncio barrier: all envs must complete ═══════════════ │
│                                                                        │
│  Return: trajectory_per_task_dict = {policy_name: DataProto}           │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Three-Level Specialization & Training Phase

```
                        ┌─────────────────────────────────────────┐
                        │        Agent Specialization Modes        │
                        └─────────────────────────────────────────┘

  L1: Prompt-based                L2: LoRA-based               L3: Full Model
  ┌──────────────────┐   ┌──────────────────────────┐   ┌────────────────────┐
  │   Base Model     │   │      Base Model           │   │  Model_1   Model_2 │
  │  ┌────┐ ┌────┐  │   │  ┌──────┐  ┌──────┐     │   │  ┌────┐   ┌────┐  │
  │  │Ag1 │ │Ag2 │  │   │  │LoRA_1│  │LoRA_2│     │   │  │Ag1 │   │Ag2 │  │
  │  │prmt│ │prmt│  │   │  │Ag1   │  │Ag2   │     │   │  │full│   │full│  │
  │  └────┘ └────┘  │   │  └──────┘  └──────┘     │   │  └────┘   └────┘  │
  │                  │   │                          │   │                    │
  │ 1 PPO Trainer    │   │ 1 PPO Trainer            │   │ N PPO Trainers    │
  │ 1 ResourcePool   │   │ 1 ResourcePool           │   │ N ResourcePools   │
  │ Shared trajectory│   │ Per-agent trajectory      │   │ Per-agent traj.   │
  └──────────────────┘   └──────────────────────────┘   └────────────────────┘


Training Data Flow (for L2 LoRA mode):
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  trajectory_per_task_dict["shared_model"]                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ DataProto:                                                         │  │
│  │  batch["input_ids"], batch["responses"], batch["attention_mask"]   │  │
│  │  non_tensor["agent_name"], non_tensor["lora_ids"]                 │  │
│  └────────────┬───────────────────────────────────────────────────────┘  │
│               │                                                          │
│               ▼                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ _update_parameters():                                              │  │
│  │                                                                    │  │
│  │  1. Pad prompts/responses                                         │  │
│  │  2. actor_wg.compute_log_prob(full_batch)         [FSDP parallel] │  │
│  │  3. ref_wg.compute_ref_log_prob(full_batch)       [FSDP parallel] │  │
│  │  4. critic_wg.compute_values(full_batch)          [FSDP parallel] │  │
│  │  5. compute_advantage(full_batch)                 [CPU driver]    │  │
│  │  6. critic_wg.update_critic(full_batch)           [FSDP parallel] │  │
│  │  7. Split batch by agent_name:                                    │  │
│  │     ┌─────────────────────────────────────────────────────────┐   │  │
│  │     │ for agent_name in [reasoning_agent, tool_agent]:        │   │  │
│  │     │   agent_batch = filter(batch, agent_name)  ◄─SEQUENTIAL │   │  │
│  │     │   actor_wg.update_actor(agent_batch)                    │   │  │
│  │     │   # Each LoRA adapter trained one at a time             │   │  │
│  │     └─────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.4 AT-GRPO Algorithm Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AT-GRPO: Agent- & Turn-wise GRPO                     │
│                                                                         │
│  Standard GRPO:          AT-GRPO:                                       │
│  Group = Env_id          Group = (Env_id, Agent_id, Turn_id)            │
│                                                                         │
│  ┌───────────────────┐   ┌─────────────────────────────────────────┐   │
│  │ Env_0:            │   │ Env_0:                                   │   │
│  │  rollout_1 ─► R_1 │   │  Agent_1, Turn_0:                       │   │
│  │  rollout_2 ─► R_2 │   │    rollout_1 ─► r_1(S_0)               │   │
│  │  rollout_3 ─► R_3 │   │    rollout_2 ─► r_1(S_0')              │   │
│  │                    │   │    rollout_3 ─► r_1(S_0'')             │   │
│  │  adv_i = R_i -    │   │    adv = r_i - mean(r) within group    │   │
│  │    mean(R_1..R_3)  │   │                                         │   │
│  └───────────────────┘   │  Agent_2, Turn_0:                       │   │
│                           │    rollout_1 ─► r_2(S_0)   ◄ separate  │   │
│                           │    rollout_2 ─► r_2(S_0')    group     │   │
│                           │    rollout_3 ─► r_2(S_0'')             │   │
│                           └─────────────────────────────────────────┘   │
│                                                                         │
│  Tree-Structured Sampling (Best-of-N at each agent step):              │
│                                                                         │
│  Turn 0:                  Turn 1:                   Turn 2:             │
│  ┌─────────────────┐     ┌─────────────────┐      ┌────────────────┐  │
│  │ Agent_1 acts:   │     │ Agent_1 acts:   │      │ Agent_1 acts:  │  │
│  │ N rollouts      │     │ N rollouts      │      │ N rollouts     │  │
│  │ ├─ r1 ──►★best │     │ ├─ r1 ──►★best │      │ ...            │  │
│  │ ├─ r2          │     │ ├─ r2          │      │                │  │
│  │ └─ r3          │     │ └─ r3          │      │                │  │
│  │ deepcopy best   │     │ deepcopy best   │      │                │  │
│  │       │         │     │       │         │      │                │  │
│  │       ▼         │     │       ▼         │      │                │  │
│  │ Agent_2 acts:   │     │ Agent_2 acts:   │      │                │  │
│  │ N rollouts      │     │ N rollouts      │      │                │  │
│  │ ├─ r1          │     │ ├─ r1 ──►★best │      │                │  │
│  │ ├─ r2 ──►★best │     │ ├─ r2          │      │                │  │
│  │ └─ r3          │     │ └─ r3          │      │                │  │
│  │ deepcopy best   │     │ deepcopy best   │      │                │  │
│  └────────┬────────┘     └────────┬────────┘      └────────────────┘  │
│           └───────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Current System Timeline (Synchronous Bottlenecks)

```
Time ──────────────────────────────────────────────────────────────────►

Step N:
              ┌──────────────────────────────┐
  Model_0 GPU │████ wake_up ████████████████ │ sleep │
              │  vLLM server   ROLLOUT GEN   │       │
              └──────────────────────────────┘       │
              ┌──────────────────────────────┐       │
  Model_1 GPU │████ wake_up ████████████████ │ sleep │
              │  vLLM server   ROLLOUT GEN   │       │
              └──────────────────────────────┘       │
                                                     │
              ═══ GLOBAL BARRIER: all rollouts done ═══
                                                     │
              ┌──────────────────────────────────────┤
  Model_0 GPU │ log_prob │ ref │ critic │ UPDATE_0   │
              └──────────────────────────────────────┤
              ┌──────────────────────────────────────┤
  Model_1 GPU │ log_prob │ ref │ critic │ UPDATE_1   │ ◄── SEQUENTIAL
              └──────────────────────────────────────┘
                                                     │
              ═══ TRAINING DONE, NEXT STEP ════════════

  IDLE periods:                                    Waste
  ┌─────────────────────────────────────────────────────────────────┐
  │ Model_0 GPUs idle during Model_1 training & vice versa         │
  │ All GPUs idle during wake_up/sleep transitions                  │
  │ Fast rollouts wait for slow rollouts (long-tail problem)        │
  │ LoRA adapters trained sequentially, not in parallel             │
  └─────────────────────────────────────────────────────────────────┘
```

### 1.6 Key Bottleneck Summary

| # | Bottleneck | Location | Impact |
|---|-----------|----------|--------|
| B1 | Sequential `wake_up`/`sleep` across models | `fit()` lines 610-649 | O(N_models) startup time |
| B2 | Global rollout barrier | `asyncio.run(generate_multiple_rollouts_concurrent())` | Slowest rollout blocks all |
| B3 | Training-Rollout alternation | Strict Phase 1→2→3 | 0% overlap |
| B4 | Sequential LoRA training | `_update_parameters()` lines 478-481 | O(N_agents) training time |
| B5 | Sequential model training (L3) | `fit()` lines 682-712 | Models wait for each other |
| B6 | Driver CPU bottleneck | Advantage computation, data routing | Single-threaded |

---

## Part II: verl Async Capabilities Analysis

### 2.1 verl Async Architecture Options

```
┌─────────────────────────────────────────────────────────────────────┐
│                   verl Async Training Modes                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Mode A: One-Step-Off (23-40% speedup)                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Step N:   ┌─ Gen(N) ─────────┐                               │ │
│  │            └───────────────────┤                               │ │
│  │                                ├─► Train(N)                    │ │
│  │  Step N+1:           ┌─ Gen(N+1) ──────┐                     │ │
│  │                      └──────────────────┤                     │ │
│  │                                         ├─► Train(N+1)        │ │
│  │  Key: Future-based, 1-step lookahead, minimal code change     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Mode B: Fully-Async (2.35-2.67x speedup at 128 GPUs)              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Rollouter: ──Gen──Gen──Gen──Gen──Gen──Gen──Gen──►            │ │
│  │                │    │    ▼    │    │    ▼                      │ │
│  │                │    └──►MQ───►│    └──►MQ──►                  │ │
│  │                │              │                                │ │
│  │  Trainer:      └──Train──────►└──Train──────►                 │ │
│  │                       ▲ ParamSync       ▲ ParamSync           │ │
│  │                                                               │ │
│  │  Components:                                                   │ │
│  │   ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │ │
│  │   │ MessageQueue │ │ ParamSync    │ │ Checkpoint Engine    │ │ │
│  │   │ (Ray Actor)  │ │ (NCCL Group) │ │ (Bucketed transfer)  │ │ │
│  │   │ Staleness    │ │ Pause/Resume │ │ 60% faster sync      │ │ │
│  │   │ control      │ │ per model    │ │                      │ │ │
│  │   └──────────────┘ └──────────────┘ └──────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key verl Abstractions Reusable for Multi-Agent

| Abstraction | Role | Multi-Model Extension |
|------------|------|----------------------|
| `MessageQueue` | Decouples rollout from training | One queue per model |
| `ParameterSynchronizer` | NCCL weight broadcast | One sync group per model |
| `RolloutSample` | Carries param_version metadata | Add model_id tracking |
| `CheckpointEngine` | Bucketed fast weight transfer | Per-model engine instance |
| Pause/Resume protocol | Interrupts rollout for sync | Per-model pause/resume |

---

## Part III: Optimized Async Multi-Agent Multi-Model Architecture

### 3.1 Design Principles

1. **Per-Model Async Loop**: Each model has its own independent rollout-train cycle
2. **Decoupled via MessageQueues**: Models don't block each other during training
3. **Multi-Agent Coordinator**: A central orchestrator coordinates cross-model agent interactions during rollouts
4. **Independent Parameter Sync**: Each model syncs weights on its own schedule
5. **Flexible Topology**: Supports L1/L2/L3 specialization with the same architecture

### 3.2 Optimized System Architecture (To-Be)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Async Multi-Agent Multi-Model RL System                     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Multi-Agent Coordinator (Ray Actor)                 │  │
│  │                                                                       │  │
│  │  Responsibilities:                                                    │  │
│  │  - Orchestrate multi-agent rollout episodes                          │  │
│  │  - Route LLM generation requests to correct model's Rollouter       │  │
│  │  - Manage environment state across agents                            │  │
│  │  - Dispatch completed trajectories to per-model MessageQueues        │  │
│  │  - Support Graph / FSM / AutoEvol MAS paradigms                     │  │
│  │                                                                       │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────────┐ │  │
│  │  │ Env Pool  │  │ Agent     │  │ Turn Order│  │ Reward Fn Pool   │ │  │
│  │  │ (Ray)     │  │ Registry  │  │ / Graph   │  │ (per-domain)     │ │  │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────────┬─────────┘ │  │
│  │        └───────────────┴──────────────┴─────────────────┘           │  │
│  └──────────────────────────────┬────────────────────────────────────────┘  │
│                                 │                                           │
│         ┌───────────────────────┼──────────────────────────┐               │
│         │                       │                          │               │
│         ▼                       ▼                          ▼               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐        │
│  │  Model_0 Async   │  │  Model_1 Async   │  │  Model_N Async   │        │
│  │  Pipeline        │  │  Pipeline        │  │  Pipeline        │        │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │        │
│  │  │ Rollouter  │  │  │  │ Rollouter  │  │  │  │ Rollouter  │  │        │
│  │  │ (Streaming)│  │  │  │ (Streaming)│  │  │  │ (Streaming)│  │        │
│  │  └─────┬──────┘  │  │  └─────┬──────┘  │  │  └─────┬──────┘  │        │
│  │        │         │  │        │         │  │        │         │        │
│  │        ▼         │  │        ▼         │  │        ▼         │        │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │        │
│  │  │ MsgQueue_0 │  │  │  │ MsgQueue_1 │  │  │  │ MsgQueue_N │  │        │
│  │  │ (Ray)      │  │  │  │ (Ray)      │  │  │  │ (Ray)      │  │        │
│  │  └─────┬──────┘  │  │  └─────┬──────┘  │  │  └─────┬──────┘  │        │
│  │        │         │  │        │         │  │        │         │        │
│  │        ▼         │  │        ▼         │  │        ▼         │        │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │        │
│  │  │ Trainer_0  │  │  │  │ Trainer_1  │  │  │  │ Trainer_N  │  │        │
│  │  │ (FSDP/Meg) │  │  │  │ (FSDP/Meg) │  │  │  │ (FSDP/Meg) │  │        │
│  │  └─────┬──────┘  │  │  └─────┬──────┘  │  │  └─────┬──────┘  │        │
│  │        │         │  │        │         │  │        │         │        │
│  │        ▼         │  │        ▼         │  │        ▼         │        │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │        │
│  │  │ ParamSync_0│  │  │  │ ParamSync_1│  │  │  │ ParamSync_N│  │        │
│  │  │ (NCCL grp) │  │  │  │ (NCCL grp) │  │  │  │ (NCCL grp) │  │        │
│  │  └────────────┘  │  │  └────────────┘  │  │  └────────────┘  │        │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   GPU Resource Pools                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐        ┌──────────────┐        │  │
│  │  │ Pool_0       │  │ Pool_1       │  ...   │ Pool_N       │        │  │
│  │  │ GPU [0..k]   │  │ GPU [k..m]   │        │ GPU [m..p]   │        │  │
│  │  │ Actor+Rollout│  │ Actor+Rollout│        │ Actor+Rollout│        │  │
│  │  └──────────────┘  └──────────────┘        └──────────────┘        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Multi-Agent Coordinator Detail

The Coordinator is the critical new component. It **decouples multi-agent episode logic from per-model async pipelines**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent Coordinator                              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                  Episode Generation Loop                           │  │
│  │                                                                    │  │
│  │  For each episode (env_idx, rollout_idx):                         │  │
│  │                                                                    │  │
│  │   ┌──────────────────────────────────────────────────────────┐    │  │
│  │   │ Turn 0:                                                   │    │  │
│  │   │                                                           │    │  │
│  │   │  Agent_1 (→ Model_0):                                    │    │  │
│  │   │    1. update_from_env(env_state)                          │    │  │
│  │   │    2. ──► InferenceRouter.request(model_0, prompt) ──►   │    │  │
│  │   │           ┌──────────────────────────────────────┐       │    │  │
│  │   │           │ Model_0 Rollouter (async, streaming) │       │    │  │
│  │   │           │ Returns: Future<response>            │       │    │  │
│  │   │           └──────────────────────────────────────┘       │    │  │
│  │   │    3. ◄── await response                                 │    │  │
│  │   │    4. update_from_model(response)                        │    │  │
│  │   │    5. step(env, env_worker) → reward                     │    │  │
│  │   │                                                           │    │  │
│  │   │  Agent_2 (→ Model_1):                                    │    │  │
│  │   │    1. update_from_env(env_state)  ◄ sees Agent_1 output  │    │  │
│  │   │    2. ──► InferenceRouter.request(model_1, prompt) ──►   │    │  │
│  │   │           ┌──────────────────────────────────────┐       │    │  │
│  │   │           │ Model_1 Rollouter (async, streaming) │       │    │  │
│  │   │           │ May use DIFFERENT GPU pool            │       │    │  │
│  │   │           └──────────────────────────────────────┘       │    │  │
│  │   │    3. ◄── await response                                 │    │  │
│  │   │    4-5. update + step + reward                           │    │  │
│  │   └──────────────────────────────────────────────────────────┘    │  │
│  │                                                                    │  │
│  │   After episode completes:                                        │  │
│  │   ┌──────────────────────────────────────────────────────────┐    │  │
│  │   │ TrajectoryRouter:                                         │    │  │
│  │   │   Split trajectory by agent → model mapping               │    │  │
│  │   │   Agent_1 trajectory ──► MsgQueue_0.put(sample, ver_0)   │    │  │
│  │   │   Agent_2 trajectory ──► MsgQueue_1.put(sample, ver_1)   │    │  │
│  │   └──────────────────────────────────────────────────────────┘    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                  InferenceRouter                                   │  │
│  │                                                                    │  │
│  │  Maps (agent_name, policy_name) → Rollouter endpoint              │  │
│  │                                                                    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │ Supports:                                                     │ │  │
│  │  │  - L1: All agents → same Rollouter (different prompts)       │ │  │
│  │  │  - L2: All agents → same Rollouter (different LoRA IDs)      │ │  │
│  │  │  - L3: Each agent → different Rollouter (different models)   │ │  │
│  │  │  - Mixed: Some agents share, some independent                │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                  VersionTracker                                    │  │
│  │                                                                    │  │
│  │  param_versions = {model_0: 5, model_1: 3, model_2: 7}           │  │
│  │                                                                    │  │
│  │  On each trajectory sample:                                       │  │
│  │    sample.param_version_per_model = snapshot(param_versions)      │  │
│  │    → Enables per-model staleness checking in MessageQueue         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Per-Model Async Pipeline Detail

Each model runs an independent fully-async training loop:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   Model_k Async Pipeline                                 │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Rollouter_k (Ray Actor, persistent)                            │    │
│  │                                                                  │    │
│  │  Serves inference requests from Coordinator:                     │    │
│  │  ┌────────────────────────────────────────────────────────────┐ │    │
│  │  │ async def generate(prompt, lora_id=None):                  │ │    │
│  │  │   """Called by Coordinator's InferenceRouter"""             │ │    │
│  │  │   return await vllm_server.completions(prompt, lora_id)    │ │    │
│  │  └────────────────────────────────────────────────────────────┘ │    │
│  │                                                                  │    │
│  │  State: paused / running                                        │    │
│  │  On pause():  stop accepting new requests, drain in-flight      │    │
│  │  On resume(): accept new requests with updated weights          │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                              │ (Coordinator dispatches completed        │
│                              │  episodes to appropriate MQ)             │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  MessageQueue_k (Ray Actor)                                     │    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │ Staleness control:                                        │   │    │
│  │  │   sample.param_version_per_model[k] vs current_version   │   │    │
│  │  │   Drop if too stale (configurable threshold)             │   │    │
│  │  │                                                           │   │    │
│  │  │ Backpressure:                                             │   │    │
│  │  │   If queue full → Coordinator slows episode generation   │   │    │
│  │  │   (per-model independent backpressure)                   │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Trainer_k (Ray Actor)                                          │    │
│  │                                                                  │    │
│  │  Independent training loop:                                      │    │
│  │  ┌────────────────────────────────────────────────────────────┐ │    │
│  │  │ while not done:                                             │ │    │
│  │  │   batch = MsgQueue_k.get_samples(required=N)               │ │    │
│  │  │   log_prob = actor_wg_k.compute_log_prob(batch)            │ │    │
│  │  │   ref_prob = ref_wg_k.compute_ref_log_prob(batch)          │ │    │
│  │  │   values  = critic_wg_k.compute_values(batch)              │ │    │
│  │  │   adv     = compute_advantage(batch)  # AT-GRPO grouping  │ │    │
│  │  │   actor_wg_k.update_actor(batch)                           │ │    │
│  │  │   critic_wg_k.update_critic(batch)                         │ │    │
│  │  │                                                             │ │    │
│  │  │   if steps % trigger_sync == 0:                            │ │    │
│  │  │     ParamSync_k.sync_weights(version++)                    │ │    │
│  │  │     # Rollouter_k pauses, receives new weights, resumes   │ │    │
│  │  └────────────────────────────────────────────────────────────┘ │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  ParameterSynchronizer_k                                        │    │
│  │                                                                  │    │
│  │  NCCL Group: actor_wg_k.workers + rollouter_k.workers          │    │
│  │                                                                  │    │
│  │  sync_weights(version):                                         │    │
│  │    1. Rollouter_k.pause()                                       │    │
│  │    2. NCCL broadcast: Actor_k → Rollouter_k weights            │    │
│  │    3. MsgQueue_k.update_param_version(version)                  │    │
│  │    4. Rollouter_k.resume()  (async, non-blocking)              │    │
│  │                                                                  │    │
│  │  Independent of other models' sync schedules                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Optimized Timeline (Async, Overlapping)

```
Time ──────────────────────────────────────────────────────────────────────►

Model_0 Rollouter:
  ┌──Gen──Gen──Gen──┤pause├──Gen──Gen──Gen──┤pause├──Gen──Gen──►
  │                  │sync │                 │sync │
  │                  │ v1  │                 │ v2  │

Model_0 Trainer:
              ┌──Train(batch_0)──┤──Train(batch_1)──┤──Train(batch_2)──►
              │                   │                   │

Model_1 Rollouter:
  ┌──Gen──Gen──Gen──Gen──┤pause├──Gen──Gen──Gen──Gen──┤pause├──►
  │                       │sync │                      │sync │
  │                       │ v1  │                      │ v2  │

Model_1 Trainer:
                    ┌──Train(batch_0)──┤──Train(batch_1)──┤──►
                    │                   │                   │

Model_2 Rollouter:
  ┌──Gen──┤pause├──Gen──Gen──Gen──┤pause├──Gen──►
  │        │sync │                 │sync │
  │        │ v1  │                 │ v2  │

Model_2 Trainer:
      ┌──Train(batch_0)──┤──Train(batch_1)──┤──►


  KEY IMPROVEMENTS:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ ✓ Models train at independent rates (Model_2 small → trains fast)  │
  │ ✓ Rollout & training overlap continuously for each model           │
  │ ✓ No global barrier across models                                   │
  │ ✓ Parameter sync is per-model, non-blocking for other models       │
  │ ✓ Coordinator streams episodes → MQs without waiting for training  │
  │ ✓ Fast models don't wait for slow models                           │
  └──────────────────────────────────────────────────────────────────────┘
```

### 3.6 Handling Cross-Model Dependencies in Multi-Agent Episodes

The key challenge: in a multi-agent episode, Agent_1 (Model_0) must complete a turn **before** Agent_2 (Model_1) can act, because Agent_2's prompt depends on Agent_1's output. This creates an **inherent sequential dependency within a single episode**, even though training is async.

```
┌──────────────────────────────────────────────────────────────────────────┐
│           Cross-Model Episode Execution Strategy                         │
│                                                                          │
│  Problem:                                                                │
│    Agent_1 (Model_0) → output → Agent_2 (Model_1) → output → ...       │
│    Sequential within episode, but models train independently             │
│                                                                          │
│  Solution: Decoupled Inference & Training                               │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  INFERENCE PLANE (always-on, shared by Coordinator):              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │  │
│  │  │ Rollouter_0  │  │ Rollouter_1  │  │ Rollouter_2  │           │  │
│  │  │ (vLLM srv)   │  │ (vLLM srv)   │  │ (vLLM srv)   │           │  │
│  │  │ always alive  │  │ always alive  │  │ always alive  │           │  │
│  │  │ serving reqs  │  │ serving reqs  │  │ serving reqs  │           │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │  │
│  │         │  ▲               │  ▲               │  ▲               │  │
│  │         │  │ weight sync   │  │ weight sync   │  │ weight sync   │  │
│  │         │  │ (periodic)    │  │ (periodic)    │  │ (periodic)    │  │
│  │         ▼  │               ▼  │               ▼  │               │  │
│  │                                                                    │  │
│  │  TRAINING PLANE (independent per-model):                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │  │
│  │  │ Trainer_0    │  │ Trainer_1    │  │ Trainer_2    │           │  │
│  │  │ consuming    │  │ consuming    │  │ consuming    │           │  │
│  │  │ from MQ_0    │  │ from MQ_1    │  │ from MQ_2    │           │  │
│  │  │ at own pace  │  │ at own pace  │  │ at own pace  │           │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │  │
│  │                                                                    │  │
│  │  Note: Rollouters serve inference during training.               │  │
│  │  Unlike current design, NO wake_up/sleep cycle.                  │  │
│  │  vLLM servers are persistent, weight sync is in-place.           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Episode execution (Coordinator):                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  # Multiple episodes run concurrently                              │  │
│  │  async def run_episode(env_idx):                                  │  │
│  │    for turn in range(max_turns):                                  │  │
│  │      for agent in turn_order:                                     │  │
│  │        model_id = agent_to_model[agent.name]                     │  │
│  │        # Non-blocking inference request to persistent server      │  │
│  │        response = await Rollouter[model_id].generate(prompt)     │  │
│  │        agent.step(env, response)                                  │  │
│  │    # Episode done, split and dispatch trajectories                │  │
│  │    for model_id, traj in split_by_model(trajectories):           │  │
│  │      await MsgQueue[model_id].put(traj, param_ver[model_id])    │  │
│  │                                                                    │  │
│  │  # Run many episodes concurrently:                                │  │
│  │  await asyncio.gather(*[run_episode(i) for i in env_indices])    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.7 Unified Specialization Support

```
┌──────────────────────────────────────────────────────────────────────────┐
│         Unified Architecture Across Specialization Levels                │
│                                                                          │
│  L1 (Prompt):           L2 (LoRA):             L3 (Full Model):         │
│  ┌───────────────┐     ┌───────────────┐      ┌───────────────┐        │
│  │ 1 Rollouter   │     │ 1 Rollouter   │      │ N Rollouters  │        │
│  │ 1 MsgQueue    │     │ 1 MsgQueue    │      │ N MsgQueues   │        │
│  │ 1 Trainer     │     │ 1 Trainer     │      │ N Trainers    │        │
│  │ 1 ParamSync   │     │ 1 ParamSync   │      │ N ParamSyncs  │        │
│  │               │     │               │      │               │        │
│  │ Agent routing │     │ Agent routing │      │ Agent routing │        │
│  │ by prompt     │     │ by lora_id    │      │ by model_id   │        │
│  └───────────────┘     └───────────────┘      └───────────────┘        │
│                                                                          │
│  L2+ (LoRA Multi-Trainer):                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Optimization: Parallel LoRA Training within single Trainer     │   │
│  │                                                                  │   │
│  │  Current: sequential per-agent update_actor()                   │   │
│  │  Optimized:                                                      │   │
│  │    ┌────────────────────────────────────────────────────────┐   │   │
│  │    │ batch_all_agents = concat_with_lora_routing(            │   │   │
│  │    │   agent_1_batch (lora_id=1),                           │   │   │
│  │    │   agent_2_batch (lora_id=2),                           │   │   │
│  │    │   agent_3_batch (lora_id=3),                           │   │   │
│  │    │ )                                                       │   │   │
│  │    │ actor_wg.update_actor(batch_all_agents)  ◄─ SINGLE call │   │   │
│  │    │ # vLLM/FSDP handles LoRA routing internally             │   │   │
│  │    └────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Mixed (L2+L3):                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Group_A: Model_0 + LoRA adapters (Agent_1, Agent_2)           │   │
│  │    → 1 Rollouter, 1 MsgQueue, 1 Trainer                       │   │
│  │  Group_B: Model_1 (Agent_3 only)                                │   │
│  │    → 1 Rollouter, 1 MsgQueue, 1 Trainer                       │   │
│  │                                                                  │   │
│  │  Both groups run fully async, independent training rates        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.8 LoRA-Aware Async Pipeline (L2 Optimization)

```
┌──────────────────────────────────────────────────────────────────────────┐
│              L2 LoRA-Aware Async Architecture                            │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Single Rollouter (vLLM with multi-LoRA)                            │  │
│  │                                                                    │  │
│  │  vLLM Server (persistent, no wake/sleep):                         │  │
│  │  ┌──────────────────────────────────────────┐                     │  │
│  │  │  Base Model (frozen during inference)     │                     │  │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐       │                     │  │
│  │  │  │LoRA_1  │ │LoRA_2  │ │LoRA_3  │       │                     │  │
│  │  │  │(Ag_1)  │ │(Ag_2)  │ │(Ag_3)  │       │                     │  │
│  │  │  └────────┘ └────────┘ └────────┘       │                     │  │
│  │  │  Concurrent requests with different LoRAs │                     │  │
│  │  └──────────────────────────────────────────┘                     │  │
│  └──────────────────────────────┬─────────────────────────────────────┘  │
│                                 │                                        │
│                                 ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Single MessageQueue (but tagged by agent/LoRA)                     │  │
│  │                                                                    │  │
│  │  Queue contents: [(sample, lora_id, param_version), ...]          │  │
│  │                                                                    │  │
│  │  On get_samples():                                                │  │
│  │    Collect N samples, preserve lora_id labels                     │  │
│  └──────────────────────────────┬─────────────────────────────────────┘  │
│                                 │                                        │
│                                 ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Single Trainer (parallel LoRA update)                              │  │
│  │                                                                    │  │
│  │  batch = MQ.get_samples(N)                                        │  │
│  │  # Compute log_prob for all agents in ONE call:                   │  │
│  │  old_log_prob = actor_wg.compute_log_prob(batch)  # LoRA-aware    │  │
│  │                                                                    │  │
│  │  # Compute advantages with AT-GRPO grouping:                     │  │
│  │  adv = at_grpo_advantage(batch, group_by=[env_id, agent_id, turn])│  │
│  │                                                                    │  │
│  │  # Option A: Batched LoRA update (if FSDP supports):             │  │
│  │  actor_wg.update_actor(batch)  # All LoRAs in one pass           │  │
│  │                                                                    │  │
│  │  # Option B: Parallel LoRA updates (ThreadPool):                 │  │
│  │  with ThreadPoolExecutor() as pool:                               │  │
│  │    futures = [pool.submit(update_lora, agent_batch)               │  │
│  │              for agent_batch in split_by_agent(batch)]            │  │
│  │    concurrent.futures.wait(futures)                                │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.9 Complete Data Flow (Optimized Architecture)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Complete Async Data Flow                             │
│                                                                          │
│                                                                          │
│  ┌─────────────┐                                                        │
│  │ Data Loader  │                                                        │
│  │ (env prompts)│                                                        │
│  └──────┬──────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Multi-Agent Coordinator                              │   │
│  │              (many concurrent episodes)                           │   │
│  │                                                                    │   │
│  │  ┌─── Episode_0 ───┐  ┌─── Episode_1 ───┐  ┌─── Episode_K ─┐  │   │
│  │  │ Turn 0:          │  │ Turn 0:          │  │ ...            │  │   │
│  │  │  Ag1→Rollout_0  │  │  Ag1→Rollout_0  │  │               │  │   │
│  │  │  Ag2→Rollout_1  │  │  Ag2→Rollout_1  │  │               │  │   │
│  │  │ Turn 1:          │  │ Turn 1:          │  │               │  │   │
│  │  │  Ag1→Rollout_0  │  │  Ag1→Rollout_0  │  │               │  │   │
│  │  │  Ag2→Rollout_1  │  │  Ag2→Rollout_1  │  │               │  │   │
│  │  │ ...              │  │ ...              │  │               │  │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │   │
│  │           │                      │                     │          │   │
│  │           ▼                      ▼                     ▼          │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │             TrajectoryRouter                                │  │   │
│  │  │  Split trajectories by (agent → model) mapping             │  │   │
│  │  │                                                             │  │   │
│  │  │  Agent_1 data  Agent_2 data  Agent_3 data                 │  │   │
│  │  │      │              │              │                       │  │   │
│  │  │      ▼              ▼              ▼                       │  │   │
│  │  │  ┌────────┐   ┌────────┐   ┌────────┐                   │  │   │
│  │  │  │ MQ_0   │   │ MQ_0   │   │ MQ_1   │  (Ag1,Ag2→M0)   │  │   │
│  │  │  │(lora=1)│   │(lora=2)│   │        │  (Ag3→M1)       │  │   │
│  │  │  └───┬────┘   └───┬────┘   └───┬────┘                   │  │   │
│  │  │      └─────┬───────┘            │                        │  │   │
│  │  └────────────┼────────────────────┼────────────────────────┘  │   │
│  └───────────────┼────────────────────┼───────────────────────────┘   │
│                  │                    │                                │
│                  ▼                    ▼                                │
│  ┌───────────────────────┐  ┌───────────────────────┐                │
│  │  Trainer_0             │  │  Trainer_1             │                │
│  │  (Model_0 + LoRAs)    │  │  (Model_1)             │                │
│  │                        │  │                        │                │
│  │  ┌──────────────────┐ │  │  ┌──────────────────┐ │                │
│  │  │ get_samples()    │ │  │  │ get_samples()    │ │                │
│  │  │ compute_log_prob │ │  │  │ compute_log_prob │ │                │
│  │  │ AT-GRPO advantage│ │  │  │ AT-GRPO advantage│ │                │
│  │  │ update_actor     │ │  │  │ update_actor     │ │                │
│  │  │ update_critic    │ │  │  │ update_critic    │ │                │
│  │  └────────┬─────────┘ │  │  └────────┬─────────┘ │                │
│  │           │            │  │           │            │                │
│  │           ▼            │  │           ▼            │                │
│  │  ParamSync_0           │  │  ParamSync_1           │                │
│  │  NCCL broadcast        │  │  NCCL broadcast        │                │
│  │  to Rollouter_0        │  │  to Rollouter_1        │                │
│  └───────────────────────┘  └───────────────────────┘                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.10 AT-GRPO in Async Context

The async architecture requires careful handling of AT-GRPO advantage computation, because samples from the same episode may span multiple training batches.

```
┌──────────────────────────────────────────────────────────────────────────┐
│            AT-GRPO Advantage in Async Training                           │
│                                                                          │
│  Challenge:                                                              │
│    AT-GRPO groups by (env_id, agent_id, turn_id)                        │
│    In async, samples from same group may arrive at different times       │
│                                                                          │
│  Strategy: Episode-Complete Dispatch                                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  Coordinator waits for ALL rollouts of an env_idx to complete:    │  │
│  │                                                                    │  │
│  │  env_idx=0:                                                       │  │
│  │    rollout_0: [Ag1_t0, Ag2_t0, Ag1_t1, Ag2_t1] ──┐              │  │
│  │    rollout_1: [Ag1_t0, Ag2_t0, Ag1_t1, Ag2_t1] ──┤ GROUPED     │  │
│  │    rollout_2: [Ag1_t0, Ag2_t0, Ag1_t1, Ag2_t1] ──┘              │  │
│  │                                                                    │  │
│  │  Pre-compute AT-GRPO advantages in Coordinator:                   │  │
│  │    for each (agent_id, turn_id) group:                            │  │
│  │      rewards = [r_rollout_0, r_rollout_1, r_rollout_2]           │  │
│  │      adv_i = reward_i - mean(rewards)                             │  │
│  │                                                                    │  │
│  │  Then dispatch to per-model MQs with pre-computed advantages      │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Alternative: Trainer-Side Grouping                                     │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  If advantages must account for cross-model interactions:         │  │
│  │                                                                    │  │
│  │  Trainer_k collects batch from MQ_k                               │  │
│  │  Groups by (env_id, agent_id, turn_id) within batch              │  │
│  │  Computes advantages within available groups                      │  │
│  │                                                                    │  │
│  │  Trade-off: smaller groups → higher variance                     │  │
│  │  Mitigation: require_batches > 1 to accumulate larger groups     │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Part IV: Architecture Comparison Summary

### 4.1 Before vs After

```
┌──────────────────────────────┬──────────────────────────────────────────┐
│       CURRENT (Sync)         │         OPTIMIZED (Async)                │
├──────────────────────────────┼──────────────────────────────────────────┤
│                              │                                          │
│  Rollout ═══► Train ═══►    │  Rollout_0 ────────────────────────►     │
│  Rollout ═══► Train ═══►    │  Train_0   ────────────────────────►     │
│        (strictly             │  Rollout_1 ────────────────────────►     │
│         alternating)         │  Train_1   ────────────────────────►     │
│                              │        (continuous, overlapping)          │
├──────────────────────────────┼──────────────────────────────────────────┤
│ Global sync barrier          │ Per-model sync (independent)             │
│ between all models           │ No cross-model blocking                  │
├──────────────────────────────┼──────────────────────────────────────────┤
│ wake_up/sleep per step       │ Persistent vLLM servers                  │
│ (expensive server restart)   │ In-place weight update                   │
├──────────────────────────────┼──────────────────────────────────────────┤
│ Sequential LoRA training     │ Parallel/batched LoRA update             │
│ per agent                    │ Single FSDP call with LoRA routing       │
├──────────────────────────────┼──────────────────────────────────────────┤
│ All rollouts must finish     │ Episode-complete dispatch                │
│ before any training          │ Training starts as episodes complete     │
├──────────────────────────────┼──────────────────────────────────────────┤
│ Homogeneous training rate    │ Heterogeneous: each model at own pace   │
│ (all models same speed)      │ Small model trains faster than large    │
├──────────────────────────────┼──────────────────────────────────────────┤
│ Fixed GPU allocation         │ Dynamic: inference pools can be shared  │
│ per phase                    │ training pools are dedicated per model   │
├──────────────────────────────┼──────────────────────────────────────────┤
│ One driver controls all      │ Coordinator + per-model Trainers        │
│                              │ (distributed control)                    │
└──────────────────────────────┴──────────────────────────────────────────┘
```

### 4.2 Component Mapping

```
┌──────────────────────────────┬──────────────────────────────────────────┐
│      Current Component       │         Optimized Component              │
├──────────────────────────────┼──────────────────────────────────────────┤
│ MultiAgentsPPOTrainer        │ AsyncMultiAgentTrainerOrchestrator       │
│   (single driver)            │   (launches Coordinator + N Trainers)    │
├──────────────────────────────┼──────────────────────────────────────────┤
│ MultiAgentsExecutionEngine   │ MultiAgentCoordinator (Ray Actor)        │
│   (sync rollout gen)         │   + InferenceRouter                      │
│                              │   + TrajectoryRouter                     │
│                              │   + VersionTracker                       │
├──────────────────────────────┼──────────────────────────────────────────┤
│ AsyncLLMServerManager        │ PersistentRollouter_k (no wake/sleep)   │
│   (wake/sleep per step)      │   (continuous serving with weight sync)  │
├──────────────────────────────┼──────────────────────────────────────────┤
│ RayPPOTrainer                │ AsyncModelTrainer_k (Ray Actor)          │
│   (one per model, sync)      │   (consumes from MQ_k, independent)     │
├──────────────────────────────┼──────────────────────────────────────────┤
│ (none)                       │ MessageQueue_k (Ray Actor)               │
│                              │   (decouples rollout from training)      │
├──────────────────────────────┼──────────────────────────────────────────┤
│ (none)                       │ ParameterSynchronizer_k                  │
│                              │   (NCCL group per model)                 │
├──────────────────────────────┼──────────────────────────────────────────┤
│ core_algo.calculate_reward   │ Same (in Coordinator, before dispatch)   │
├──────────────────────────────┼──────────────────────────────────────────┤
│ mas_turn_order_register      │ Same (registry pattern preserved)        │
└──────────────────────────────┴──────────────────────────────────────────┘
```

### 4.3 Expected Performance Gains

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Expected Speedup Analysis                            │
│                                                                          │
│  Scenario: 2 models (7B + 1.5B), 4 agents, 32 GPUs                    │
│                                                                          │
│  Current bottlenecks breakdown (hypothetical 100-unit step):            │
│  ┌─────────────────────────────────────────────────────┐                │
│  │ wake_up (both models):      5 units                  │                │
│  │ Rollout generation:        50 units (long-tail)      │                │
│  │ sleep (both models):        5 units                  │                │
│  │ Training Model_0 (7B):     25 units                  │                │
│  │ Training Model_1 (1.5B):   10 units                  │                │
│  │ Other overhead:             5 units                  │                │
│  │ TOTAL:                    100 units                  │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                          │
│  Optimized (fully async):                                               │
│  ┌─────────────────────────────────────────────────────┐                │
│  │ No wake/sleep:             -10 units (eliminated)    │                │
│  │ Rollout overlaps training: -25 units (Model_0)       │                │
│  │                            -10 units (Model_1)       │                │
│  │ Parallel LoRA training:     -5 units                 │                │
│  │ No cross-model barrier:     -5 units                 │                │
│  │ EFFECTIVE:                 ~45 units                  │                │
│  │ SPEEDUP:                  ~2.2x                      │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                          │
│  At larger scale (64-128 GPUs), expected 2.5-3.0x based on verl data   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Part V: Implementation Roadmap

### Phase 1: Persistent Rollouters (eliminate wake/sleep)
- Replace AsyncLLMServerManager with PersistentRollouter
- vLLM servers stay alive, use in-place weight loading
- Estimated improvement: 10-15%

### Phase 2: Per-Model MessageQueues (decouple training)
- Create MessageQueue Ray actor per model/policy
- TrajectoryRouter splits episode data to correct queues
- Trainers consume independently
- Estimated improvement: 30-50%

### Phase 3: Full Async Overlap (overlap rollout and training)
- ParameterSynchronizer per model with NCCL groups
- Checkpoint engine for fast weight transfer
- Staleness control and partial rollout support
- Estimated improvement: 2.0-2.5x total

### Phase 4: Parallel LoRA Training (L2 optimization)
- Batch LoRA updates into single FSDP call
- Or use ThreadPoolExecutor for concurrent updates
- Estimated improvement: O(N_agents) → O(1) for training phase

### Phase 5: Heterogeneous Model Support
- Dynamic GPU allocation per model based on compute needs
- Smaller models get fewer GPUs, train faster
- Rate-aware episode scheduling (avoid starving fast models)
