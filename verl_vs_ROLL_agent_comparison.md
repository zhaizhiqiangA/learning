# verl vs ROLL Agent 训练流程对比分析

> 分析时间: 2026-02-09
> 对比框架: verl (ByteDance) 内置 Agent 支持 vs ROLL (Alibaba) Agentic Pipeline

## 目录

- [一、架构设计对比](#一架构设计对比)
- [二、环境/工具集成方式](#二环境工具集成方式)
- [三、多轮交互机制](#三多轮交互机制)
- [四、数据流和训练循环](#四数据流和训练循环)
- [五、分布式执行对比](#五分布式执行对比)
- [六、核心特性对比表](#六核心特性对比表)
- [七、使用场景建议](#七使用场景建议)

---

## 一、架构设计对比

### verl 架构：异步 Agent Loop + 工具系统

#### 核心组件

```
RayPPOTrainer
    ↓
AgentLoopWorker (Ray Actor)
    ├─ AsyncLLMServerManager (负载均衡，sticky session)
    │   ├─ vLLM Server
    │   ├─ SGLang Server
    │   └─ TRTLLM Server
    │
    ├─ ToolAgentLoop (状态机)
    │   ├─ PENDING (应用 chat template)
    │   ├─ GENERATING (从 LLM 服务器生成)
    │   ├─ PROCESSING_TOOLS (并行执行工具)
    │   ├─ INTERACTING (可选用户反馈)
    │   └─ TERMINATED (输出结果)
    │
    └─ Tool/Interaction 注册系统
        ├─ BaseTool (execute, calc_reward)
        └─ BaseInteraction (generate_response)
```

**关键特点:**
- **工具驱动**: 环境交互通过工具抽象实现
- **状态机设计**: 清晰的状态转换管理多轮交互
- **异步执行**: 每个样本独立异步处理
- **Response Mask**: `1` = LLM token, `0` = 工具/交互 token

**关键文件:**
- `/verl/experimental/agent_loop/agent_loop.py` - 核心框架 (1006 行)
- `/verl/experimental/agent_loop/tool_agent_loop.py` - 工具 Agent (475 行)
- `/verl/tools/` - 工具系统
- `/verl/interactions/` - 交互系统

---

### ROLL 架构：独立 Environment Worker + 异步队列

#### 核心组件

```
AgenticPipeline
    ↓
RolloutScheduler
    ├─ GroupQueueManager (异步队列，Ray Actor)
    │   └─ GroupQueue (per environment group)
    │
    ├─ GenerateScheduler (LLM 调度)
    │
    └─ EnvironmentWorker Cluster (Ray Actors)
        ├─ EnvironmentWorker (管理多个环境)
        │   ├─ EnvManager (ThreadPoolExecutor)
        │   │   ├─ GEM 环境实例
        │   │   └─ Tool Wrapper (可选)
        │   │
        │   └─ LLM Proxy
        │       ├─ PolicyProxy (vLLM/SGLang)
        │       ├─ OpenAIProxy
        │       └─ RandomProxy
        │
        └─ run_rollout_loop()
            ├─ reset()
            ├─ while running:
            │   ├─ make_decision() (LLM 生成)
            │   ├─ step() (环境执行)
            │   └─ formulate_rollouts()
            └─ queue.put(rollouts)
```

**关键特点:**
- **环境驱动**: 完整的环境作为一等公民
- **独立 Worker**: 环境在独立的 Ray Actor 中运行
- **异步队列**: GroupQueueManager 管理环境输出
- **Active Masks**: 标记哪些样本仍在运行

**关键文件:**
- `/ROLL/roll/pipeline/agentic/agentic_pipeline.py` - 主流程 (594 行)
- `/ROLL/roll/pipeline/agentic/environment_worker.py` - 环境 Worker (130 行)
- `/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py` - 轨迹管理 (400+ 行)
- `/ROLL/roll/distributed/scheduler/rollout_scheduler.py` - 异步调度 (402 行)

---

### 架构对比表

| 维度 | verl | ROLL |
|------|------|------|
| **核心抽象** | 工具 (Tool) + 交互 (Interaction) | 环境 (Environment) + 轨迹 (Trajectory) |
| **执行模型** | 异步任务 per sample | 独立 Worker per environment group |
| **LLM 集成** | AsyncLLMServerManager | LLM Proxy (多后端) |
| **多轮控制** | 状态机 (PENDING→TERMINATED) | 循环 (while running) |
| **数据标记** | response_mask (0/1) | active_masks (bool) |
| **部署方式** | LLM 服务器 + Agent Worker | Environment Worker + Training Worker |

---

## 二、环境/工具集成方式

### verl：工具和交互抽象

#### 工具系统

**BaseTool 接口:**
```python
class BaseTool:
    @abstractmethod
    def create(self, **kwargs) -> "BaseTool":
        """创建工具实例"""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResponse:
        """执行工具并返回响应"""

    @abstractmethod
    def calc_reward(self, **kwargs) -> float:
        """计算工具相关奖励"""

    @abstractmethod
    def release(self):
        """释放资源"""
```

**工具响应:**
```python
class ToolResponse:
    text: str  # 文本响应
    image: Optional[List[Image.Image]]  # 多模态图像
    video: Optional[List[Tuple[Tensor, dict]]]  # 视频
```

**内置工具:**
- `Gsm8kTool` - GSM8K 数学题目
- `SearchTool` - 网络搜索
- `SandboxFusionTools` - ROCK 沙箱工具集
- `ImageZoomInTool` - 图像处理
- `MCPBaseTool` - Model Context Protocol 工具

#### 交互系统

**BaseInteraction 接口:**
```python
class BaseInteraction:
    @abstractmethod
    def start_interaction(self, **kwargs):
        """初始化交互"""

    @abstractmethod
    def generate_response(self, **kwargs) -> Tuple[str, float]:
        """生成用户响应和奖励"""

    @abstractmethod
    def finalize_interaction(self):
        """清理资源"""
```

**内置交互:**
- `Gsm8kInteraction` - GSM8K 题目交互
- `WeatherInteraction` - 天气查询交互

#### 配置示例

**工具配置** (`tool_config.yaml`):
```yaml
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "计算 GSM8K 数学问题奖励"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
```

**多轮配置** (`ppo_trainer.yaml`):
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True
      max_assistant_turns: 5
      max_parallel_calls: 1  # 并行工具调用数
      tool_config_path: "config/tool_config/gsm8k_tool_config.yaml"
      interaction_config_path: "config/interaction_config/gsm8k_interaction_config.yaml"
      format: hermes  # 或 gpt-oss, qwen
```

---

### ROLL：GEM 环境协议

#### GEM 环境接口

**标准环境接口:**
```python
class GEMEnvironment:
    def reset(self) -> Tuple[obs, info]:
        """重置环境，返回初始观察"""

    def step(self, action) -> Tuple[obs, reward, done, info]:
        """执行动作，返回新观察、奖励、终止标志、信息"""

    def get_admissible_commands(self) -> List[str]:
        """获取可用动作集合（可选）"""
```

#### 环境管理器

**BaseEnvManager 接口:**
```python
class BaseEnvManager:
    def reset(self) -> Tuple[Dict, info]:
        """返回 {'text': ..., 'image': ..., 'anchor': ...}"""

    def step(self, text_actions: List[str]) -> Tuple[obs, rewards, dones, info]:
        """执行文本动作"""

    def make_decision(self, messages) -> str:
        """通过 LLM Proxy 生成动作"""

    def format_messages(self, obs) -> List[Dict]:
        """格式化观察为 LLM 消息"""
```

#### LLM Proxy 系统

**多后端支持:**
```python
BaseLLMProxy
├─ PolicyProxy (vLLM/SGLang)
├─ OpenAIProxy (外部 API)
└─ RandomProxy (基线)
```

**配置示例:**
```yaml
train_env_manager:
  num_env_groups: 128
  group_size: 8
  max_env_num_per_worker: 16
  llm_proxy:
    type: policy  # policy / openai / random
    backend: vllm
    generating_args:
      max_new_tokens: 128
      temperature: 0.99
```

#### 环境示例

**内置环境支持:**
- Sokoban - 推箱子游戏
- WebShop - 电商环境
- FrozenLake - 冰湖游戏
- ALFWorld (通过 verl-agent)
- 任何 GEM 兼容环境

**自定义环境配置:**
```yaml
custom_envs:
  SimpleSokoban:
    env_type: sokoban
    dim_room: [6, 6]
    num_boxes: 1

  LargerSokoban:
    env_type: sokoban
    dim_room: [10, 10]
    num_boxes: 2
```

---

### 环境/工具集成对比

| 维度 | verl (工具系统) | ROLL (环境系统) |
|------|----------------|----------------|
| **抽象层次** | 工具调用 (函数级) | 环境交互 (游戏级) |
| **适用场景** | 工具调用、API 集成、问答 | 游戏、仿真、具身智能 |
| **状态管理** | 隐式 (在消息历史中) | 显式 (环境状态) |
| **奖励来源** | 工具返回 + 交互模块 | 环境奖励 + 规则奖励 |
| **多模态** | ✅ 原生支持 (图像/视频) | ⚖️ 部分支持 (需适配) |
| **并行化** | 工具并行调用 | 环境并行执行 |
| **标准化** | 自定义工具接口 | GEM 协议 |
| **扩展性** | 添加新工具类 | 添加新环境类 |

---

## 三、多轮交互机制

### verl：状态机驱动

#### ToolAgentLoop 状态机

**状态定义:**
```python
class AgentState(Enum):
    PENDING = "pending"           # 初始状态，准备提示
    GENERATING = "generating"     # LLM 生成中
    PROCESSING_TOOLS = "processing_tools"  # 执行工具
    INTERACTING = "interacting"   # 用户交互
    TERMINATED = "terminated"     # 终止
```

**状态转换流程:**
```
PENDING
  ↓ apply_chat_template()
  ↓ 添加 tool schemas 到消息
  ↓
GENERATING
  ↓ AsyncLLMServerManager.generate()
  ↓ 提取工具调用 (ToolParser)
  ↓
  ├─ 有工具调用? → PROCESSING_TOOLS
  │   ↓ 并行执行工具 (max_parallel_calls)
  │   ↓ 编码 ToolResponse → token_ids
  │   ↓ 更新 response_mask (工具 token = 0)
  │   ↓ 添加工具响应到消息
  │   ↓ 检查 max_assistant_turns
  │   └─ 未达到上限? → GENERATING (下一轮)
  │
  ├─ 有交互? → INTERACTING
  │   ↓ BaseInteraction.generate_response()
  │   ↓ 收集 turn_score
  │   ↓ 添加用户反馈到消息
  │   └─ 未达到 max_user_turns? → GENERATING
  │
  └─ 否则 → TERMINATED
      ↓ 输出 AgentLoopOutput
```

#### AgentData 状态容器

**核心字段:**
```python
class AgentData:
    messages: list[dict]  # 对话历史
    prompt_ids: list[int]  # 累积的所有 token
    response_ids: list[int]  # 最后一个响应
    response_mask: list[int]  # 0 = 工具/交互, 1 = LLM

    tool_calls: list[FunctionCall]  # 工具调用缓冲
    turn_scores: list[float]  # 每轮交互奖励
    tool_rewards: list[float]  # 工具执行奖励

    user_turns: int  # 用户轮次计数
    assistant_turns: int  # 助手轮次计数

    extra_fields: dict  # 动态字段
```

#### Response Mask 机制

**关键特性:**
- `response_mask[i] = 1` → token i 由 LLM 生成，计算梯度
- `response_mask[i] = 0` → token i 来自工具/交互，不计算梯度

**示例:**
```
消息: "用户: 2+3=?" → LLM: "<tool_call>calc(2+3)</tool_call>" → 工具: "5" → LLM: "答案是5"

prompt_ids:    [用户, 2, +, 3, =, ?]
response_ids:  [<tool>, calc, ..., </tool>, 5, 答案, 是, 5]
response_mask: [1, 1, 1, 1, 1, 0, 1, 1, 1]
                └─ LLM 生成 ─┘  └工具┘ └─ LLM ─┘
```

#### 工具并行化

**配置:**
```yaml
multi_turn:
  max_parallel_calls: 4  # 最多并行 4 个工具调用
```

**实现:**
```python
# tool_agent_loop.py
async def _process_tools_async(self, tool_calls):
    tasks = [tool.execute(**kwargs) for tool in tool_calls]
    results = await asyncio.gather(*tasks)  # 并行执行
    return results
```

---

### ROLL：轨迹收集循环

#### TrajEnvManager 流程

**run_rollout_loop() 伪代码:**
```python
def run_rollout_loop(self, env_manager, llm_proxy, config):
    # 1. 重置环境
    obs, info = env_manager.reset()

    # 2. 初始化轨迹
    trajectory = []
    done = False

    # 3. 主循环
    while not done and len(trajectory) < max_steps:
        # 3a. 格式化观察
        messages = env_manager.format_messages(obs)

        # 3b. LLM 生成动作
        action_text = llm_proxy.generate(messages)

        # 3c. 环境步进
        next_obs, reward, done, info = env_manager.step(action_text)

        # 3d. 记录轨迹
        trajectory.append({
            'obs': obs,
            'action': action_text,
            'reward': reward,
            'done': done,
            'info': info
        })

        obs = next_obs

    # 4. 格式化轨迹为 DataProto
    return formulate_rollouts(trajectory)
```

#### 环境分组机制

**GroupQueue 系统:**
```python
# 配置
num_env_groups: 128  # 128 个环境组
group_size: 8        # 每组 8 个环境（相同初始状态）

# 实现
for group_id in range(num_env_groups):
    env_group = []
    for i in range(group_size):
        env = create_env(seed=group_id)  # 同组共享种子
        env_group.append(env)

    # 并行运行环境组
    rollouts = run_parallel(env_group)
    queue.put(group_id, rollouts)
```

#### 异步队列管理

**GroupQueueManager 特性:**
```python
class GroupQueueManager:
    async def wait_for_group(self, group_id):
        """异步等待某个组完成"""

    async def collect_batch(self, batch_size):
        """收集指定数量的 rollouts"""
        rollouts = []
        while len(rollouts) < batch_size:
            for group_id in pending_groups:
                if group_ready(group_id):
                    rollouts.extend(await self.get_group(group_id))
        return rollouts
```

#### Active Masks 机制

**与 response_mask 的区别:**
- **ROLL active_masks**: 标记哪些环境仍在运行 (批次级别)
- **verl response_mask**: 标记哪些 token 由 LLM 生成 (token 级别)

**示例:**
```python
# ROLL
active_masks = [True, True, False, True]  # 环境 2 已完成
rewards[active_masks] += step_rewards  # 只更新活跃环境

# verl
response_mask = [1, 1, 0, 0, 1, 1]  # token 2-3 来自工具
loss = (policy_loss * response_mask).sum()  # 只计算 LLM token 的损失
```

---

### 多轮机制对比

| 维度 | verl (状态机) | ROLL (循环) |
|------|--------------|------------|
| **控制流** | 显式状态转换 | 隐式循环条件 |
| **复杂度** | 高（5 种状态） | 低（while 循环） |
| **可扩展性** | 易于添加新状态 | 需修改循环逻辑 |
| **调试友好** | ✅ 状态清晰 | ⚖️ 需日志追踪 |
| **并行化** | 工具级并行 | 环境级并行 |
| **终止条件** | max_assistant_turns | max_steps + done |
| **中间奖励** | turn_scores + tool_rewards | step_rewards |

---

## 四、数据流和训练循环

### verl 数据流

#### 端到端流程

```
DataLoader (Parquet)
  ↓ raw_prompt, tools_kwargs
  ↓
AgentLoopWorker.generate_sequences()
  ├─ For each sample (异步):
  │   ├─ ToolAgentLoop.run()
  │   │   ├─ PENDING → 应用 chat template
  │   │   ├─ GENERATING → 从 LLM 服务器生成
  │   │   ├─ PROCESSING_TOOLS → 执行工具
  │   │   ├─ INTERACTING → 可选交互
  │   │   └─ TERMINATED → AgentLoopOutput
  │   │
  │   └─ _agent_loop_postprocess()
  │       ├─ Tokenizer padding (左对齐)
  │       ├─ 构建 attention_mask, position_ids
  │       └─ _InternalAgentLoopOutput
  │
  └─ 合并为 DataProto batch
      ├─ input_ids: [bsz, seq_len]
      ├─ attention_mask: [bsz, seq_len]
      ├─ response_mask: [bsz, max_response_len]
      └─ extra_fields: {turn_scores, tool_rewards}
  ↓
RewardModelWorker (可选)
  ↓ scores
  ↓
RayPPOTrainer.update()
  ├─ ActorWorker: 计算 policy loss (使用 response_mask)
  ├─ CriticWorker: 计算 value loss
  └─ ReferenceWorker: 计算 KL divergence
```

#### 关键数据结构

**AgentLoopOutput:**
```python
class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]  # 提示 token
    response_ids: list[int]  # 响应 token (包含工具调用和响应)
    response_mask: list[int]  # ← 关键: 标记 LLM token
    response_logprobs: Optional[list[float]]
    reward_score: Optional[float]
    num_turns: int  # 交互轮数
    metrics: AgentLoopMetrics
    extra_fields: dict  # {turn_scores, tool_rewards}
```

**_InternalAgentLoopOutput (批处理):**
```python
class _InternalAgentLoopOutput:
    input_ids: Tensor  # [bsz, seq_len]
    attention_mask: Tensor  # [bsz, seq_len]
    position_ids: Tensor  # [bsz, seq_len]
    response_mask: Tensor  # [bsz, max_response_len]
    responses: Tensor  # [bsz, max_response_len]
    # ... 其他字段
```

---

### ROLL 数据流

#### 端到端流程

```
DataLoader (Parquet) / 无数据（纯环境）
  ↓ prompts (可选)
  ↓
AgenticPipeline.run()
  ↓ 分发到 EnvironmentWorker
  ↓
EnvironmentWorker.rollout()
  ├─ For each environment:
  │   ├─ EnvManager.reset()
  │   ├─ While not done:
  │   │   ├─ format_messages()
  │   │   ├─ LLMProxy.generate()
  │   │   ├─ EnvManager.step()
  │   │   └─ 记录 (obs, action, reward, done)
  │   └─ formulate_rollouts()
  │
  └─ GroupQueue.put(rollouts)
  ↓
RolloutScheduler.collect_batch()
  ├─ 从 GroupQueueManager 收集
  ├─ Batch adjustment (copy/delete/auto)
  └─ DataProto batch
      ├─ input_ids: [bsz, seq_len]
      ├─ responses: [bsz, response_len]
      ├─ active_masks: [bsz]
      └─ non_tensor_batch: {uid, traj_uid, rewards, step_scores}
  ↓
AgenticPipeline.compute_rewards()
  ├─ agentic_reward_norm() (按组归一化)
  └─ compute_discounted_returns() (GiGPO)
  ↓
AgenticPipeline.compute_advantages()
  ├─ agentic_compute_advantage()
  └─ GAE / GRPO / GiGPO 估计器
  ↓
AgenticPipeline.update_policy()
  ├─ ActorWorker: PPO loss
  ├─ CriticWorker: Value loss
  └─ ReferenceWorker: KL divergence
```

#### 关键数据结构

**DataProto (ROLL):**
```python
class DataProto:
    batch: dict[str, Tensor]
        input_ids: [bsz, seq_len]
        responses: [bsz, response_len]
        advantages: [bsz, response_len]
        old_log_probs: [bsz, response_len]
        # ...

    non_tensor_batch: dict[str, np.ndarray]
        uid: [bsz]  # 环境组 ID
        traj_uid: [bsz]  # 轨迹 ID
        rewards: [bsz]  # episode 奖励
        step_scores: [bsz]  # 步级奖励 (GiGPO)
        active_masks: [bsz]  # 是否活跃
        # ...

    meta_info: dict
```

---

### 数据流对比

| 维度 | verl | ROLL |
|------|------|------|
| **输入来源** | Parquet + tools_kwargs | Parquet / 无 (纯环境) |
| **生成方式** | 异步任务 per sample | 并行环境 per worker |
| **中间表示** | AgentLoopOutput | 轨迹列表 |
| **批处理** | _InternalAgentLoopOutput | DataProto |
| **梯度掩码** | response_mask (token 级) | active_masks (样本级) |
| **奖励来源** | tool_rewards + turn_scores | environment rewards + step_scores |
| **后处理** | Tokenizer padding | Batch adjustment |

---

## 五、分布式执行对比

### verl 分布式架构

#### Ray 部署拓扑

```
Driver Process (main_ppo.py)
  ↓
TaskRunner (Ray Actor)
  ↓
RayPPOTrainer
  ├─ ActorRolloutRefWorkerGroup
  │   ├─ ActorWorker (训练, FSDP/Megatron)
  │   ├─ RolloutWorker (推理, vLLM/SGLang)
  │   │   └─ AgentLoopWorker (Agent 模式)
  │   │       ├─ AsyncLLMServerManager
  │   │       │   ├─ vLLM Server 1 (独立进程)
  │   │       │   ├─ vLLM Server 2
  │   │       │   └─ ...
  │   │       └─ ToolAgentLoop × num_workers
  │   └─ ReferenceWorker (KL 计算)
  │
  └─ CriticWorkerGroup (可选)
      └─ CriticWorker (FSDP/Megatron)
```

**关键特性:**
- **LLM 服务器独立**: vLLM/SGLang 作为独立服务运行
- **异步任务**: 每个样本独立异步处理
- **Sticky Session**: 样本优先路由到之前的服务器（缓存 KV）
- **负载均衡**: AsyncLLMServerManager 自动分配

**配置示例:**
```yaml
actor_rollout_ref:
  rollout:
    name: sglang  # 或 vllm, trtllm
    tensor_model_parallel_size: 2
    num_workers: 4  # AgentLoopWorker 数量

    # LLM 服务器配置
    servers:
      - url: "http://localhost:30000"
      - url: "http://localhost:30001"
```

---

### ROLL 分布式架构

#### Ray 部署拓扑

```
Driver Process (start_agentic_pipeline.py)
  ↓
AgenticPipeline
  ├─ RolloutScheduler (Ray Actor)
  │   ├─ GroupQueueManager (Ray Actor)
  │   │   └─ GroupQueue × num_env_groups
  │   │
  │   ├─ GenerateScheduler
  │   │   └─ LLM 服务器池
  │   │
  │   └─ EnvironmentWorker Cluster
  │       ├─ EnvironmentWorker 1 (Ray Actor)
  │       │   ├─ EnvManager 1 (Thread)
  │       │   ├─ EnvManager 2 (Thread)
  │       │   └─ ...
  │       ├─ EnvironmentWorker 2
  │       └─ ...
  │
  ├─ ActorWorkerGroup
  │   ├─ ActorTrainWorker (Megatron/DeepSpeed)
  │   ├─ ActorInferWorker (vLLM/SGLang)
  │   └─ ReferenceWorker
  │
  └─ CriticWorkerGroup (可选)
```

**关键特性:**
- **环境独立部署**: EnvironmentWorker 可在不同机器
- **异步队列**: GroupQueueManager 解耦环境和训练
- **线程池**: 每个 Worker 管理多个环境 (ThreadPoolExecutor)
- **灵活后端**: LLM Proxy 支持多种推理后端

**配置示例:**
```yaml
# 环境资源配置
train_env_manager:
  num_env_groups: 128
  group_size: 8
  max_env_num_per_worker: 16
  resources_per_worker:
    num_cpus: 0.1
    num_gpus: 0

# 训练资源配置
actor_train:
  strategy: megatron_train
  device_mapping: [0, 1, 2, 3]

# 推理资源配置
actor_infer:
  strategy: vllm
  device_mapping: [4, 5, 6, 7]
```

---

### 分布式对比

| 维度 | verl | ROLL |
|------|------|------|
| **环境执行** | 集成在 AgentLoopWorker | 独立 EnvironmentWorker |
| **LLM 推理** | 独立服务器 (OpenAI 兼容) | LLM Proxy (集成或独立) |
| **任务调度** | 异步任务 per sample | 异步队列 per group |
| **资源隔离** | Agent + LLM 分离 | Environment + Training 分离 |
| **扩展性** | LLM 服务器可独立扩展 | 环境和训练可独立扩展 |
| **容错性** | 样本级容错 | 环境组级容错 |
| **跨机部署** | LLM 服务器可跨机 | 环境 Worker 可跨机 |

---

## 六、核心特性对比表

### 架构和设计

| 特性 | verl | ROLL |
|------|------|------|
| **核心抽象** | 工具 + 交互 | 环境 + 轨迹 |
| **多轮控制** | 状态机 (5 状态) | 循环 (while done) |
| **执行模型** | 异步任务 | 独立 Worker + 队列 |
| **LLM 集成** | AsyncLLMServerManager | LLM Proxy (多后端) |
| **环境协议** | 自定义 Tool/Interaction | GEM 标准协议 |
| **配置系统** | Hydra (嵌入式) | Hydra (独立配置) |

### 功能支持

| 特性 | verl | ROLL |
|------|------|------|
| **工具调用** | ✅ 原生支持 | ⚖️ Tool Wrapper |
| **具身环境** | ❌ 不支持 | ✅ 原生支持 (GEM) |
| **多模态** | ✅ 图像/视频 | ⚖️ 需适配 |
| **并行工具** | ✅ max_parallel_calls | ❌ 顺序执行 |
| **外部 API** | ✅ OpenAI Proxy | ✅ OpenAI Proxy |
| **交互模块** | ✅ BaseInteraction | ❌ 不支持 |
| **环境渲染** | ❌ 不支持 | ✅ dump_rollout_render |

### 数据和训练

| 特性 | verl | ROLL |
|------|------|------|
| **梯度掩码** | response_mask (token 级) | active_masks (样本级) |
| **中间奖励** | turn_scores + tool_rewards | step_scores + episode_rewards |
| **奖励归一化** | 算法内置 | 可配置 (3 种分组 × 3 种方法) |
| **优势估计** | GAE / GRPO | GAE / GRPO / GiGPO (需 verl-agent) |
| **批次调整** | 固定 | 4 种模式 (copy/delete/auto/random) |
| **数据来源** | Parquet + tools_kwargs | Parquet / 纯环境 |

### 分布式和部署

| 特性 | verl | ROLL |
|------|------|------|
| **LLM 部署** | 独立服务器 (必需) | 集成或独立 |
| **环境部署** | 集成在 Worker | 独立 Worker |
| **跨机部署** | LLM 服务器 | 环境 + 训练 |
| **异步支持** | ✅ 任务级异步 | ✅ 队列级异步 |
| **负载均衡** | ✅ Sticky Session | ⚖️ 手动配置 |
| **容错性** | 样本级 | 环境组级 |
| **资源隔离** | Agent + LLM | Environment + Training |

### 易用性和生态

| 特性 | verl | ROLL |
|------|------|------|
| **学习曲线** | ⚖️ 中等（状态机） | ⚖️ 中等（配置驱动） |
| **文档完善** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **示例丰富** | ⭐⭐⭐⭐ (GSM8K, Weather) | ⭐⭐⭐⭐ (Sokoban, WebShop) |
| **社区支持** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **调试工具** | ⚖️ 日志 + metrics | ✅ 渲染 + 详细 metrics |
| **可扩展性** | ✅ 易添加工具 | ✅ 易添加环境 |

---

## 七、使用场景建议

### 选择 verl 的场景

#### ✅ 工具调用和 API 集成

**适用任务:**
- LLM + 工具 (搜索、计算器、数据库查询)
- API 集成 (天气、股票、新闻)
- 函数调用 (OpenAI function calling)
- 代码解释器

**优势:**
- 原生工具系统，无需额外适配
- 并行工具调用提高效率
- 状态机清晰管理复杂交互
- 多模态工具响应 (图像/视频)

**示例:**
```yaml
# GSM8K 数学问题求解
tools:
  - class_name: verl.tools.gsm8k_tool.Gsm8kTool
    tool_schema:
      function:
        name: calc_gsm8k_reward
        description: "计算数学题目答案"
```

---

#### ✅ 问答和对话系统

**适用任务:**
- 多轮问答
- 带反馈的对话
- 教育辅导 (Socratic questioning)
- 客服系统

**优势:**
- 交互模块支持动态用户反馈
- turn_scores 实现细粒度奖励
- 易于实现人类反馈循环

**示例:**
```yaml
interaction:
  - name: weather
    class_name: verl.interactions.weather_interaction.WeatherInteraction
```

---

#### ✅ 独立 LLM 服务器架构

**适用场景:**
- LLM 服务器与训练分离
- 多个训练任务共享 LLM 服务器
- 需要 sticky session (KV 缓存复用)

**优势:**
- AsyncLLMServerManager 管理多服务器
- 负载均衡和容错
- 独立扩展推理能力

---

### 选择 ROLL 的场景

#### ✅ 游戏和仿真环境

**适用任务:**
- 游戏 AI (Sokoban, FrozenLake, Atari)
- 物理仿真
- 机器人控制
- 策略游戏

**优势:**
- GEM 协议标准化环境接口
- 环境渲染和可视化
- 环境分组提高采样效率
- 支持复杂环境状态

**示例:**
```yaml
custom_envs:
  Sokoban:
    env_type: sokoban
    dim_room: [10, 10]
    num_boxes: 3
```

---

#### ✅ 具身智能和机器人

**适用任务:**
- ALFWorld (具身 AI)
- WebShop (网页交互)
- 机器人导航
- 物理操作

**优势:**
- 完整的环境状态管理
- 支持长轨迹 (max_steps)
- 环境独立部署 (跨机)
- 详细的轨迹记录

---

#### ✅ 生产级分布式训练

**适用场景:**
- 大规模环境并行
- 环境计算密集 (需独立机器)
- 需要灵活资源调度
- 需要高容错性

**优势:**
- 环境和训练完全解耦
- 异步队列缓冲波动
- GroupQueueManager 自动负载均衡
- 环境故障不影响训练

---

#### ✅ 实验和研究

**适用场景:**
- 快速原型验证
- 多域混合训练
- 奖励归一化研究
- 批次调整策略研究

**优势:**
- 配置驱动，易于实验
- 支持多种奖励归一化策略
- 4 种批次调整模式
- RLVR Pipeline 支持多域

---

### 混合使用建议

#### 场景 1: 工具 + 环境混合

**任务:** Agent 需要同时使用工具和交互环境

**方案:**
1. 在 ROLL 中使用 Tool Wrapper 包装工具
2. 或在 verl 中将环境封装为 Tool/Interaction

**示例:**
```python
# ROLL 方案
class ToolWrapper:
    def __init__(self, tools):
        self.tools = tools

    def step(self, action):
        if is_tool_call(action):
            return self.tools.execute(action)
        else:
            return self.env.step(action)
```

---

#### 场景 2: 多阶段训练

**任务:** 先用工具训练，再用环境训练

**方案:**
1. **阶段 1 (verl)**: 使用工具系统训练基础能力
2. **阶段 2 (ROLL)**: 在环境中训练决策能力

---

#### 场景 3: 分布式异构训练

**任务:** 不同类型的任务需要不同资源

**方案:**
- **verl**: 处理工具调用任务 (GPU 轻量)
- **ROLL**: 处理环境仿真任务 (CPU 密集)

---

## 八、核心文件索引

### verl 核心文件

| 文件路径 | 功能 | 行数 |
|---------|------|-----|
| `/verl/experimental/agent_loop/agent_loop.py` | Agent 框架核心 | 1006 |
| `/verl/experimental/agent_loop/tool_agent_loop.py` | 工具 Agent 状态机 | 475 |
| `/verl/experimental/agent_loop/single_turn_agent_loop.py` | 单轮 Agent | 85 |
| `/verl/experimental/agent_loop/tool_parser.py` | 工具提取器 | - |
| `/verl/tools/base_tool.py` | 工具抽象基类 | - |
| `/verl/interactions/base_interaction.py` | 交互抽象基类 | - |
| `/verl/workers/config/rollout.py` | Rollout 配置 (MultiTurnConfig) | - |
| `/verl/trainer/ppo/ray_trainer.py` | RayPPOTrainer 集成 | 1650+ |
| `/verl/examples/sglang_multiturn/` | 完整示例 | - |

### ROLL 核心文件

| 文件路径 | 功能 | 行数 |
|---------|------|-----|
| `/ROLL/roll/pipeline/agentic/agentic_pipeline.py` | Agentic Pipeline 主流程 | 594 |
| `/ROLL/roll/pipeline/agentic/agentic_config.py` | 配置定义 | 290 |
| `/ROLL/roll/pipeline/agentic/environment_worker.py` | 环境 Worker | 130 |
| `/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py` | 轨迹管理器 | 400+ |
| `/ROLL/roll/pipeline/agentic/env_manager/base_env_manager.py` | 环境管理抽象 | - |
| `/ROLL/roll/pipeline/agentic/llm_proxy/` | LLM Proxy 系统 | - |
| `/ROLL/roll/distributed/scheduler/rollout_scheduler.py` | 异步调度器 | 402 |
| `/ROLL/roll/distributed/scheduler/protocol.py` | DataProto 定义 | - |
| `/ROLL/roll/pipeline/agentic/utils.py` | 奖励/优势工具 | - |
| `/ROLL/examples/qwen2.5-0.5B-agentic/` | 完整示例 | - |

---

## 九、总结

### verl 的核心优势

1. **工具调用原生支持** - 状态机 + 并行工具执行
2. **多模态能力** - 图像/视频工具响应
3. **LLM 服务器架构** - 独立部署，负载均衡
4. **response_mask 机制** - token 级精确梯度控制
5. **易于扩展** - 添加工具只需实现 BaseTool

### ROLL 的核心优势

1. **环境完整支持** - GEM 协议，标准化接口
2. **分布式灵活性** - 环境和训练完全解耦
3. **配置驱动** - Hydra 配置，易于实验
4. **异步队列系统** - 高容错，负载均衡
5. **生产级稳定性** - 详细监控，渲染可视化

### 关键区别

- **verl**: 工具驱动，适合 API 集成和问答
- **ROLL**: 环境驱动，适合游戏和具身智能

### 未来融合方向

1. **verl 集成 GEM 环境** - 将 GEM 环境封装为 Tool/Interaction
2. **ROLL 集成工具系统** - 在 EnvManager 中支持并行工具调用
3. **统一配置系统** - 合并两者的配置优势
4. **共享分布式架构** - 结合异步队列和 LLM 服务器

---

*本分析基于对 verl 和 ROLL 代码库的深入探索，实际使用中应根据具体任务选择合适的框架。*
