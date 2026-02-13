## 一、问题背景

### 1.1 现有技术
当前提升大语言模型（LLM）智能体性能主要有两种互补方法：

1. **多智能体系统（MAS）**
   - 通过基于角色的协作增强任务专业化性能
   - 通常采用基于提示的增强，在共享的LLM策略上实现角色协调
   - 主要在推理时进行设计优化

2. **强化学习（RL）**
   - 将LLM视为策略，通过迭代更新权重来增强决策能力
   - 利用环境奖励训练更强的策略（如GRPO风格优化）
   - 需要支持可扩展rollouts和在线更新的训练系统

### 1.2 核心挑战
将RL应用于MAS训练面临两个耦合挑战：

#### 算法层面
- **标准GRPO的分组假设失效**：在MAS中，提示因角色和轮次而异
- 不同角色在不同轮次的提示包含：
  - 角色特定的上下文
  - 跨智能体交互历史
  - 无法使用传统的"相同问题、多个响应"分组方式

#### 系统层面
- 需要支持基于MAS工作流的rollouts
- 需要同时支持单策略和多策略的在线更新
- 现有RL框架（VERL、AReaL等）主要支持单一模型

---

## 二、主要解决问题

### 2.1 核心贡献

1. **AT-GRPO算法**
   - Agent- and Turn-wise grouped RL算法
   - 专为MAS设计的分组式强化学习方法
   - 支持细粒度的信用分配（credit assignment）

2. **MAS训练系统**
   - 支持多样化的MAS工作流执行rollouts
   - 支持单策略（role-sharing）和多策略（role-specialized）的在线RL更新
   - 基于HybridFlow风格的资源池架构

3. **显著性能提升**
   - **长期规划任务**：准确率从14-47%提升到96.0-99.5%
   - **代码任务**：平均提升3.87-7.62%
   - **数学任务**：平均提升9.0-17.93%

### 2.2 关键发现

1. **RL训练强化角色专业化**
   - RL训练促进MAS中角色特定的专业化
   - 交换训练后的角色会导致灾难性性能下降（96%→6%）

2. **策略选择取决于任务特性**
   - **代码任务**：角色专业化策略更优（Coder和Tester功能高度不同）
   - **数学任务**：共享策略可能更优（Tool和Reasoner功能重叠）
   - **游戏/规划任务**：两种配置都接近饱和性能

---

## 三、核心方法论

### 3.1 AT-GRPO算法设计

#### 3.1.1 问题形式化
将N个智能体的LLM系统建模为马尔可夫博弈：
```
M = (S, {Ai}^N_{i=1}, T, {ri}^N_{i=1}, T, H)
```
- S: 状态空间
- Ai: 智能体i的动作空间
- T: 转移函数（intra-turn微转移）
- ri: 智能体i的奖励函数
- T: 轮次范围
- H: 优化步骤范围

#### 3.1.2 三大核心设计

##### 1. 树状采样（Tree-structured Sampling）
- 在每个轮次t，为每个智能体i采样K个候选动作
- 计算这K个候选的优势值（advantages）
- **贪婪选择**奖励最高的候选作为执行动作
- 目的：集中探索在协调关键决策上，保持正负样本平衡

**对比传统并行采样**：
- 并行采样：从初始状态采样K条完整轨迹
- 问题：t>1时，每组大小=1（无其他样本共享相同提示）
- 树状采样：每个智能体-轮次位置都有K个候选，确保有效分组

##### 2. Agent和Turn级分组（Agent- and Turn-wise Grouping）
- 基于**acting agent**和**turn number**分组经验
- 使用轻量级哈希函数定义唯一组键：`g ← hash(e, i, t)`
- 确保组内所有候选共享相同的提示（角色+历史）

**公式**：
```
对于环境e中的智能体i在轮次t：
- 采样K个动作：a^(c)_{t,i,e} ~ π_{θ^(σ(i))}(·|o_{t,i,e}; T_samp)
- 计算奖励：r^(c)_{t,i,e}
- 计算优势：{A^(c)_g}^K_{c=1}（使用Eq. 1）
```

##### 3. Agent级信用分配（Agent-wise Credit Assignment）
受合作式多智能体RL的混合奖励设计启发：

**奖励混合公式**：
```
r_{t,i} = α · r^{team}_t + r^{loc}_{t,i}
```

- `r^{team}_t`：全局团队奖励（如代码通过率）
- `r^{loc}_{t,i}`：智能体特定的局部奖励
  - Coder：代码的通过率
  - Tester：参考实现对生成测试的通过率
- `α`：平衡超参数（实验中设为1）

#### 3.1.3 算法流程

**算法1：AT-GRPO训练流程**

```
输入：马尔可夫博弈M，策略集Θ={θ^(m)}^M_{m=1}，
      角色映射σ，采样温度T_samp，分支数K，
      总步数S，批次大小E，轮次范围T

for training step s = 1 to S:
    // 阶段1：在线策略Rollout和数据收集
    初始化每个智能体的数据集 {Di}^N_{i=1} ← ∅
    重采样E个环境

    for each 环境实例 e in {1,...,E} (并行):
        for t = 0 to T-1:
            st,0,e ← st,e  // 初始化微步状态

            for each 智能体 i in {1,...,N}:
                // 树状采样K个候选
                ∀c∈{1,...,K}, a^(c)_{t,i,e} ~ π_{θ^(σ(i))}(·|o_{t,i,e}; T_samp)
                计算 r^(c)_{t,i,e}

                // Agent和Turn级分组
                g ← hash(e, i, t)
                计算优势 {A^(c)_g}^K_{c=1}

                // 添加到智能体数据集
                Di.append((g, o_{t,i,e}, {a^(c)_{t,i,e}}^K_{c=1}, {A^(c)_g}^K_{c=1}))

                // 贪婪选择最佳动作
                c* ← arg max_c r^(c)_{t,i,e}
                a_{t,i,e} ← a^(c*)_{t,i,e}

                // Agent级微转移
                s_{t,i,e} ← T(s_{t,i-1,e}, a_{t,i,e}, i)

            s_{t+1,e} ← s_{t,N,e}  // 轮次结束状态

            if I_term(s_{t+1,e}): break

    // 阶段2：每个模型的策略更新
    for each model m in {1,...,M} (并行):
        构建每个模型的批次 Bm
        使用Eq. 2计算损失 L(θ^(m))
        更新策略m
```

### 3.2 MAS训练系统架构

#### 3.2.1 系统设计原则

解决三大挑战：
1. 训练多个模型的在线策略RL
2. 维护干净的在线训练数据
3. 支持多样化的MAS工作流

#### 3.2.2 架构组件

**1. LLM资源池（GPU）**
- 每个策略管理独立的资源池
- 采用HybridFlow风格，每个池包含两个worker：
  - **RolloutWorker**：用于推理
  - **UpdateWorker**：用于优化
- 在rollout阶段：所有策略根据MAS工作流集体交互
- 收集后：每条轨迹路由到对应的UpdateWorker

**2. 环境执行（CPU）和数据流**
- 环境步骤在CPU EnvWorker队列中运行
- 每个worker管理单个沙箱实例，确保：
  - 安全性
  - 可重现性（seeding、超时、IO配额）
  - 确定性工具harnesses
- 一个actor-一个实例映射，高效支持数千个并发rollouts

**3. 数据路由**
- EnvWorkers将观察、工具日志、基于规则的奖励流式传输到Router
- Router根据策略分配调度收集的经验
- 智能体i生成的经验发送到其指定策略σ(i)的Updateworker

#### 3.2.3 系统架构图解析

```
┌─────────────────────────────────────────────────────────┐
│              LLM Resource Pool 1 (GPU Group i)          │
│  ┌──────────────┐        ┌──────┐      ┌─────────────┐ │
│  │   ROLLOUT    │◄─Serve─┤ LLM  │─────►│   UPDATE    │ │
│  │   WORKER     │        │(Model│Refresh│   WORKER    │ │
│  │              │        │  i)  │Params │             │ │
│  └──────┬───────┘        └──────┘      └──────▲──────┘ │
│         │                                      │         │
└─────────┼──────────────────────────────────────┼─────────┘
          │                                      │
          │                                      │ Model i Batches
          │                                      │
┌─────────┼──────────────────────────────────────┼─────────┐
│         │     Env Resource Pool (CPU)          │         │
│         │                                      │         │
│    Observation                        Trajectory Data   │
│         │                                      │         │
│  ┌──────▼────┐  ┌──────────┐                 │         │
│  │    Env    │  │   Env    │    ───────►ROUTING────────┘
│  │  Worker   │  │  Worker  │              │             │
│  └───────────┘  └──────────┘              │             │
│         ▲              ▲                   │             │
│         │              │                   ▼             │
│         └──────────────┴────Action    Model j Batches   │
│                                            │             │
└────────────────────────────────────────────┼─────────────┘
                                             │
┌────────────────────────────────────────────┼─────────────┐
│              LLM Resource Pool 2 (GPU Group j)           │
│                                            │             │
│  ┌──────────────┐        ┌──────┐      ┌──▼──────────┐ │
│  │   ROLLOUT    │◄─Serve─┤ LLM  │─────►│   UPDATE    │ │
│  │   WORKER     │        │(Model│Refresh│   WORKER    │ │
│  │              │        │  j)  │Params │             │ │
│  └──────────────┘        └──────┘      └─────────────┘ │
└─────────────────────────────────────────────────────────┘

          ┌───────────────────────┐
          │        MAS            │
          │   ┌────┐  ┌────┐     │
          │   │Ag 1│──│Ag 2│ ··· │
          │   └────┘  └────┘     │
          └───────────────────────┘
```

---

## 四、架构设计详解

### 4.1 整体架构


### 4.2 核心模块设计

#### 4.2.1 Multi-Agent Environment（多智能体环境）

**设计理念**：
- 提供统一的环境接口，支持不同领域任务
- 实现细粒度的奖励计算（team + local）
- 支持多轮交互和早停机制

**关键特性**：
```python
# 基础环境接口（推断）
class MultiAgentEnv:
    def reset(self, task):
        """初始化环境和任务"""

    def step(self, agent_id, action):
        """执行智能体动作，返回下一状态、奖励、done标志"""

    def get_observation(self, agent_id):
        """获取智能体特定的观察"""

    def compute_reward(self, agent_id, action):
        """计算混合奖励: α*r_team + r_local"""
```

**具体实现**：

1. **代码环境（Code）**：
   - Coder智能体：生成代码
   - Tester智能体：生成单元测试
   - 环境：执行代码和测试，返回通过率
   - 终止条件：所有测试通过或达到最大轮次

2. **数学环境（Math）**：
   - Reasoner智能体：推理求解
   - Tool-User智能体：使用代码解释器
   - 环境：验证数值答案
   - 终止条件：两个智能体答案一致或达到最大轮次

3. **游戏/规划环境（Stateful）**：
   - Tool-User智能体：执行工具（BFS、A*搜索）
   - Executor智能体：验证工具输出并执行动作
   - 环境：网格世界模拟器
   - 终止条件：达成目标或达到最大轮次

#### 4.2.2 MAS Graph（工作流图）

**设计理念**：
- 支持灵活的智能体交互模式
- 三种设计范式：
  - **Graph-based**：灵活拓扑，集成AutoGen、LangChain
  - **Turn-based**：有限状态机，精确操作顺序
  - **AFlow Co-Evolve**：自动化设计MAS结构

**工作流示例**：

```python
# 代码任务工作流（推断）
class CodeMASGraph:
    def __init__(self):
        self.agents = {
            'coder': CoderAgent(),
            'tester': TesterAgent()
        }
        self.max_turns = 4

    def run_episode(self, problem):
        for turn in range(self.max_turns):
            # 并行生成
            code = self.agents['coder'].generate(problem, history)
            tests = self.agents['tester'].generate(problem, history)

            # 环境验证
            results = self.env.execute(code, tests)

            # 检查对齐
            if results.all_pass:
                return True, turn

            # 更新历史
            history.append((code, tests, results))

        return False, self.max_turns
```

#### 4.2.3 AT-GRPO Trainer（训练器）

**核心实现**：

```python
# 简化的AT-GRPO训练器伪代码
class ATGRPOTrainer:
    def __init__(self, models, mas_graph, env):
        self.models = models  # {agent_id: model}
        self.mas_graph = mas_graph
        self.env = env
        self.K = 4  # 采样分支数

    def train_step(self):
        # 阶段1：Rollout收集
        datasets = {agent_id: [] for agent_id in self.models}

        for env_instance in parallel_envs:
            for turn in range(max_turns):
                for agent_id in self.mas_graph.get_agents():
                    obs = env_instance.get_observation(agent_id)

                    # 树状采样K个候选
                    candidates = []
                    for k in range(self.K):
                        action = self.models[agent_id].sample(obs)
                        reward = env_instance.compute_reward(agent_id, action)
                        candidates.append((action, reward))

                    # 计算优势
                    group_key = hash((env_instance.id, agent_id, turn))
                    advantages = self.compute_advantages(candidates)

                    # 存储到智能体数据集
                    datasets[agent_id].append({
                        'group': group_key,
                        'observation': obs,
                        'candidates': candidates,
                        'advantages': advantages
                    })

                    # 贪婪选择执行
                    best_action = max(candidates, key=lambda x: x[1])[0]
                    env_instance.step(agent_id, best_action)

                if env_instance.is_done():
                    break

        # 阶段2：策略更新
        for agent_id, model in self.models.items():
            batch = datasets[agent_id]
            loss = self.compute_ppo_loss(model, batch)
            model.update(loss)

    def compute_advantages(self, candidates):
        """Agent和Turn级分组的优势计算"""
        rewards = [c[1] for c in candidates]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8

        advantages = [(r - mean_reward) / std_reward for r in rewards]
        return advantages
```

### 4.3 支持的三种策略配置

#### Level 1: Shared Policy（共享策略）
```
所有智能体 → 同一个基础模型 + 不同的系统提示
- 参数共享，角色通过提示区分
- 训练数据：合并所有智能体的轨迹
- 更新：单一模型联合更新
```

#### Level 2: Agent-specific LoRA（智能体特定LoRA）
```
所有智能体 → 同一个基础模型 + 角色特定的LoRA适配器
- 基础模型共享，轻量级角色专业化
- 训练数据：每个智能体独立轨迹
- 更新：每个LoRA独立更新
```

#### Level 3: Agent-specific Model（智能体特定模型）
```
每个智能体 → 独立的模型实例
- 完全独立的模型参数
- 训练数据：每个智能体独立轨迹
- 更新：每个模型完全独立更新
```

### 4.4 奖励设计架构

**混合奖励公式**：
```
r_{t,i} = α · r^{team}_t + r^{loc}_{t,i}
```

**具体实现**（以代码任务为例）：

```python
# 团队奖励（全局）
def compute_team_reward(code, golden_tests):
    """
    所有智能体共享相同的团队奖励
    """
    pass_rate = run_tests(code, golden_tests)
    return pass_rate  # [0, 1]

# Coder的局部奖励
def compute_coder_local_reward(code, golden_tests):
    """
    组合：
    - 0.1 * build_success（编译成功）
    - 0.1 * run_success（smoke测试）
    - 0.8 * pass_rate（golden测试通过率）
    """
    build_ok = check_syntax(code)
    run_ok = check_smoke_tests(code, golden_tests[:3])
    pass_rate = run_tests(code, golden_tests)

    return 0.1*build_ok + 0.1*run_ok + 0.8*pass_rate

# Tester的局部奖励
def compute_tester_local_reward(tests, golden_code):
    """
    组合：
    - 0.2 * valid（测试可执行）
    - 0.8 * pass_rate（golden代码通过率）
    """
    valid = check_tests_valid(tests)
    pass_rate = run_tests(golden_code, tests)

    return 0.2*valid + 0.8*pass_rate
```

**数学任务奖励**：

```python
# 团队奖励：数值匹配
def compute_team_reward(final_answer, ground_truth):
    return numeric_equals(final_answer, ground_truth, tol=1e-6)

# Reasoner局部奖励
def compute_reasoner_local_reward(reasoning_output, ground_truth):
    """
    - 0.2 * format_valid（格式匹配）
    - 0.8 * numeric_match（数值匹配）
    """

# Tool-User局部奖励
def compute_tool_local_reward(code_output, ground_truth):
    """
    - 0.1 * build_ok
    - 0.1 * exec_ok
    - 0.8 * numeric_match
    """
```

### 4.5 系统复杂度分析

#### 推理时间复杂度

**Sequential MAS**（如游戏/规划）：
```
Time_infer^Seq / Time_infer^SA ≤ N·T / T = N
```
- 延迟开销与智能体数量N线性相关

**Parallel MAS**（如代码/数学）：
```
当 N ≤ N_max = ⌊B_max / (E·K)⌋ 时：
Time_infer^Para / Time_infer^SA ≲ T

当 N > N_max 时：
Time_infer^Para / Time_infer^SA ≲ T · ⌈N/N_max⌉
```
- B_max：集群可服务的最大并发序列数
- E：并行环境数
- K：每个智能体的采样因子

#### 训练时间复杂度

```
Time_train^MAS / Time_train^SA ≤ |D_MAS| / |D_SA| = NT
```
- Agent和Turn级分组只引入轻量级哈希开销O(|D_MAS|)
- 主要开销来自token级模型执行O(|D_MAS|·L·C_model)

**实证延迟研究**（4×H100，batch=32×8）：
- **代码任务（单智能体N=1, T=1）**：
  - Rollout: ~4分钟（占80%）
  - Training: ~1分钟
- **代码任务（MAS N=2，多轮）**：
  - Rollout: ~8分钟
  - Training: ~2分钟
- **游戏任务**：
  - 单智能体rollout: 2.8分钟
  - MAS rollout: 1.5分钟（因MAS早停效果好）

---

## 五、实验结果

### 5.1 实验设置

**模型**：Qwen3（1.7B和8B，无thinking模式）
**硬件**：单节点8×H100 GPU
**超参数**：
- Rollout采样大小K=4
- 轮次范围T=4
- 奖励混合系数α=1
- 全局批次大小128，PPO mini-batch 64

**任务和数据集**：

1. **游戏和规划**
   - Sudoku（4×4）、Sokoban（6×6）
   - Plan-Path（10×10网格）

2. **代码生成**
   - 训练：APPS（1.7B），CodeContests（8B）
   - 评估：APPS、LiveCodeBench-v6、CodeContests

3. **数学推理**
   - 训练：Polaris-Dataset-53K
   - 评估：AIME24/25、OlympiadBench

### 5.2 主要结果

3. **实证层面**：在长期规划任务上实现突破性提升（14-47% → 96-99.5%），证明了MAS RL的巨大潜力

4. **开源贡献**：PettingLLMs框架为社区提供了完整的MAS RL训练工具链，支持多领域、多模态扩展

该工作为未来的LLM智能体协作研究奠定了坚实基础，特别是在需要复杂协调和角色专业化的任务场景中具有重要应用价值。
