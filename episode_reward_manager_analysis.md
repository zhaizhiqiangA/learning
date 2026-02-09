# EpisodeRewardManager Analysis

## Overview

`EpisodeRewardManager` is a specialized reward manager for agent-based reinforcement learning that assigns episode-level rewards to the final timestep of multi-turn trajectories. It processes episodic returns collected during environment interactions and optionally normalizes them by episode length.

## Architecture

### Component Hierarchy

```
Reward Manager Ecosystem
├── verl/workers/reward_manager/  (Base RL reward managers)
│   ├── NaiveRewardManager       (Step-level scoring with ground truth)
│   ├── BatchRewardManager        (Batch processing optimization)
│   ├── DAPORewardManager         (DAPO algorithm specific)
│   └── PrimeRewardManager        (PRIME algorithm specific)
└── agent_system/reward_manager/  (Agent-specific reward managers)
    └── EpisodeRewardManager      (Episode-level rewards for agents)
```

### Integration Points

**Data Flow:**
```
TrajectoryCollector (rollout_loop.py)
    ↓ collects episode_rewards/episode_lengths
DataProto (protocol.py)
    ↓ packages data with non_tensor_batch metadata
EpisodeRewardManager (episode.py)
    ↓ extracts and assigns rewards
reward_tensor
    ↓ used by PPO/GiGPO training
Policy Updates
```

### Key Differences from NaiveRewardManager

| Aspect | NaiveRewardManager | EpisodeRewardManager |
|--------|-------------------|---------------------|
| **Reward Source** | Computed via `compute_score()` function with ground truth | Pre-computed during environment rollout |
| **Metadata Keys** | `reward_model.ground_truth` | `episode_rewards`, `episode_lengths` |
| **Use Case** | Single-turn QA, math, code tasks | Multi-turn agent tasks (ALFWorld, WebShop) |
| **Tokenization** | `skip_special_tokens=True` | `skip_special_tokens=False` |
| **Extra Info** | Rich dict with scoring details | Minimal (empty dict) |

## Core Principles

### 1. Episode-Level Credit Assignment

**Why**: In multi-turn agent tasks, only the final outcome (success/failure) determines the reward. Individual steps don't receive immediate feedback.

**Implementation**:
- `rollout_loop.py:329` - Initialize accumulators: `episode_rewards = np.zeros(batch_size, dtype=np.float32)`
- `rollout_loop.py:383` - Accumulate step rewards: `episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]`
- `rollout_loop.py:268-270` - Store in metadata:
  ```python
  data['episode_rewards'] = episode_rewards[bs]
  data['episode_lengths'] = episode_lengths[bs]
  ```
- `episode.py:72-73` - Extract from non-tensor batch:
  ```python
  episode_rewards = data_item.non_tensor_batch['episode_rewards']
  episode_lengths = data_item.non_tensor_batch['episode_lengths']
  ```

**Result**: Entire episode trajectory gets a single reward signal at the final timestep, enabling proper credit assignment through advantage estimation.

### 2. Reward-at-Terminal Design

**Why**: Reinforcement learning algorithms (PPO, GiGPO) need rewards aligned with terminal states for value function bootstrapping and advantage calculation.

**Implementation** (lines 79):
```python
reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)
```

**Key details**:
- Only the final token position (`valid_response_length - 1`) receives the reward
- All other positions remain zero (from `torch.zeros_like()` initialization at line 39)
- Device placement matches input tensors for efficient GPU computation

**Result**: Sparse reward signal that aligns with RL theory while maintaining computational efficiency.

### 3. Optional Length Normalization

**Why**: Longer episodes may accumulate more step rewards by chance. Normalization provides fairer comparison across episodes of varying lengths.

**Implementation** (lines 75-78):
```python
if self.normalize_by_length:
    score = episode_rewards / episode_lengths
else:
    score = episode_rewards
```

**Configuration** (main_ppo.py:152, 155):
```python
reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
```

**Trade-off**:
- **Without normalization**: Favors longer successful episodes (good for exploration)
- **With normalization**: Favors efficiency (good for minimizing steps)

### 4. Pre-Computed Reward Bypass

**Why**: Flexibility to use external reward models or pre-scored data without recomputation.

**Implementation** (lines 33-37):
```python
if "rm_scores" in data.batch.keys():
    if return_dict:
        return {"reward_tensor": data.batch["rm_scores"]}
    else:
        return data.batch["rm_scores"]
```

**Usage**: Allows integration with learned reward models or human annotations stored in the dataset.

## Implementation Details

### Class Definition

**Purpose**: Process episodic rewards for multi-turn agent trajectories

**Initialization** (lines 24-27):
```python
def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
    self.tokenizer = tokenizer
    self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
    self.normalize_by_length = normalize_by_length
```

**Parameters**:
- `tokenizer`: Required for decoding prompt/response IDs for debugging
- `num_examine`: Controls console logging frequency (0 for training, 1+ for validation)
- `normalize_by_length`: Boolean flag for episode length normalization

### Main Processing Method: `__call__`

**Purpose**: Convert episode metadata into reward tensors aligned with model outputs

**Signature** (line 29):
```python
def __call__(self, data: DataProto, return_dict=False):
```

**Algorithm Flow**:

1. **Check for pre-computed rewards** (lines 33-37)
2. **Initialize reward tensor** (line 39):
   ```python
   reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
   ```
3. **Process each trajectory** (lines 43-89):
   - Extract prompt/response with attention masking
   - Decode for logging
   - Retrieve episode metadata
   - Compute normalized/unnormalized score
   - Assign to final timestep
   - Optionally print samples

4. **Return formatted output** (lines 90-96)

### Attention Mask Handling

**Purpose**: Extract only valid (non-padded) tokens for accurate decoding

**Implementation** (lines 46-59):
```python
prompt_ids = data_item.batch['prompts']
prompt_length = prompt_ids.shape[-1]

valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
valid_prompt_ids = prompt_ids[-valid_prompt_length:]

response_ids = data_item.batch['responses']
valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
valid_response_ids = response_ids[:valid_response_length]

# decode
prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
```

**Key details**:
- Splits attention mask between prompt and response regions
- Uses `.sum()` to count valid tokens (1s in attention mask)
- Preserves special tokens (`skip_special_tokens=False`) for exact reproduction
- Right-aligns prompt extraction (`prompt_ids[-valid_prompt_length:]`)
- Left-aligns response extraction (`response_ids[:valid_response_length]`)

### Debugging and Logging

**Purpose**: Provide visibility into reward assignment during training/validation

**Implementation** (lines 81-88):
```python
if data_source not in already_print_data_sources:
    already_print_data_sources[data_source] = 0

if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
    already_print_data_sources[data_source] += 1
    print(f"[{data_source}][prompt]", prompt_str)
    print(f"[{data_source}][response]", response_str)
    print(f"[{data_source}][score]", score)
```

**Key details**:
- Tracks printing per `data_source` to avoid spam
- 10% random sampling (`np.random.random() < 0.1`) for variety
- Respects `num_examine` limit per data source
- Includes data source prefix for multi-domain training

### Multi-Modal Support Hooks

**Purpose**: Future-proofing for vision-language agent tasks

**Implementation** (lines 65-69):
```python
extra_info = data_item.non_tensor_batch.get('extra_info', None)
multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
if multi_modal_inputs is not None:
    pixel_values = multi_modal_inputs['pixel_values']
    image_grid_thw = multi_modal_inputs['image_grid_thw']
```

**Status**: Currently unused (variables extracted but not processed), prepared for future extensions.

## Integration

### Usage in Training Pipeline

**Instantiation** (main_ppo.py:145-155):
```python
reward_manager_name = config.reward_model.get("reward_manager", "episode")
if reward_manager_name == 'episode':
    from agent_system.reward_manager import EpisodeRewardManager
    reward_manager_cls = EpisodeRewardManager
else:
    raise NotImplementedError

reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

# Note that we always use function-based RM for validation
val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
```

**Configuration Key**: `config.reward_model.reward_manager` (defaults to `"episode"`)

### Data Flow from Trajectory Collection

**Source** (rollout_loop.py:328-330, 383-384):
```python
episode_lengths = np.zeros(batch_size, dtype=np.float32)
episode_rewards = np.zeros(batch_size, dtype=np.float32)
...
# Inside loop for each step:
episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
episode_lengths[active_masks] += 1
```

**Packaging** (rollout_loop.py:268-270):
```python
data['episode_rewards'] = episode_rewards[bs]
data['episode_lengths'] = episode_lengths[bs]
```

**Protocol**: Data transferred via `DataProto.non_tensor_batch` dictionary, preserving numpy arrays through the pipeline.

### Dynamic Sampling Compatibility

**Purpose**: DAPO/Dynamic GiGPO filter out episode groups with identical rewards

**Integration** (utils.py:133-148):
```python
def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
```

**Key detail**: Episode rewards are used BEFORE entering the reward manager for filtering, then re-packaged for reward tensor construction.

## Trade-offs

### Performance

**Memory**:
- ✅ Efficient: Single reward value per episode (not per step)
- ✅ Zero-copy: Uses `torch.zeros_like()` for initialization
- ✅ Device-aware: Tensor creation respects input device placement

**Computation**:
- ✅ Minimal: No reward function calls (pre-computed during rollout)
- ⚠️ Decoding overhead: Tokenizer decode for every sample (only for logging)
- Optimization: Could batch decode or skip entirely when `num_examine=0`

### Flexibility

**Strengths**:
- ✅ Pre-computed reward bypass (`rm_scores` check)
- ✅ Configurable length normalization
- ✅ Multi-modal hooks for future extension

**Limitations**:
- ❌ No support for step-level rewards (by design)
- ❌ No intermediate feedback for partial success
- ❌ Tied to episode completion (can't handle infinite-horizon tasks)

### Complexity

**Simplicity**:
- ✅ Clear separation: Reward computation in environment, assignment in manager
- ✅ Minimal logic: ~50 lines of core functionality
- ✅ Single responsibility: Just reward tensor construction

**Debugging**:
- ✅ Controlled logging with `num_examine`
- ✅ Per-source tracking prevents spam
- ⚠️ Random sampling (10%) means not all examples logged

### Algorithm Compatibility

**Works with**:
- ✅ PPO (episode-level value functions)
- ✅ GiGPO (episode grouping + step-level advantages)
- ✅ GRPO (group-based preference optimization)
- ✅ DAPO (dynamic sampling of diverse episode groups)

**Assumptions**:
- Requires complete episodes (no partial trajectories)
- Expects `episode_rewards`/`episode_lengths` in metadata
- Assumes terminal reward structure

## File Locations

### Main Implementation
- **Primary**: `/data1/zzq/rl-proj/verl-agent/agent_system/reward_manager/episode.py`
- **Init**: `/data1/zzq/rl-proj/verl-agent/agent_system/reward_manager/__init__.py`

### Related Files
- **Base reward managers**: `/data1/zzq/rl-proj/verl-agent/verl/workers/reward_manager/naive.py`
- **Data protocol**: `/data1/zzq/rl-proj/verl-agent/verl/protocol.py` (DataProto, DataProtoItem)
- **Trajectory collection**: `/data1/zzq/rl-proj/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py`
- **Dynamic sampling**: `/data1/zzq/rl-proj/verl-agent/agent_system/multi_turn_rollout/utils.py`
- **Training integration**: `/data1/zzq/rl-proj/verl-agent/verl/trainer/main_ppo.py`

## Design Rationale

### Why Episode-Level Instead of Step-Level?

**Agent tasks are inherently sparse-reward**:
- ALFWorld: Success only when goal achieved
- WebShop: Reward only for final purchase decision
- Sokoban: Only complete solutions get positive reward

**Step-level rewards would require**:
- Dense reward shaping (hard to design, can cause unintended behaviors)
- Intermediate state evaluation (expensive for complex tasks)
- Risk of reward hacking (optimizing shaped rewards instead of task)

**Episode-level approach**:
- Mirrors natural task structure
- Simpler, more robust
- Leverages RL algorithms' credit assignment (GAE, value functions)

### Why Store in non_tensor_batch?

**Scalar vs. Sequence Data**:
- `episode_rewards`: Single float per trajectory
- `batch['responses']`: Tensor of shape `[batch_size, seq_len]`

**Storage efficiency**:
- `non_tensor_batch` holds Python objects (numpy arrays, dicts)
- No need to broadcast scalar to sequence length
- Easier to manipulate (filtering, grouping) before reward assignment

**Protocol compliance**:
- `DataProto` designed for mixed tensor/non-tensor data
- Follows existing patterns in verl ecosystem

### Why Assign Only at Final Timestep?

**RL Algorithm Requirements**:
- PPO/GiGPO compute advantages: `A_t = δ_t + γλδ_{t+1} + ...`
- Terminal rewards simplify: `δ_T = R_T - V(s_T)` (no bootstrapping)
- Intermediate zeros don't affect advantage calculation (absorbed by value function)

**Computational efficiency**:
- Sparse reward tensor (mostly zeros)
- Gradient computation focuses on relevant timesteps
- Aligns with typical language model loss masks (only predict response tokens)

## Extension Points

### Adding New Normalization Strategies

**Current**: Binary choice (normalize by length or not)

**Extension example** (lines 75-78):
```python
if self.normalize_by_length:
    score = episode_rewards / episode_lengths
elif self.normalize_by_variance:
    # Normalize by episode reward variance for stability
    score = (episode_rewards - self.reward_mean) / (self.reward_std + 1e-8)
else:
    score = episode_rewards
```

**Configuration**: Add to `config.reward_model.normalization_type`

### Multi-Modal Reward Integration

**Current**: Hooks exist but unused (lines 65-69)

**Extension example**:
```python
if multi_modal_inputs is not None:
    pixel_values = multi_modal_inputs['pixel_values']
    # Use vision encoder to assess visual goal achievement
    visual_success_score = self.vision_reward_model(pixel_values)
    score = episode_rewards * visual_success_score
```

**Use case**: Vision-language agent tasks (visual goal verification)

### Hybrid Step+Episode Rewards

**Current**: Pure episode-level

**Extension example**:
```python
if 'step_rewards' in data_item.non_tensor_batch:
    step_rewards_array = data_item.non_tensor_batch['step_rewards']
    # Assign step rewards to corresponding positions
    for step_idx, step_reward in enumerate(step_rewards_array):
        if step_idx < valid_response_length:
            reward_tensor[i, step_idx] = step_reward
    # Add episode reward to final step
    reward_tensor[i, valid_response_length - 1] += episode_rewards
```

**Use case**: Tasks with both intermediate milestones and final goals

## Comparison with Alternatives

### vs. NaiveRewardManager

**When to use EpisodeRewardManager**:
- Multi-turn agent tasks with environment interaction
- Pre-computed rewards from environment simulation
- Need length normalization option

**When to use NaiveRewardManager**:
- Single-turn generation tasks (math, code, QA)
- Rewards computed from ground truth comparison
- Rich extra info needed (correctness, reasoning quality, etc.)

### vs. DAPORewardManager

**Similarity**: Both handle episode-level data

**Difference**: DAPO includes advantage/group processing logic specific to DAPO algorithm

**When to use EpisodeRewardManager**: Standard PPO/GiGPO training

**When to use DAPORewardManager**: Specifically training with DAPO algorithm

### vs. Custom Reward Functions

**Advantage of EpisodeRewardManager**:
- Standardized interface
- Built-in logging
- Length normalization option
- Multi-modal hooks

**When to build custom**:
- Highly specialized reward computation
- Non-standard tensor structures
- Integration with external reward models requiring complex preprocessing
