# GiGPO Algorithm: In-Depth Analysis

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Algorithm Principles](#algorithm-principles)
4. [Code Implementation Analysis](#code-implementation-analysis)
5. [Training Pipeline](#training-pipeline)
6. [Key Design Decisions](#key-design-decisions)
7. [Performance Characteristics](#performance-characteristics)

---

## Overview

**GiGPO (Group-in-Group Policy Optimization)** is a novel reinforcement learning algorithm specifically designed for training LLM agents in long-horizon, multi-turn environments. Published at NeurIPS 2025, it addresses the critical challenge of credit assignment in agent tasks where episodes can span 30-50 steps.

**Paper**: [https://arxiv.org/abs/2505.10978](https://arxiv.org/abs/2505.10978)

**Key Innovation**: Two-level grouping mechanism that provides both episode-level and step-level advantage estimates without requiring a critic network.

### Core Features

- **Critic-free**: Same memory footprint as GRPO but with better credit assignment
- **Two-level grouping**: Episode-level groups (total returns) + step-level groups (relative advantages)
- **Scalable**: Designed for very long-horizon tasks (30-50+ steps)
- **Step-independent rollout**: Each step constructs concise, customizable inputs

---

## Core Concepts

### 1. Episode-Level Grouping

**Purpose**: Normalize rewards across different rollouts of the same initial state to estimate overall trajectory quality.

**Mechanism**:
- Multiple rollouts share the same initial state (controlled by `env.rollout.n`)
- Each rollout produces a total episode return
- Advantages computed by comparing returns within the same episode group

**Code Location**: `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py` - `episode_norm_reward()`

### 2. Step-Level Grouping

**Purpose**: Provide fine-grained credit assignment by grouping steps with identical or similar observations across different trajectories.

**Mechanism**:
- Observations at each step are clustered based on equality or similarity
- Steps reaching the same state form a "step group"
- Advantages computed by comparing step-level rewards within each step group

**Code Location**: `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py` - `build_step_group()`

### 3. Anchor Observations

**Definition**: A canonical representation of the environment state at each step, stripped of history or additional context.

**Purpose**: Enable accurate step-level grouping by identifying when different trajectories reach the same state.

**Example** (from ALFWorld):
- Full observation might include task description, history, current state
- Anchor observation contains only the current room state description

**Code Location**: Stored in `batch.non_tensor_batch['anchor_obs']` during rollout

---

## Algorithm Principles

### Mathematical Formulation

#### Episode-Level Advantage (Equation 3 in paper)

For a response \( r_i \) in episode group \( G_e \):

```
A_episode(r_i) = R_i - mean(R_j : j ∈ G_e)
```

Where:
- \( R_i \) = Total discounted return for trajectory \( i \)
- \( G_e \) = Set of trajectories starting from the same initial state
- Mean normalization removes baseline

**Variants**:
- `mode="mean_norm"`: Only mean subtraction (default, more stable)
- `mode="mean_std_norm"`: Mean subtraction + std division

#### Step-Level Advantage (Equation 7 in paper)

For a step \( s_i \) in step group \( G_s \):

```
A_step(s_i) = r_i - mean(r_j : j ∈ G_s)
```

Where:
- \( r_i \) = Immediate step reward
- \( G_s \) = Set of steps with identical/similar anchor observations

#### Joint Advantage (Equation 8 in paper)

```
A_total = A_episode + λ * A_step
```

Where \( λ \) (`step_advantage_w`) controls the balance between episode and step-level credit assignment.

### Discounted Returns (Equation 5 in paper)

```python
# Backward computation from episode end
for t in reversed(range(T)):
    G_t = r_t + γ * G_{t+1}
```

This provides proper temporal credit assignment within each trajectory.

---

## Code Implementation Analysis

### 1. Core Entry Point: `compute_gigpo_outcome_advantage()`

**File**: `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py` (Lines 138-171)

```python
def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   ):
```

**Parameters**:
- `token_level_rewards`: Total episode rewards broadcast to token level (bs, seq_len)
- `step_rewards`: Immediate rewards for each step (bs,)
- `response_mask`: Indicates valid response tokens (bs, seq_len)
- `anchor_obs`: Canonical state representations for step grouping (bs,)
- `index`: Episode group UIDs (bs,)
- `traj_index`: Trajectory UIDs within episode groups (bs,)
- `step_advantage_w`: Weight for step-level advantages (default: 1.0)
- `mode`: Normalization mode ("mean_norm" or "mean_std_norm")
- `enable_similarity`: Use similarity-based step grouping (default: False)
- `similarity_thresh`: Similarity threshold when enabled (default: 0.95)

**Algorithm Flow**:
```python
# Step 1: Compute episode-level advantages
episode_advantages = episode_norm_reward(
    token_level_rewards, response_mask, index, traj_index, epsilon, remove_std
)

# Step 2: Build step groups based on anchor observations
step_group_uids = build_step_group(
    anchor_obs, index, enable_similarity, similarity_thresh
)

# Step 3: Compute step-level advantages
step_advantages = step_norm_reward(
    step_rewards, response_mask, step_group_uids, epsilon, remove_std
)

# Step 4: Combine advantages
scores = episode_advantages + step_advantage_w * step_advantages
return scores, scores
```

### 2. Episode-Level Normalization: `episode_norm_reward()`

**File**: `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py` (Lines 174-240)

**Key Implementation Details**:

```python
# Compute total returns per trajectory
scores = token_level_rewards.sum(dim=-1)  # (bs,)

# Group by episode index
id2score = defaultdict(list)
for i in range(bsz):
    if (index[i], traj_index[i]) in seen_pairs:
        continue
    id2score[index[i]].append(scores[i])
    if not compute_mean_std_cross_steps:
        seen_pairs.add((index[i], traj_index[i]))

# Compute mean/std per group
for idx in id2score:
    if len(id2score[idx]) == 1:
        id2mean[idx] = torch.tensor(0.0)  # No normalization for single sample
        id2std[idx] = torch.tensor(1.0)
    elif len(id2score[idx]) > 1:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))

# Apply normalization
for i in range(bsz):
    if remove_std:
        scores[i] = scores[i] - id2mean[index[i]]
    else:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

# Broadcast to token level
episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
```

**Design Notes**:
- Single-sample groups get zero mean (no comparison available)
- `compute_mean_std_cross_steps=True` (default) computes statistics across all steps in an episode group, providing more stable estimates
- Response mask ensures advantages only applied to valid tokens

### 3. Step-Level Grouping: `build_step_group()`

**File**: `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py` (Lines 243-331)

**Implementation Strategy**:

```python
# Initialize result array
step_group_uids = np.empty(len(anchor_obs), dtype=object)

# Process each episode group separately
unique_indices = np.unique(index)

for idx in unique_indices:
    indices = np.where(index == idx)[0]
    obs_group = anchor_obs[indices]

    if not enable_similarity:
        # Exact matching via hashable conversion
        clusters = defaultdict(list)
        for i, obs in enumerate(obs_group):
            clusters[to_hashable(obs)].append(indices[i])

        # Assign UUID to each cluster
        for obs, original_indices in clusters.items():
            uid = str(uuid.uuid4())
            for original_idx in original_indices:
                step_group_uids[original_idx] = uid
    else:
        # Similarity-based clustering
        clusters: List[Dict[str, Any]] = []

        for obs, loc in zip(obs_group, locs):
            placed = False
            for cluster in clusters:
                if are_similar(obs, cluster["rep"], similarity_thresh):
                    cluster["locs"].append(loc)
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": obs, "locs": [loc]})

        # Assign UUIDs
        for cluster in clusters:
            uid = str(uuid.uuid4())
            for loc in cluster["locs"]:
                step_group_uids[loc] = uid
```

**Helper Function: `to_hashable()`** (Lines 34-47)

Converts observations to hashable types for exact matching:

```python
def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
```

**Similarity Function: `are_similar()`** (Lines 72-85)

```python
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """Check whether two text observations are similar enough using SequenceMatcher."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations supported for similarity-based GiGPO")
    return SequenceMatcher(None, a, b).ratio() >= threshold
```

**Design Notes**:
- Episode groups processed independently (step groups don't cross episode boundaries)
- UUID-based identification ensures uniqueness
- Similarity-based grouping uses greedy first-match strategy
- Group size statistics printed for monitoring

### 4. Step-Level Normalization: `step_norm_reward()`

**File**: `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py` (Lines 334-384)

**Implementation**:

```python
scores = step_rewards.clone()  # (bs,)

id2score = defaultdict(list)
id2mean = {}
id2std = {}

# Group rewards by step group UID
for i in range(bsz):
    id2score[index[i]].append(scores[i])

# Compute statistics per step group
for idx in id2score:
    if len(id2score[idx]) == 1:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.tensor(1.0)
    elif len(id2score[idx]) > 1:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))

# Apply normalization
for i in range(bsz):
    if remove_std:
        scores[i] = scores[i] - id2mean[index[i]]
    else:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

# Broadcast to token level
step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
```

**Difference from Episode Normalization**:
- Uses `step_group_uids` instead of episode `index`
- No trajectory deduplication (each step is unique)
- Same broadcasting strategy to token level

### 5. Discounted Returns: `compute_step_discounted_returns()`

**File**: `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py` (Lines 87-132)

**Implementation**:

```python
def compute_step_discounted_returns(batch: DataProto, gamma: float):
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)

    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)

    for uid in unique_traj_uids:
        traj_indices = np.where(traj_uids == uid)[0]
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]

        # Backward computation
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0

        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return

        returns_by_traj[uid] = traj_returns

    # Recombine into original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]
        all_returns[i] = returns_by_traj[uid][idx_in_traj]

    all_returns = torch.tensor(all_returns, dtype=torch.float32,
                               device=batch.batch['input_ids'].device)
    return all_returns
```

**Design Notes**:
- Processes each trajectory independently
- Standard backward RL return computation
- Preserves original batch ordering for efficient training
- Active masks ensure proper handling of terminated episodes

---

## Training Pipeline

### 1. Multi-Turn Rollout Collection

**File**: `/data1/zzq/rl-proj/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py`

**Class**: `TrajectoryCollector`

#### Vanilla Rollout Loop (Lines 285-414)

```python
def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
    batch_size = len(gen_batch.batch)

    # Initialize environment
    obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

    # Assign trajectory UIDs and episode group UIDs
    if self.config.env.rollout.n > 0:  # env grouping enabled
        uid_batch = []
        for i in range(batch_size):
            if i % self.config.env.rollout.n == 0:
                uid = str(uuid.uuid4())  # New group every n samples
            uid_batch.append(uid)
        uid_batch = np.array(uid_batch, dtype=object)
    else:
        uid = str(uuid.uuid4())
        uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)

    traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

    # Main interaction loop
    for _step in range(self.config.env.max_steps):
        active_masks = np.logical_not(is_done)

        # 1. Preprocess observations
        batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

        # 2. Generate actions
        batch_output = actor_rollout_wg.generate_sequences(batch_input)

        # 3. Execute in environment
        text_actions = self.tokenizer.batch_decode(batch.batch['responses'],
                                                    skip_special_tokens=True)
        next_obs, rewards, dones, infos = envs.step(text_actions)

        # 4. Store trajectory data
        batch.non_tensor_batch['uid'] = uid_batch
        batch.non_tensor_batch['traj_uid'] = traj_uid
        batch.non_tensor_batch['rewards'] = rewards
        batch.non_tensor_batch['active_masks'] = active_masks

        episode_rewards[active_masks] += rewards[active_masks]
        episode_lengths[active_masks] += 1

        # 5. Append to trajectory buffer
        batch_list = to_list_of_dict(batch)
        for i in range(batch_size):
            total_batch_list[i].append(batch_list[i])

        # Update state
        is_done = np.logical_or(is_done, dones)
        obs = next_obs

        if is_done.all():
            break

    return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
```

**Key Aspects**:
- **Episode Grouping**: `env.rollout.n` consecutive environments share the same `uid`
- **Trajectory IDs**: Each environment gets unique `traj_uid` for tracking
- **Active Masks**: Track which environments are still running
- **Step-by-step construction**: Each step processes only current observation + memory

#### Dynamic Rollout Loop (Lines 416-482)

Extension for DAPO-style dynamic sampling:

```python
def dynamic_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
    total_batch_list = []
    total_episode_rewards = []
    # ... other accumulators

    try_count = 0
    max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

    # Keep sampling until target batch size met
    while len(total_batch_list) < target_size and try_count < max_try_count:
        try_count += 1

        # Run vanilla loop
        batch_list, episode_rewards, ... = self.vanilla_multi_turn_loop(...)

        # Filter out groups with identical rewards
        batch_list, episode_rewards, ... = filter_group_data(
            batch_list, episode_rewards, ..., config, last_try=(try_count == max_try_count)
        )

        # Accumulate valid groups
        total_batch_list += batch_list
        total_episode_rewards.append(episode_rewards)
        # ...

    return concatenated_results
```

**Filter Logic** (from `/data1/zzq/rl-proj/verl-agent/agent_system/multi_turn_rollout/utils.py`, Lines 133-184):

```python
def filter_group_data(batch_list, episode_rewards, ..., config, last_try=False):
    """Remove episode groups where all trajectories have identical rewards."""
    if last_try:
        return batch_list, ...  # Keep everything on final attempt

    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # Only keep groups with reward variance
        if not np.all(group_rewards == group_rewards[0]):
            keep_indices = np.concatenate((keep_indices, group_indices))

    # Filter all data structures
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    # ...
    return batch_list, episode_rewards, ...
```

### 2. Observation Preprocessing

**File**: `/data1/zzq/rl-proj/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py` (Lines 43-188)

```python
def preprocess_single_sample(self, item, gen_batch, obs):
    """Process single observation into model input."""

    # Extract observation components
    obs_text = obs['text'][item] if obs['text'] is not None else None
    obs_image = obs['image'][item] if obs['image'] is not None else None
    obs_anchor = obs['anchor'][item] if obs['anchor'] is not None else None

    # Build chat structure with current observation
    chat = np.array([{
        "content": obs_text,
        "role": "user",
    }])

    # Apply chat template
    prompt_with_chat_template = self.tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False, **kwargs
    )

    # Handle multimodal inputs
    if obs_image is not None:
        # Process image and add vision tokens
        row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
        image_inputs = self.processor.image_processor(...)
        # ... vision token insertion

    # Tokenize
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=prompt_with_chat_template,
        tokenizer=self.tokenizer,
        max_length=self.config.data.max_prompt_length,
        truncation=self.config.data.truncation,
    )

    # Build output
    row_dict = {
        'input_ids': input_ids[0],
        'attention_mask': attention_mask[0],
        'position_ids': position_ids[0],
        'anchor_obs': obs_anchor,  # Critical for GiGPO step grouping
        'index': item,
        'data_source': data_source
    }

    return row_dict
```

**Critical Component**: `anchor_obs` extraction enables step-level grouping

### 3. Integration in Trainer

**File**: `/data1/zzq/rl-proj/verl-agent/verl/trainer/ppo/ray_trainer.py`

**Advantage Computation** (Lines 345-356):

```python
elif adv_estimator == AdvantageEstimator.GiGPO:
    advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=data.batch['token_level_rewards'],
        step_rewards=data.batch['step_rewards'],
        response_mask=data.batch['response_mask'],
        anchor_obs=data.non_tensor_batch['anchor_obs'],
        index=data.non_tensor_batch['uid'],
        traj_index=data.non_tensor_batch['traj_uid'],
        epsilon=epsilon,
        step_advantage_w=config.algorithm.gigpo.step_advantage_w,
        mode=config.algorithm.gigpo.mode,
        enable_similarity=config.algorithm.gigpo.get('enable_similarity', False),
        similarity_thresh=config.algorithm.gigpo.get('similarity_thresh', 0.95),
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

### 4. Reward Assignment

**File**: `/data1/zzq/rl-proj/verl-agent/agent_system/reward_manager/episode.py`

```python
class EpisodeRewardManager:
    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards

            # Assign reward to final token only (outcome reward)
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, ...)

        return reward_tensor
```

**Design**: Outcome-based rewards placed at trajectory end (standard for episodic tasks)

---

## Key Design Decisions

### 1. Why Two-Level Grouping?

**Episode-Level**:
- Captures overall trajectory quality
- Reduces variance by comparing rollouts from same initial state
- Provides coarse-grained credit assignment

**Step-Level**:
- Enables fine-grained credit assignment
- Rewards good actions even in failed trajectories
- Penalizes bad actions even in successful trajectories

**Combination**: Balances exploration and exploitation at different granularities

### 2. Critic-Free Design

**Motivation**:
- Value functions difficult to train for sparse, long-horizon tasks
- Reduces memory and computational requirements
- Simpler implementation and hyperparameter tuning

**Trade-off**:
- Requires multiple rollouts per state (controlled by `env.rollout.n`)
- Higher sample complexity than critic-based methods
- But: More stable training without critic warm-up issues

### 3. Step-Independent Rollout

**Traditional Approach** (e.g., RAGEN, Search-R1):
```
Step 1: [Task] → Response 1
Step 2: [Task][Response 1][Obs 2] → Response 2
Step 3: [Task][Response 1][Obs 2][Response 2][Obs 3] → Response 3
...
Context length: O(n) where n = number of steps
```

**verl-agent Approach**:
```
Step 1: [Task] → Response 1
Step 2: [Obs 2 + Memory] → Response 2
Step 3: [Obs 3 + Memory] → Response 3
...
Context length: O(1) (approximately constant)
```

**Benefits**:
- Scalable to 30-50+ steps without hitting token limits
- Flexible memory management (see `/data1/zzq/rl-proj/verl-agent/agent_system/memory/`)
- Customizable per-step inputs

### 4. Similarity-Based Grouping

**Exact Matching** (`enable_similarity=False`):
- Default mode
- Uses `to_hashable()` for exact state matching
- Fast and deterministic

**Similarity-Based** (`enable_similarity=True`):
- Uses `SequenceMatcher` for text similarity
- Useful when observations have minor variations but represent same state
- Example: "You are in a kitchen" vs "You're in a kitchen."
- Trade-off: More flexible but slower and potentially less precise

**Configuration** (from training script):
```bash
algorithm.gigpo.enable_similarity=True \
algorithm.gigpo.similarity_thresh=0.95
```

### 5. Normalization Modes

**Mean Normalization** (`mode="mean_norm"`):
```python
advantage = reward - mean(rewards_in_group)
```
- More stable (recommended default)
- Avoids division by small std values
- Used in most experiments

**Mean-Std Normalization** (`mode="mean_std_norm"`):
```python
advantage = (reward - mean(rewards_in_group)) / (std(rewards_in_group) + ε)
```
- Better for groups with high reward variance
- Can be unstable with small groups

### 6. Dynamic Sampling (DAPO Extension)

**Purpose**: Improve sample efficiency by filtering uninformative groups

**Mechanism**:
1. Over-sample: Generate more episode groups than needed
2. Filter: Remove groups where all trajectories have identical rewards
3. Repeat: Until target batch size reached or max attempts exceeded

**Benefits**:
- Reduces wasted computation on trivial states
- Improves gradient signal quality
- Works with both GiGPO and GRPO

**Configuration**:
```bash
algorithm.filter_groups.enable=True \
algorithm.filter_groups.max_num_gen_batches=10
```

---

## Performance Characteristics

### Computational Complexity

**Per Training Step**:

1. **Rollout**: O(B × T × L) where:
   - B = batch size
   - T = average episode length
   - L = sequence length per step

2. **Advantage Computation**:
   - Episode grouping: O(B) hash table operations
   - Step grouping: O(B × T) for exact matching, O(B × T²) for similarity
   - Normalization: O(B × T)

3. **Memory**: O(B × T × L) for trajectory storage

**Comparison to PPO**:
- No critic network: Saves ~50% model memory
- More rollouts needed: 2-8x more samples per update
- Overall: Comparable or better sample efficiency due to better credit assignment

### Hyperparameter Sensitivity

**Critical Parameters**:

1. **`env.rollout.n`** (Episode group size):
   - Typical: 4-8
   - Too small: High variance in advantages
   - Too large: Computational overhead, memory pressure

2. **`algorithm.gigpo.step_advantage_w`** (Step advantage weight):
   - Typical: 0.5-1.5
   - Task-dependent: Higher for long-horizon tasks requiring fine-grained credit

3. **`algorithm.gamma`** (Discount factor):
   - Typical: 0.95-0.99
   - Higher for longer episodes

4. **`algorithm.gigpo.mode`**:
   - "mean_norm": More stable (recommended)
   - "mean_std_norm": Can help with high variance

### Empirical Results

From verl-agent README and paper:

**ALFWorld** (30-50 steps):
- Qwen2.5-1.5B: 86.7% success rate
- Qwen2.5-7B: 90.8% success rate
- Superior to GRPO, PPO, RLOO

**WebShop** (10-20 steps):
- Qwen2.5-1.5B: 67.4% success rate
- Qwen2.5-7B: 75.2% success rate

**Search-R1** (Question Answering):
- Qwen2.5-3B: 42.1% average accuracy (best among all methods)
- Qwen2.5-7B: Superior to Search-R1, ZeroSearch, StepSearch

**Visual Tasks**:
- Sokoban (Qwen2.5-VL-3B): 81.0% success
- EZPoints (Qwen2.5-VL-3B): 100.0% success

---

## Usage Example

### Training Configuration

From `/data1/zzq/rl-proj/verl-agent/examples/gigpo_trainer/run_alfworld.sh`:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_batch_size=16 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    env.env_name=alfworld/AlfredTWEnv \
    env.max_steps=50 \
    env.rollout.n=8 \
    trainer.total_epochs=150
```

**Key Settings**:
- `algorithm.adv_estimator=gigpo`: Use GiGPO algorithm
- `algorithm.gamma=0.95`: Discount factor for long episodes
- `algorithm.gigpo.step_advantage_w=1.0`: Equal weight for step and episode advantages
- `algorithm.gigpo.mode=mean_norm`: Use mean normalization (more stable)
- `env.rollout.n=8`: 8 trajectories per episode group
- `env.max_steps=50`: Allow up to 50 steps per episode

### Custom Environment Integration

To use GiGPO with a new environment:

1. **Implement anchor observation extraction**:
   ```python
   class MyEnvironment(EnvironmentManagerBase):
       def reset(self):
           obs = {
               'text': [...],  # Full observation with history
               'anchor': [...],  # Canonical state representation
           }
           return obs, infos
   ```

2. **Configure prompts** in `/data1/zzq/rl-proj/verl-agent/agent_system/environments/prompts/`:
   ```python
   def build_text_obs(self, env_info, memory=None):
       # Construct step-specific input
       obs_text = env_info['current_state']
       if memory:
           obs_text = memory.get_summary() + "\n" + obs_text
       return obs_text
   ```

3. **Set up reward function**:
   ```python
   def compute_reward(self, action, next_state):
       # Sparse or dense rewards
       return reward, done, info
   ```

---

## Summary

**GiGPO** represents a significant advancement in RL for LLM agents:

**Strengths**:
1. Effective credit assignment without critic networks
2. Scalable to very long-horizon tasks (30-50+ steps)
3. Flexible memory and input management
4. Strong empirical performance across diverse tasks
5. Compatible with existing RL infrastructure (veRL)

**When to Use GiGPO**:
- Long-horizon agent tasks (10+ steps)
- Environments with identifiable state transitions
- When critic training is unstable or memory-constrained
- Multi-turn interactive scenarios

**When to Consider Alternatives**:
- Very short episodes (1-3 steps): GRPO may suffice
- Continuous state spaces: PPO with critic may be more sample-efficient
- When rollout costs are extremely high: Critic-based methods may be preferable

**Implementation Quality**:
The verl-agent codebase provides a clean, modular implementation with:
- Well-documented core algorithms
- Flexible environment interface
- Support for text and vision modalities
- Integration with modern LLM infrastructure (vLLM, SGLang)
- Extensive configuration system

This makes it an excellent foundation for research and applications in LLM agent training.

---

## References

- **Paper**: He et al. "GiGPO: Group-in-Group Policy Optimization for Multi-Turn Reinforcement Learning of Large Language Model Agents" NeurIPS 2025. [https://arxiv.org/abs/2505.10978](https://arxiv.org/abs/2505.10978)
- **Code**: [https://github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)
- **veRL**: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)

---

**Document Created**: 2026-02-05
**Code Version**: verl-agent (post-2025.06.03 major update)
**Analysis Based On**:
- `/data1/zzq/rl-proj/verl-agent/gigpo/core_gigpo.py`
- `/data1/zzq/rl-proj/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py`
- `/data1/zzq/rl-proj/verl-agent/verl/trainer/ppo/ray_trainer.py`
- `/data1/zzq/rl-proj/verl-agent/agent_system/reward_manager/episode.py`
