# verl-agent Memory System Analysis

## Overview

The memory system in verl-agent is a critical component that enables agents to maintain and recall interaction history during multi-turn reinforcement learning tasks. Located in `/data1/zzq/rl-proj/verl-agent/agent_system/memory/`, the system provides a flexible abstraction for storing, retrieving, and formatting agent-environment interaction histories.

## Architecture

### Component Structure

The memory system consists of three main files:

1. **`base.py`**: Abstract base class defining the memory interface
2. **`memory.py`**: Concrete implementations (SimpleMemory and SearchMemory)
3. **`__init__.py`**: Module exports

### Class Hierarchy

```
BaseMemory (Abstract Base Class)
├── SimpleMemory (General-purpose memory)
└── SearchMemory (Search-task specialized memory)
```

## Core Design Principles

### 1. Batched Environment Support

The memory system is designed to handle multiple parallel environment instances simultaneously. Each memory instance maintains separate history records for `batch_size` environments, enabling efficient vectorized training across multiple episodes.

**Key insight**: This batching capability is essential for modern RL training where parallelization significantly improves sample efficiency and training speed.

### 2. Step-Independent Rollout

A fundamental design goal is to support **step-independent multi-turn rollouts**. Unlike traditional approaches that concatenate full conversation histories, this system:

- Stores complete history internally
- Retrieves only recent steps (controlled by `history_length`)
- Keeps context length nearly constant over time
- Enables scaling to very long-horizon tasks (30-50+ steps)

### 3. Flexible Schema Support

The memory system uses a dictionary-based schema where keys can represent any type of data (observations, actions, rewards, etc.). This flexibility allows different environments to store environment-specific information without modifying the core memory implementation.

## Implementation Details

### BaseMemory Abstract Class

**File**: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/base.py`

Defines the contract that all memory implementations must follow:

```python
class BaseMemory(ABC):
    @abstractmethod
    def __len__(self): pass

    @abstractmethod
    def __getitem__(self, idx: int): pass

    @abstractmethod
    def reset(self, batch_size: int): pass

    @abstractmethod
    def store(self, record: Dict[str, List[Any]]): pass

    @abstractmethod
    def fetch(self, step: int): pass
```

**Purpose**: Ensures consistency across different memory implementations and enables easy extension with custom memory strategies.

### SimpleMemory Implementation

**File**: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/memory.py` (lines 19-101)

#### Data Structure

```python
self._data = [[] for _ in range(batch_size)]
```

- **Outer list**: One entry per environment instance (length = batch_size)
- **Inner list**: Sequential history of interaction steps
- **Each step**: Dictionary mapping keys (e.g., 'text_obs', 'action') to values

**Example structure**:
```python
_data = [
    # Environment 0
    [
        {'text_obs': 'You see a door', 'action': 'open door'},
        {'text_obs': 'Door is open', 'action': 'go through'},
        ...
    ],
    # Environment 1
    [
        {'text_obs': 'You are in room', 'action': 'examine table'},
        ...
    ],
    ...
]
```

#### Core Methods

##### 1. `reset(batch_size: int)`

```python
def reset(self, batch_size: int):
    if self._data is not None:
        self._data.clear()
    self._data = [[] for _ in range(batch_size)]
    self.batch_size = batch_size
    self.keys = None
```

**Purpose**: Initialize or reinitialize memory for a new batch of episodes.

**When called**: At the start of each training episode or when environment is reset.

**Effect**:
- Clears all existing history
- Creates fresh empty lists for each environment
- Resets key schema to None (will be inferred from first store call)

##### 2. `store(record: Dict[str, List[Any]])`

```python
def store(self, record: Dict[str, List[Any]]):
    if self.keys is None:
        self.keys = list(record.keys())
    assert self.keys == list(record.keys())

    for env_idx in range(self.batch_size):
        self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})
```

**Purpose**: Add one step of interaction history for all environments simultaneously.

**Input format**:
```python
record = {
    'text_obs': [obs_env0, obs_env1, ..., obs_envN],
    'action': [action_env0, action_env1, ..., action_envN]
}
```

**Process**:
1. On first call, infer and store schema keys from record
2. Assert schema consistency on subsequent calls (fail-fast on schema mismatch)
3. Transpose data from batch-first to environment-first format
4. Append step record to each environment's history

**Key insight**: The transposition from `{key: [batch]}` to `[env][{key: value}]` enables efficient per-environment retrieval while accepting batch-oriented inputs.

##### 3. `fetch(history_length: int, obs_key: str, action_key: str)`

```python
def fetch(
    self,
    history_length: int,
    obs_key: str = "text_obs",
    action_key: str = "action",
) -> Tuple[List[str], List[int]]:
```

**Purpose**: Retrieve and format recent interaction history for constructing agent prompts.

**Process**:

1. **Retrieve recent steps**:
   ```python
   recent = self._data[env_idx][-history_length:]
   ```
   Uses Python's negative indexing to get last N steps (handles cases where total history < history_length)

2. **Calculate valid length**:
   ```python
   valid_len = len(recent)
   start_idx = len(self._data[env_idx]) - valid_len
   ```
   Tracks actual number of steps retrieved (may be less than history_length at episode start)

3. **Format history**:
   ```python
   for j, rec in enumerate(recent):
       step_num = start_idx + j + 1
       act = rec[action_key]
       obs = rec[obs_key]
       lines.append(
           f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
       )
   ```

   Creates formatted string like:
   ```
   [Observation 1: 'You see a door', Action 1: 'open door']
   [Observation 2: 'Door is open', Action 2: 'go through']
   ```

**Returns**:
- `memory_contexts: List[str]`: Formatted history strings, one per environment
- `valid_lengths: List[int]`: Actual number of steps in each history (for prompt engineering)

**Key insight**: By numbering steps with absolute indices (`start_idx + j + 1`), the agent maintains consistent step counting even when history is truncated.

### SearchMemory Implementation

**File**: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/memory.py` (lines 103-184)

#### Differences from SimpleMemory

SearchMemory is nearly identical to SimpleMemory but customizes the `fetch()` method for search-specific formatting:

```python
def fetch(
    self,
    history_length: int,
    obs_key: str,
    action_key: str,
) -> Tuple[List[str], List[int]]:
    # ... same retrieval logic ...

    lines.append(
        f"Step {step_num}:{act} {obs}\n"
    )
```

**Formatting difference**:
- **SimpleMemory**: `[Observation N: 'obs', Action N: 'act']`
- **SearchMemory**: `Step N:act obs\n`

**Rationale**: Different environments may benefit from different prompt formats. Search tasks typically have shorter, more structured observations that work better with compact formatting.

**Design pattern**: Environment-specific memory classes enable fine-grained control over prompt construction without modifying core logic.

## Integration with Environment System

### Usage Pattern

The memory system integrates with environment managers following this lifecycle:

#### 1. Initialization

```python
class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()  # Create memory instance
        super().__init__(envs, projection_f, config)
```

Different environment managers choose appropriate memory implementations:
- `AlfWorldEnvironmentManager`: Uses `SimpleMemory`
- `SearchEnvironmentManager`: Uses `SearchMemory`

#### 2. Episode Reset

```python
def reset(self, kwargs):
    text_obs, image_obs, infos = self.envs.reset()
    self.memory.reset(batch_size=len(text_obs))  # Initialize memory
    # ... build initial observations ...
```

#### 3. Step Execution and Storage

```python
def step(self, text_actions: List[str]):
    actions, valids = self.projection_f(text_actions, ...)
    text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)

    # Store step in memory
    self.memory.store({
        'text_obs': self.pre_text_obs,
        'action': actions
    })

    self.pre_text_obs = text_obs  # Save for next step
    # ... build next observations ...
```

**Important pattern**: Stores the **previous observation** with the **current action**, aligning observation-action pairs correctly.

#### 4. Observation Construction

```python
def build_text_obs(self, text_obs: List[str], admissible_actions, init: bool = False):
    postprocess_text_obs = []

    if not init and self.config.env.history_length > 0:
        # Retrieve memory for prompt construction
        memory_contexts, valid_lens = self.memory.fetch(
            self.config.env.history_length,
            obs_key="text_obs",
            action_key="action"
        )

    for i in range(len(text_obs)):
        if init or self.config.env.history_length <= 0:
            # No history template
            obs = ALFWORLD_TEMPLATE_NO_HIS.format(...)
        else:
            # Include history in prompt
            obs = ALFWORLD_TEMPLATE.format(
                task_description=self.tasks[i],
                step_count=len(self.memory[i]),
                history_length=valid_lens[i],
                action_history=memory_contexts[i],
                current_step=len(self.memory[i]) + 1,
                current_observation=text_obs[i],
                admissible_actions=...
            )
        postprocess_text_obs.append(obs)

    return postprocess_text_obs
```

**Key integration points**:
- `len(self.memory[i])`: Total steps taken (enables step counting in prompts)
- `memory_contexts[i]`: Formatted history string (injected into prompt template)
- `valid_lens[i]`: Number of history steps (useful for conditional formatting)

## Design Advantages

### 1. Constant Context Length

By fetching only the most recent `history_length` steps, the system prevents context length from growing linearly with episode length. This is crucial for:

- **Memory efficiency**: GPU memory doesn't explode for long episodes
- **Computational efficiency**: Attention mechanisms scale quadratically with context length
- **Training stability**: Consistent input sizes across episodes
- **Scalability**: Enables tasks with 30-50+ steps that would otherwise exceed model context limits

### 2. Separation of Concerns

The memory system cleanly separates:
- **Storage logic**: How history is maintained (memory module)
- **Retrieval logic**: What history to retrieve (configurable history_length)
- **Formatting logic**: How history is presented (environment-specific fetch methods)
- **Prompt construction**: How history integrates with task context (environment manager)

This modularity enables:
- Easy testing of individual components
- Independent optimization of each layer
- Custom memory strategies without touching environment code

### 3. Extensibility

The abstract base class enables custom memory implementations for specialized needs:

**Potential extensions**:
- **Summarization memory**: Use LLM to compress old history into summaries
- **Selective memory**: Keep only important steps (high reward, failures)
- **External memory**: Store in database for very long episodes
- **Episodic memory**: Retrieve similar past situations from previous episodes
- **Working memory**: Different retention policies for different information types

Example custom extension:
```python
class SummarizingMemory(BaseMemory):
    def fetch(self, history_length: int, ...):
        recent = self._data[env_idx][-history_length:]
        if len(self._data[env_idx]) > history_length * 2:
            # Summarize older history
            old_summary = self._summarize(
                self._data[env_idx][:-history_length]
            )
            return f"{old_summary}\n{format(recent)}", ...
        return format(recent), ...
```

### 4. Batch-First Interface

The `store()` method accepts batch-first data format (`{key: [values]}`) which aligns with:
- PyTorch tensor conventions
- Vectorized environment outputs
- RL training batch processing

Yet internally stores in environment-first format for efficient retrieval.

## Performance Considerations

### Time Complexity

- **reset()**: O(batch_size) - creates empty lists
- **store()**: O(batch_size × num_keys) - appends to each environment's history
- **fetch()**: O(batch_size × history_length) - retrieves and formats recent steps
- **__getitem__()**: O(1) - direct list access

### Memory Complexity

- **Per environment**: O(max_episode_length × num_keys × value_size)
- **Total**: O(batch_size × max_episode_length × num_keys × value_size)

For typical configurations:
- batch_size: 16-64
- max_episode_length: 30-50 steps
- num_keys: 2-5 fields
- value_size: Strings of 100-1000 characters

Total memory is modest (typically < 100MB) and dominated by model parameters rather than history storage.

### Optimization Opportunities

1. **Circular buffer**: For very long episodes, use fixed-size buffer instead of unbounded list
2. **Lazy formatting**: Defer string formatting until actually needed (if fetch happens less often than store)
3. **Compression**: Compress old history entries that are unlikely to be retrieved

## Configuration System Integration

The memory system is controlled via configuration parameters:

```yaml
env:
  history_length: 5  # Number of recent steps to include in prompts
```

**Common settings**:
- `history_length: 0` - No history (stateless prompts)
- `history_length: 3-5` - Short-term memory (recent context)
- `history_length: 10-20` - Long-term memory (full episode context for short tasks)

**Trade-offs**:
- **Smaller history_length**:
  - ✓ Shorter context (faster, more efficient)
  - ✗ Less information for decision-making

- **Larger history_length**:
  - ✓ More context for agent reasoning
  - ✗ Longer prompts (slower, more memory)
  - ✗ May exceed model context limits

## Comparison with Alternative Approaches

### Traditional Full-History Concatenation

**Approach**: Concatenate all previous turns into a single growing string.

```python
history = ""
for step in range(episode_length):
    history += f"\nObs: {obs[step]}\nAction: {action[step]}"
    prompt = f"{task}\n{history}\n{current_obs}"
```

**Problems**:
- Context length grows linearly with episode length
- Exceeds model limits for long episodes
- Increasingly expensive attention computation
- Redundant information in later steps

**verl-agent solution**: Fixed-size sliding window keeps only recent steps.

### Recurrent/Stateful Models

**Approach**: Use RNN/LSTM hidden states to maintain memory.

**Problems**:
- Requires training recurrent architectures (more complex)
- Hidden states are opaque (hard to interpret/debug)
- Limited adoption in modern LLM era
- Doesn't leverage pretrained language model knowledge

**verl-agent solution**: Explicit text-based memory works with any pretrained LLM.

### Retrieval-Augmented Generation (RAG)

**Approach**: Store all history in vector database, retrieve relevant steps.

**Problems**:
- Significant engineering complexity (embedding models, vector DBs)
- Retrieval adds latency to every step
- May miss temporally-adjacent relevant information
- Overkill for short-horizon tasks

**verl-agent solution**: Simple recency-based retrieval is sufficient and efficient for RL tasks.

## Real-World Usage Examples

### Example 1: ALFWorld Navigation

**Task**: Find and pick up objects in a house.

**Memory usage**:
```python
# Step 1
memory.store({
    'text_obs': ["You are in a kitchen. You see a table."],
    'action': ["go to table"]
})

# Step 2
memory.store({
    'text_obs': ["You are at the table. You see an apple."],
    'action': ["take apple"]
})

# Step 3 - Fetch for prompt construction
history, lengths = memory.fetch(
    history_length=2,
    obs_key="text_obs",
    action_key="action"
)
# Returns:
# history[0] = "[Observation 1: 'You are in kitchen...', Action 1: 'go to table']
#               [Observation 2: 'You are at table...', Action 2: 'take apple']"
```

**Prompt constructed**:
```
Task: Pick up the apple and put it in the fridge.
Step count: 2
Action history (last 2 steps):
[Observation 1: 'You are in kitchen...', Action 1: 'go to table']
[Observation 2: 'You are at table...', Action 2: 'take apple']

Current observation: You are holding an apple. You see a fridge.
Admissible actions:
 'go to fridge'
 'drop apple'
 ...
```

### Example 2: Search Task

**Task**: Perform web searches to answer a question.

**Memory usage**:
```python
# Using SearchMemory
memory.store({
    'search': ["What is quantum computing"],
    'information': ["Quantum computing uses quantum bits..."]
})

memory.store({
    'search': ["Applications of quantum computing"],
    'information': ["Drug discovery, cryptography, optimization..."]
})

history, lengths = memory.fetch(
    history_length=5,
    obs_key="information",
    action_key="search"
)
# Returns:
# history[0] = "Step 1:What is quantum computing Quantum computing uses...\n
#               Step 2:Applications of quantum computing Drug discovery...\n"
```

**Prompt constructed**:
```
Task: Find information about quantum computing applications in medicine.

Previous searches (Step 1-2):
Step 1:What is quantum computing Quantum computing uses quantum bits...
Step 2:Applications of quantum computing Drug discovery, cryptography...

Current step: 3
What would you like to search next?
```

## Best Practices

### 1. Choose Appropriate history_length

- **Short tasks (5-10 steps)**: Use `history_length = max_steps` to include full history
- **Medium tasks (10-30 steps)**: Use `history_length = 5-10` for recent context
- **Long tasks (30+ steps)**: Use `history_length = 5-10` to prevent context explosion

### 2. Consistent Schema

Always store the same keys in every step:
```python
# Good: Consistent keys
memory.store({'text_obs': obs, 'action': actions})
memory.store({'text_obs': new_obs, 'action': new_actions})

# Bad: Changing keys (will trigger assertion error)
memory.store({'text_obs': obs, 'action': actions})
memory.store({'observation': new_obs, 'response': new_actions})  # Error!
```

### 3. Observation-Action Alignment

Store previous observation with current action (since action was taken based on that observation):
```python
text_obs, _, _, _, _ = env.step(actions)
memory.store({'text_obs': previous_obs, 'action': actions})
previous_obs = text_obs  # Update for next step
```

### 4. Reset Between Episodes

Always call `memory.reset()` when starting new episodes:
```python
def reset(self, kwargs):
    obs, infos = self.envs.reset(kwargs=kwargs)
    self.memory.reset(batch_size=len(obs))  # Critical!
    # ... rest of reset logic ...
```

Forgetting to reset will leak history from previous episodes into new ones.

### 5. Custom Formatting for Environment Types

For new environments, consider whether custom formatting improves prompt quality:
```python
class CustomMemory(BaseMemory):
    def fetch(self, history_length, obs_key, action_key):
        # ... retrieval logic ...

        # Custom format for your domain
        for j, rec in enumerate(recent):
            if rec.get('error'):
                lines.append(f"❌ Step {step_num}: {act} (FAILED)")
            else:
                lines.append(f"✓ Step {step_num}: {act}")

        return "\n".join(lines), valid_lengths
```

## Future Extensions and Research Directions

### 1. Adaptive History Length

Dynamically adjust `history_length` based on:
- Task complexity (easier tasks need less history)
- Episode progress (early steps may need more history)
- Model uncertainty (retrieve more history when uncertain)

### 2. Importance-Weighted Memory

Instead of simple recency, weight steps by importance:
```python
def fetch_important(self, max_steps):
    # Retrieve steps with highest reward, failures, or key decisions
    important_steps = self._rank_by_importance()
    return important_steps[:max_steps]
```

### 3. Hierarchical Memory

Maintain multiple timescales:
- **Working memory**: Last 3 steps (detailed)
- **Short-term memory**: Last 10 steps (summaries)
- **Long-term memory**: Full episode (compressed)

### 4. Cross-Episode Memory

Learn from previous episodes:
```python
class EpisodicMemory(BaseMemory):
    def __init__(self):
        self.episode_db = []  # Store completed episodes

    def fetch_similar_episodes(self, current_obs):
        # Retrieve similar situations from past episodes
        # Use for few-shot prompting or learning from mistakes
```

### 5. Multi-Modal Memory

Extend beyond text to store:
- Images/screenshots
- Audio observations
- Structured data (graphs, tables)
- Embeddings for semantic retrieval

## Conclusion

The verl-agent memory system exemplifies elegant engineering:

**Simplicity**: Core implementation is under 100 lines per class, yet handles complex multi-turn RL scenarios.

**Flexibility**: Abstract base class enables environment-specific customization while maintaining consistent interface.

**Efficiency**: Batched operations, constant context length, and minimal overhead enable large-scale training.

**Scalability**: Supports long-horizon tasks (30-50+ steps) that would be infeasible with naive full-history approaches.

**Integration**: Clean separation of storage, retrieval, and formatting concerns enables modular development.

The design demonstrates how thoughtful abstractions can solve complex problems in RL systems. By recognizing that agents don't need full history—just relevant recent context—the system achieves both efficiency and effectiveness.

This memory system is a cornerstone of verl-agent's ability to train agents on long-horizon tasks, enabling the GiGPO algorithm to perform fine-grained credit assignment across extended interaction sequences while maintaining constant computational costs per step.

## File Locations

- Base class: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/base.py`
- Implementations: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/memory.py`
- Module exports: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/__init__.py`
- Usage example: `/data1/zzq/rl-proj/verl-agent/agent_system/environments/env_manager.py`

## References

For more context on how this memory system fits into the broader architecture:
- GiGPO algorithm: Uses memory for step-level advantage estimation
- Environment managers: Integrate memory with prompt construction
- Training pipeline: Configures history_length via YAML configs
- verl-agent README: High-level overview of step-independent rollouts
