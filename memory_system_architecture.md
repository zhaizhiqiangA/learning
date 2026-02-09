# verl-agent Memory System Architecture Analysis

## Overview

The verl-agent memory system provides a flexible, step-independent history management module for long-horizon reinforcement learning tasks. It stores per-environment interaction histories and enables customizable retrieval strategies to maintain constant context length across multi-step episodes.

## Architecture

### Class Hierarchy

```
BaseMemory (Abstract Base Class)
├── SimpleMemory (General-purpose implementation)
└── SearchMemory (Search-specific implementation)
```

The memory system follows an interface-based design where `BaseMemory` defines the contract, and concrete implementations provide task-specific formatting strategies.

### Component Structure

**Core Classes:**
- **BaseMemory**: Abstract interface defining memory operations
- **SimpleMemory**: Default implementation for most agent environments (ALFWorld, Sokoban, WebShop, AppWorld)
- **SearchMemory**: Specialized implementation for search tasks with different formatting

**Integration Points:**
- **EnvironmentManager**: Each environment manager instantiates appropriate memory type
- **Prompt Templates**: Memory contexts are injected into prompts during observation building
- **Rollout System**: Memory is reset/stored/fetched during episode execution

## Core Principles

### 1. Constant Context Length Through Sliding Window

**Why**: Prevent memory explosion and context overflow in long-horizon tasks (30-50+ steps)

**Implementation**:
- `memory.py:84` - Sliding window retrieval: `recent = self._data[env_idx][-history_length:]`
- `env_manager.py:186-189` - Configuration-driven: `self.memory.fetch(self.config.env.history_length, ...)`
- `memory.py:90` - Absolute step indexing: `step_num = start_idx + j + 1`

**Result**: Context length stays bounded regardless of episode length. A 50-step episode with `history_length=5` maintains the same context size as step 5.

### 2. Per-Environment Independent Storage

**Why**: Batch parallel execution with different episode lengths and states

**Implementation**:
- `memory.py:37` - Separate storage per environment: `self._data = [[] for _ in range(batch_size)]`
- `memory.py:55-56` - Per-environment record appending:
```python
for env_idx in range(self.batch_size):
    self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})
```

**Result**: Each environment in a batch maintains independent history, supporting group sampling (GRPO/DAPO) with shared initial states but divergent trajectories.

### 3. Flexible Key-Value Schema

**Why**: Different environments require different data types (text observations, actions, images, metadata)

**Implementation**:
- `memory.py:51-53` - Dynamic schema inference:
```python
if self.keys is None:
    self.keys = list(record.keys())
assert self.keys == list(record.keys())
```
- `memory.py:91-92` - Key-based retrieval: `act = rec[action_key]`, `obs = rec[obs_key]`

**Result**: Memory system adapts to any environment without modification. ALFWorld stores `{text_obs, action}`, Search stores `{search, information}`, AppWorld stores additional metadata.

### 4. Task-Specific Formatting

**Why**: Different environments have different interaction patterns and display requirements

**Implementation**:
- **SimpleMemory** (lines 93-95): `[Observation {step_num}: '{obs}', Action {step_num}: '{act}']`
- **SearchMemory** (lines 177-179): `Step {step_num}:{act} {obs}\n`

**Result**: Memory output matches environment-specific prompt templates while sharing the same storage mechanism.

### 5. Absolute Step Numbering

**Why**: Agent needs to track progress in long episodes despite sliding window

**Implementation**:
- `memory.py:86-90`:
```python
valid_len = len(recent)
start_idx = len(self._data[env_idx]) - valid_len
for j, rec in enumerate(recent):
    step_num = start_idx + j + 1
```

**Result**: At step 25 with `history_length=5`, memory displays "Step 21, 22, 23, 24, 25" rather than "Step 1, 2, 3, 4, 5", providing temporal context.

## Implementation Details

### Class: BaseMemory (Abstract Interface)

**Purpose**: Define the contract for all memory implementations

**Code** (lines 20-54 of base.py):
```python
class BaseMemory(ABC):
    """
    Base class for memory management. Defines the interface for memory modules.
    """

    @abstractmethod
    def __len__(self):
        """Return the number of memory slots."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """Access memory of specific environment index."""
        pass

    @abstractmethod
    def reset(self, batch_size: int):
        """
        Reset memory with given batch size.
        """
        pass

    @abstractmethod
    def store(self, record: Dict[str, List[Any]]):
        """
        Stores a new batch of records into memory.
        """
        pass

    @abstractmethod
    def fetch(self, step: int):
        """
        Fetches memory records at a specific time step across all environments.
        """
        pass
```

**Key details**:
- Uses ABC (Abstract Base Class) pattern for enforced polymorphism (line 16)
- `__getitem__` enables direct indexing: `memory[env_idx]` (line 31-33)
- `fetch()` signature is abstract, allowing implementations to define custom parameters

### Class: SimpleMemory

**Purpose**: General-purpose memory implementation for most agent environments

#### Method: `__init__()`

**Code** (lines 23-26):
```python
def __init__(self):
    self._data = None
    self.keys = None
    self.batch_size = 0
```

**Key details**:
- Lazy initialization pattern - storage created on `reset()` (line 24)
- Schema-agnostic initialization - keys determined at first `store()` (line 25)

#### Method: `reset()`

**Purpose**: Initialize storage for new episode batch

**Code** (lines 34-39):
```python
def reset(self, batch_size: int):
    if self._data is not None:
        self._data.clear()
    self._data = [[] for _ in range(batch_size)]
    self.batch_size = batch_size
    self.keys = None
```

**Key details**:
- Creates independent list for each environment instance (line 37)
- Clears existing data for reuse across episodes (line 35-36)
- Resets schema to allow different key structures across episodes (line 39)

#### Method: `store()`

**Purpose**: Append one step of interaction history across all environments

**Code** (lines 41-56):
```python
def store(self, record: Dict[str, List[Any]]):
    """
    Store a new record (one step of history) for each environment instance.

    Args:
        record (Dict[str, List[Any]]):
            A dictionary where each key corresponds to a type of data
            (e.g., 'text_obs', 'action'), and each value is a list of
            length `batch_size`, containing the data for each environment.
    """
    if self.keys is None:
        self.keys = list(record.keys())
    assert self.keys == list(record.keys())

    for env_idx in range(self.batch_size):
        self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})
```

**Key details**:
- Input format: `{"text_obs": [obs0, obs1, ...], "action": [act0, act1, ...]}` (lines 46-49)
- Schema inference on first call (lines 51-52)
- Validates consistent schema across steps (line 53)
- Transposes batch-first format to environment-first storage (line 55-56)

**Data flow transformation**:
```
Input:  {"text_obs": ["A", "B"], "action": ["go", "stop"]}
Output: self._data = [
    [{"text_obs": "A", "action": "go"}],
    [{"text_obs": "B", "action": "stop"}]
]
```

#### Method: `fetch()`

**Purpose**: Retrieve recent history formatted as strings for prompt injection

**Code** (lines 58-100):
```python
def fetch(
    self,
    history_length: int,
    obs_key: str = "text_obs",
    action_key: str = "action",
) -> Tuple[List[str], List[int]]:
    """
    Fetch and format recent interaction history for each environment instance.
    Args:
        history_length (int):
            Maximum number of past steps to retrieve per environment.
        obs_key (str, default="text_obs"):
            The key name used to access the observation in stored records.
        action_key (str, default="action"):
            The key name used to access the action in stored records.
    Returns:
        memory_contexts : List[str]
            A list of formatted action history strings for each environment.
        valid_lengths : List[int]
            A list of the actual number of valid history steps per environment.
    """
    memory_contexts, valid_lengths = [], []

    for env_idx in range(self.batch_size):
        recent = self._data[env_idx][-history_length:]
        valid_len = len(recent)
        start_idx = len(self._data[env_idx]) - valid_len

        lines = []
        for j, rec in enumerate(recent):
            step_num = start_idx + j + 1
            act = rec[action_key]
            obs = rec[obs_key]
            lines.append(
                f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
            )

        memory_contexts.append("\n".join(lines))
        valid_lengths.append(valid_len)

    return memory_contexts, valid_lengths
```

**Key details**:
- Sliding window with negative indexing: `[-history_length:]` (line 84)
- Handles early episodes gracefully - if only 3 steps exist but `history_length=5`, returns 3 (line 85)
- Computes absolute step numbers for temporal awareness (line 86, 90)
- Formatting matches ALFWorld/Sokoban/WebShop/AppWorld templates (lines 93-95)
- Returns both formatted strings and actual lengths for flexible prompt construction (line 98)

**Example output** (at step 7 with `history_length=3`):
```
[Observation 5: 'You see a desk', Action 5: 'go to desk 1']
[Observation 6: 'You are at desk 1', Action 6: 'take pen 1']
[Observation 7: 'You take pen 1', Action 7: 'go to drawer 1']
```

### Class: SearchMemory

**Purpose**: Specialized memory for search tasks with custom formatting

**Key Differences from SimpleMemory**:
- Different formatting style (lines 177-179): `Step {step_num}:{act} {obs}\n`
- No default parameter values for `obs_key` and `action_key` (line 145-147)
- Single-line format instead of bracketed format

**Code** (lines 173-179):
```python
for j, rec in enumerate(recent):
    step_num = start_idx + j + 1
    act = rec[action_key]
    obs = rec[obs_key]
    lines.append(
        f"Step {step_num}:{act} {obs}\n"
    )
```

**Why separate class**: Search tasks use `{search: query, information: result}` schema and different prompt templates (see `prompts/search.py`)

## Integration

### Integration with EnvironmentManager

**Memory instantiation** (env_manager.py):
- `line 50`: SearchEnvironmentManager uses SearchMemory
- `line 135`: AlfWorldEnvironmentManager uses SimpleMemory
- `line 255`: SokobanEnvironmentManager uses SimpleMemory
- `line 387`: WebshopEnvironmentManager uses SimpleMemory
- `line 521`: AppWorldEnvironmentManager uses SimpleMemory

**Lifecycle:**
1. **Initialization**: Memory created in `__init__()` (e.g., `env_manager.py:135`)
2. **Episode Start**: Reset called in `reset()` (e.g., `env_manager.py:142`)
3. **Step Execution**: Store called in `step()` after action execution (e.g., `env_manager.py:153`)
4. **Observation Building**: Fetch called in `build_text_obs()` (e.g., `env_manager.py:186-189`)

### Integration with Prompt Templates

**ALFWorld example** (env_manager.py:186-209):
```python
if not init and self.config.env.history_length > 0:
    memory_contexts, valid_lens = self.memory.fetch(
            self.config.env.history_length,
            obs_key="text_obs",
            action_key="action")

for i in range(len(text_obs)):
    if init or self.config.env.history_length <= 0:
        obs = ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=text_obs[i],
            admissible_actions=reformatted_admissible_actions
        )
    else:
        obs = ALFWORLD_TEMPLATE.format(
            task_description=self.tasks[i],
            step_count=len(self.memory[i]),
            history_length=valid_lens[i],
            action_history=memory_contexts[i],
            current_step=len(self.memory[i]) + 1,
            current_observation=text_obs[i],
            admissible_actions=reformatted_admissible_actions
        )
```

**Key integration patterns**:
- Memory disabled for first step (`init=True`) to avoid empty history (line 195-199)
- Memory can be disabled via config: `history_length <= 0` (line 195)
- Direct memory indexing for step counting: `len(self.memory[i])` (line 203)
- Fetched context injected into prompt template (line 205)

### Integration with Search Environment

**Search-specific usage** (env_manager.py:96-101):
```python
if not init and self.config.env.history_length > 0:
    memory_ctx, _ = self.memory.fetch(
        self.config.env.history_length,
        obs_key="information",
        action_key="search"
    )
```

**Key difference**: Different key names (`information`/`search` vs `text_obs`/`action`) matching search task semantics

### AppWorld Long Context Handling

**Special case** (env_manager.py:573-585):
```python
# Get last `history_length` steps
recent_history = self.memory[i][-self.config.env.history_length:]
valid_history_length = len(recent_history)
start_index = len(self.memory[i]) - valid_history_length
action_history = ""
for j, record in enumerate(recent_history):
    step_number = start_index + j + 1
    action = record["action"]
    env_obs = record["text_obs"]
    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"

if len(action_history) > 10000:
    action_history = "... " + action_history[-10000:]
```

**Key details**:
- Manual memory fetching instead of using `fetch()` method (line 574)
- Custom formatting for code-action environment (lines 580-582)
- Hard length limit (10,000 chars) with truncation prefix (lines 584-585)
- Demonstrates extensibility - custom memory strategies can be implemented without modifying core memory class

## Trade-offs

### Performance

**Memory Overhead:**
- **Space**: O(batch_size × max_steps × record_size)
- **Time**: O(1) for store, O(history_length) for fetch
- Efficient for typical RL scenarios (batch_size ≤ 32, history_length ≤ 10)

**Context Length Management:**
- Sliding window prevents unbounded growth
- Trade-off: Older history is discarded, limiting very long-term dependencies
- Mitigation: Developers can implement summarization or selective retention (as noted in README.md)

### Complexity

**Simplicity:**
- Minimal abstraction - just 100 lines per implementation
- Easy to understand and extend
- Clear separation of storage (memory) and formatting (environment manager)

**Limitations:**
- No built-in summarization or compression
- No hierarchical memory (short-term vs long-term)
- No attention-based selective retrieval
- These are intentional - kept simple as starting point for customization

### Flexibility

**Extensibility:**
- Abstract base class enables custom implementations
- Key-value schema adapts to any environment
- Format separation allows environment-specific rendering

**Constraints:**
- Single linear history per environment (no tree/graph structure)
- Synchronous storage (all environments store at same steps)
- Fixed schema per episode (cannot change keys mid-episode)

### Design Rationale

**Why not use LLM context directly?**
- Explicit memory enables fine-grained control over context composition
- Prevents token limit overflow in long episodes
- Allows preprocessing and filtering before LLM input

**Why per-environment storage?**
- Supports batch parallel execution with different episode lengths
- Enables group sampling (GRPO/DAPO) with shared initial states
- Simplifies individual trajectory tracking and debugging

**Why separate SimpleMemory and SearchMemory?**
- Different semantic meanings (observation/action vs search/information)
- Different formatting requirements for prompt templates
- Demonstrates extensibility pattern for custom environments

## Extension Patterns

### Adding Custom Memory Implementation

1. **Subclass BaseMemory** and implement required methods
2. **Custom `fetch()` logic** for specialized retrieval (e.g., attention-weighted, summarized)
3. **Register in EnvironmentManager** constructor
4. **Update prompt templates** to use formatted output

**Example** (hypothetical SummarizedMemory):
```python
class SummarizedMemory(BaseMemory):
    def fetch(self, history_length: int, obs_key: str, action_key: str):
        recent = self._data[env_idx][-history_length:]
        # Custom logic: summarize every 5 steps
        summary = self._summarize_batch(recent[::5])
        detailed = self._format_recent(recent[-3:])
        return [f"Summary: {summary}\nRecent: {detailed}"], [len(recent)]
```

### Adding Memory to New Environment

**Required steps**:
1. Instantiate memory in `__init__()`: `self.memory = SimpleMemory()`
2. Reset in `reset()`: `self.memory.reset(batch_size=len(obs))`
3. Store in `step()`: `self.memory.store({'text_obs': obs, 'action': actions})`
4. Fetch in `build_text_obs()`: `memory_ctx, _ = self.memory.fetch(...)`
5. Inject into prompt template

**Configuration**: Add `env.history_length` parameter to control sliding window size

## File Locations

- **Main Implementation**: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/memory.py`
- **Base Class**: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/base.py`
- **Module Init**: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/__init__.py`
- **Module README**: `/data1/zzq/rl-proj/verl-agent/agent_system/memory/README.md`
- **Integration Point**: `/data1/zzq/rl-proj/verl-agent/agent_system/environments/env_manager.py`
- **Prompt Templates**: `/data1/zzq/rl-proj/verl-agent/agent_system/environments/prompts/`
  - ALFWorld: `alfworld.py`
  - Search: `search.py`
  - Sokoban: `sokoban.py`
  - WebShop: `webshop.py`
  - AppWorld: `appworld.py`

## Key Insights

1. **Step-Independent Design**: Memory system enables verl-agent's core innovation - each step constructs input independently without concatenating full history, keeping context length constant.

2. **Configuration-Driven**: History length controlled via config rather than hardcoded, supporting different complexity/performance trade-offs per environment.

3. **Separation of Concerns**: Memory handles storage and retrieval, EnvironmentManager handles formatting and prompt injection, enabling independent evolution.

4. **Extensibility by Design**: Abstract base class and simple implementation encourage customization (summarization, selective retention, external knowledge) as noted in README.

5. **Production-Ready Simplicity**: Deliberately minimal implementation (100 lines) serves as robust starting point while remaining easy to understand and extend for research experimentation.
