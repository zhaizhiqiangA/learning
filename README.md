# Analyze Skill - 代码分析技能

## 简介

这是一个专门用于深入分析代码实现原理的 skill，可以帮助你理解复杂的算法、数据流和系统架构。分析结果会自动保存为结构化的 markdown 文档，方便后续学习和回顾。

## 使用方法

### 基本用法

```bash
/analyze <topic>
```

### 示例

```bash
# 分析特定算法
/analyze PPO算法的ratio计算过程

# 分析特定文件中的实现
/analyze GRPO的优势函数计算 --file verl/trainer/ppo/core_algos.py

# 指定分析重点
/analyze 重要性采样 --focus 数学推导

# 分析配置文件
/analyze GRPO训练配置参数 --file examples/grpo_trainer/run_qwen3-8b.sh
```

## 输出位置

所有分析文档保存在此目录下，文件命名格式：`{主题}_分析.md`

## 已保存的分析

### 算法原理
- [verl重要性采样三个logprob分析](../verl重要性采样三个logprob分析.md) - 2026-02-05

---

**使用 `/analyze <topic>` 命令创建新的分析文档**
