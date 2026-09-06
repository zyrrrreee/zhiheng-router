# Zhiheng Router（智衡路由）

> 基于历史感知的大模型智能网管系统

## Router MVP 快速运行

当前已实现离线 Router MVP：共享字符 TF-IDF 和结构特征、每模型独立 Logistic Regression、训练集成本/时延均值、质量约束路由，以及四个 Baseline。

**当前数据全部为 SYNTHETIC DEVELOPMENT DATA。质量、模型调用成本和模型时延结果均为 simulated；未调用真实模型，概率不代表实际质量保证。**

推荐使用 Python 3.12（本地验证版本为 3.12.14）。在仓库根目录运行以下 PowerShell 命令；其他系统使用相应的虚拟环境激活命令。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python experiments/make_development_data.py
python experiments/run.py
python -m pytest -q
python demo.py
```

若系统默认 Python 不是 3.12，第一步请使用已安装的 Python 3.12 解释器。已创建虚拟环境时直接从激活步骤开始；也可不激活，使用 `.\.venv\Scripts\python.exe` 替代各命令中的 `python`。

参数位于 [configs/mvp.json](configs/mvp.json)。实验产生 `outputs/mvp/report.json` 和 `outputs/mvp/router.joblib`；Demo 读取产物中保存的配置，不重新训练。只加载本地生成、可信的 joblib 文件。

`quality_score_threshold` 定义训练标签，修改后必须重新训练；`router_probability_threshold` 和 `cost_tolerance_abs` 定义路由策略。后者默认零，严格执行质量门槛 → 最低成本 → 时延。Cost / Latency 目前是每模型固定训练均值。

详细的数据口径、接口边界和复现说明见 [Router MVP 使用与实现约定](docs/Router-MVP.md)。下文第 8 节目录树及早期阶段计划保留为长期规划，当前包实际位于 `src/zhiheng_router/`。

## 1. 项目简介

Zhiheng Router（智衡路由）是面向多大模型服务集群的智能路由与网管系统。

本项目源自中国国际大学生创新大赛（2026）产业赛道华为企业命题：

**《大模型智能网管——基于历史请求学习的模型路由分发系统》**

在企业同时部署多个大模型的场景下，不同模型在任务能力、响应质量、调用成本和服务时延方面存在明显差异。

传统方案通常采用：

- 固定调用能力最强的模型；
- 根据任务类别人工配置固定路由规则。

前者容易造成不必要的算力与调用成本，后者难以适应请求类型、模型能力和运行状态的动态变化。

本项目拟利用历史请求数据学习不同模型在各类任务上的性能表现，为新请求动态选择更合适的目标模型，在满足响应质量要求的前提下，进一步降低调用成本和服务时延。

---

## 2. 核心问题

假设系统包含 5～10 个候选大模型。

对于任意新请求：

query_new

系统需要根据：

- 请求内容与任务特征；
- 不同模型的历史响应质量；
- 模型调用成本；
- 模型服务时延；
- 模型擅长任务类型；
- 当前或历史模型性能状态；

预测不同模型处理该请求的效果，并选择目标模型：

Query
↓
Feature Extraction
↓
Model Performance Estimation
↓
Intelligent Router
↓
Selected Model

决策目标按照以下优先级：

1. 首先满足响应质量阈值；
2. 在质量达标的模型中尽量降低调用成本；
3. 在质量与成本接近时进一步降低响应时延。

---

## 3. 基础数据形式

命题给出的历史数据可以抽象为：

D = {
    query_i,
    model_j,
    quality_score_i,
    latency_i,
    cost_i
}

系统需要从历史数据中学习：

Request Features
↓
Expected Model Performance
↓
Routing Decision

---

## 4. 初步系统架构

User Query
    │
    ▼
Feature Extractor
    │
    ▼
Model Performance Predictor
    │
    ▼
Intelligent Router
    │
    ├── Model A
    ├── Model B
    ├── Model C
    └── ...
          │
          ▼
    Model Response
          │
          ▼
Quality / Cost / Latency Monitor
          │
          ▼
    History Database

后续系统计划进一步加入：

- 多模型统一调用接口；
- 历史请求数据库；
- Router 算法模块；
- 性能监控模块；
- Benchmark 与 Baseline；
- Web Dashboard；
- 鲲鹏 CPU 部署与性能优化。

---

## 5. Baseline

项目至少建立以下基线方法：

### Baseline 1：Strongest Model

所有请求固定发送给能力最强模型。

### Baseline 2：Lowest Cost Model

所有请求固定发送给成本最低模型。

### Baseline 3：Rule-based Router

根据预设任务类别进行固定模型路由。

### Proposed Method：History-aware Intelligent Router

根据历史请求数据学习模型性能，并进行动态路由。

主要评价指标包括：

- Quality Pass Rate
- Average Quality
- Average Cost
- End-to-End Latency
- Router Latency
- Throughput / QPS

---

## 6. 鲲鹏平台

最终 Router 系统需要部署于鲲鹏 CPU 环境。

后续计划研究：

- openEuler 环境部署；
- 多核并行；
- Router 推理性能优化；
- NEON / SIMD 加速；
- 单请求时延与吞吐性能测试。

当前阶段优先完成与硬件无关的算法、数据和系统原型开发。

---

## 7. 当前阶段目标

当前阶段优先完成：

项目理解
↓
技术方案设计
↓
数据结构定义
↓
Baseline
↓
Router MVP
↓
实验框架
↓
系统原型

第一阶段重点不是追求复杂算法，而是先建立一个：

**可运行、可测试、可比较、可迭代**

的完整系统。

---

## 8. 项目目录

zhiheng-router/
├── README.md
├── AGENTS.md
│
├── 竞赛相关规则文件/
├── 华为-路由分发系统命题.pdf
│
├── docs/
│   ├── 01-命题理解.md
│   ├── 02-技术方案.md
│   ├── 03-系统架构.md
│   ├── 04-实验设计.md
│   └── 05-推进记录.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── features/
│   ├── router/
│   ├── models/
│   ├── services/
│   └── monitoring/
│
├── baselines/
├── experiments/
├── tests/
└── web/

目录会随着项目推进逐步完善。

---

## 9. 项目状态

当前状态：

- [x] 完成企业命题选择
- [x] 完成产业赛道报名
- [x] 完成项目及团队创建
- [x] 完成项目基本需求理解
- [ ] 完成解决方案 V1
- [ ] 完成数据方案
- [x] 完成 Baseline（离线四基线）
- [x] 完成 Router MVP（synthetic development 范围）
- [ ] 完成多模型接入
- [ ] 完成鲲鹏部署
- [ ] 完成完整实验
- [ ] 完成 Web Demo

---

## 10. 项目原则

1. 先建立可运行 Baseline，再增加复杂算法。
2. 所有算法改进必须通过实验验证。
3. 不使用未经实验支持的性能提升数字。
4. 数据、算法、实验和系统代码尽量解耦。
5. 保持实验结果可复现。
6. 优先解决比赛命题明确要求的问题，避免无必要的技术堆叠。
