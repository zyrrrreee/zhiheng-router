\# AGENTS.md



\## Project



Project name:



Zhiheng Router（智衡路由）



Chinese full name:



智衡路由——基于历史感知的大模型智能网管系统



This project is developed for the China International College Students' Innovation Competition 2026, Industry Track, Huawei enterprise challenge:



“大模型智能网管——基于历史请求学习的模型路由分发系统”



\---



\## Core Objective



Build a lightweight intelligent model router for a service cluster containing approximately 5–10 large language models.



Given a new user request, the system should use historical request data and model performance information to choose the most appropriate model.



Historical records contain approximately:



query

selected\_model

quality\_score

latency

cost



The routing objective has strict priority:



1\. Meet the required response quality threshold.

2\. Minimize model invocation cost among qualified models.

3\. Minimize latency when quality and cost are comparable.



The system must eventually run on a Kunpeng CPU server and should later investigate multicore parallelism and NEON/SIMD optimization.



\---



\## Current Development Stage



The project is currently in the early design and MVP stage.



Do NOT attempt to build the complete final competition system immediately.



Current priorities:



1\. Understand and document the problem.

2\. Define clean data schemas and module interfaces.

3\. Build reproducible baselines.

4\. Implement a minimal end-to-end Router MVP.

5\. Establish evaluation infrastructure.

6\. Keep interfaces extensible for future routing algorithms.

7\. Only later add Web UI, Huawei Cloud deployment and Kunpeng optimization.



\---



\## Engineering Principles



Prefer simple, testable implementations over unnecessary complexity.



Do not introduce technologies merely because they appear advanced.



Avoid unnecessary use of:



\- distributed microservices

\- Kubernetes

\- Kafka

\- Redis

\- complex agent frameworks

\- reinforcement learning

\- large neural networks



unless there is a demonstrated project requirement.



Start with the simplest implementation capable of establishing a reliable baseline.



\---



\## Architecture



The expected logical pipeline is:



Query

↓

Feature Extraction

↓

Model Performance Estimation

↓

Router Decision

↓

Model Invocation

↓

Quality / Cost / Latency Measurement

↓

History Storage



Modules should remain reasonably independent.



Expected major components:



src/features

src/router

src/models

src/services

src/monitoring

baselines

experiments

tests

web



\---



\## Baselines



Before implementing advanced routing algorithms, the project must provide at least:



1\. strongest-model routing

2\. lowest-cost-model routing

3\. rule-based routing



Every proposed intelligent router should be compared against these baselines.



\---



\## Evaluation



Primary evaluation metrics include:



quality\_pass\_rate

average\_quality

average\_cost

average\_latency

router\_latency

throughput



Experiments must be reproducible.



Do not report fabricated or estimated improvements as experimental results.



Clearly distinguish:



\- measured results

\- simulated results

\- planned targets



\---



\## Coding Style



Primary language:



Python



Prefer:



\- clear type annotations

\- small modules

\- dataclasses / typed schemas where appropriate

\- deterministic experiment configuration

\- structured logging

\- unit tests for core routing logic

\- config files instead of hard-coded experimental parameters



Avoid large monolithic scripts.



\---



\## Documentation



Important design decisions should be documented under docs/.



Before major implementation changes, update or consult:



README.md

docs/01-命题理解.md

docs/02-技术方案.md

docs/03-系统架构.md

docs/04-实验设计.md



When implementation and documentation conflict, explicitly identify the inconsistency instead of silently choosing one.



\---



\## Competition Constraints



The project must remain aligned with the Huawei challenge.



Do not silently redefine the task.



Core requirements that must remain visible throughout development:



\- historical-request-based routing

\- lightweight router

\- 5–10 candidate models

\- quality constraint

\- cost optimization

\- latency optimization

\- Kunpeng CPU deployment

\- multicore / NEON-related performance optimization

\- feature design

\- router architecture

\- core algorithm description



\---



\## Agent Working Rules



Before modifying code:



1\. Inspect the relevant repository files.

2\. Explain the intended change briefly.

3\. Prefer minimal changes.

4\. Do not refactor unrelated code.

5\. Preserve reproducibility.

6\. Add or update tests when changing core routing behavior.

7\. Do not invent unavailable Huawei APIs, datasets, server resources or competition requirements.



If a requirement is unclear, mark it as an open question instead of assuming an answer.

