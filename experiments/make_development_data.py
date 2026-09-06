"""Generate reproducible SYNTHETIC DEVELOPMENT DATA, never real model results."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

from zhiheng_router.data import read_config, validate_history
from zhiheng_router.schemas import HistoricalRecord, finite_number

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "1.0.0"
DATA_VERSION = "development-v1"

# Each item is a distinct source/template family. All variants stay in its group.
# Topics, instructions, and numeric inputs are intentionally separate from outcome noise.
TEMPLATES = {
    "code": [
        "用 Python 编写去除列表重复项的代码，保留原有顺序，输入有 {n} 项。",
        "为 {n} 条日志实现代码解析器，输出每个级别的出现次数。",
        "设计 Python 函数合并 {n} 个有序迭代器，说明接口。",
        "编程实现包含 {n} 个节点的目录树遍历，返回匹配文件。",
        "为一个容量为 {n} 的缓存编写 Python 代码，支持读取与更新。",
        "写 SQL 代码从 {n} 条订单中统计每位客户的消费。",
        "分析 Python 片段 `def f(x): return x[::-1]`，讨论 {n} 个输入样本。",
        "用代码实现 {n} 个事件的滑动窗口计数器，给出调用示例。",
        "为 {n} 行 CSV 数据编写 Python 字段校验函数。",
        "编程设计一个任务依赖检查器，处理 {n} 个任务。",
        "给出 Python 字符串匹配代码，文本由 {n} 个字符构成。",
        "实现 {n} 个对象的分组聚合代码，输出 JSON。",
    ],
    "math": [
        "求解方程 3x + {n} = 2x + {k}，给出计算过程。",
        "计算边长为 {n} 与 {k} 的矩形面积并解释单位。",
        "证明数列 a_i = {n} + i 的前 {k} 项求和公式。",
        "盒中有 {n} 个红球和 {k} 个蓝球，计算抽样概率。",
        "求函数 f(x) = x^2 + {n}x 的变化规律，在 x={k} 处讨论。",
        "某商品价格 {n} 元，经过 {k}% 的折扣，计算最终金额。",
        "计算 {n} 个观测值的平均数，其中一个值从 {k} 变为零。",
        "分析 {n} 个顶点构成的几何图形，给出对角线数量的证明。",
        "解一个初值为 {n}、增长参数为 {k} 的递推关系。",
        "比较两个分数 {n}/{k} 与 {k}/{n}，证明大小关系。",
        "求解 {n} 个资源在 {k} 个容器之间分配的计数问题。",
        "给定样本量 {n} 和成功次数 {k}，解释概率估计的不确定性。",
    ],
    "summary": [
        "总结会议记录：研发提出 {n} 项需求，运营保留 {k} 项，最终安排分批评审。",
        "概括研究摘要：观察 {n} 处样地，其中 {k} 处出现季节性变化，原因尚待确认。",
        "为新闻写摘要：公交新增 {n} 班次，覆盖 {k} 个站点，试运营后再评估。",
        "提炼产品访谈：{n} 人强调易用性，{k} 人关注价格，部分反馈存在冲突。",
        "总结项目周报：计划处理 {n} 个事项，完成 {k} 个，其余等待外部资料。",
        "概括调查记录：图书馆新增 {n} 本书，有 {k} 本需要重新分类。",
        "写出活动摘要：{n} 名参与者完成报名，{k} 人选择线上参与，现场人数待核实。",
        "提炼教学材料：课程包含 {n} 个练习，{k} 个例题展示常见误区。",
        "总结采购说明：比较 {n} 个方案，{k} 个满足接口要求，尚未决定供应商。",
        "概括维护报告：检查 {n} 台设备，发现 {k} 处异常，需要后续复查。",
        "为采访写摘要：{n} 位志愿者参与服务，累计整理 {k} 份材料。",
        "总结实验记录：重复实验 {n} 次，{k} 次出现偏差，不能直接断言因果。",
    ],
    "translation": [
        "翻译为英文：仓库收到 {n} 个包裹，其中 {k} 个需要检查。",
        "将英文译成中文：We reviewed {n} drafts and retained {k} alternatives.",
        "翻译通知为英文：本次课程有 {n} 人报名，分为 {k} 个小组。",
        "将产品说明译为中文：The device stores {n} entries across {k} categories.",
        "翻译邮件为英文：请在 {n} 日前核对附件中的 {k} 处修改。",
        "将旅游说明译成中文：The route spans {n} kilometers and passes {k} villages.",
        "翻译档案说明为英文：展览收录 {n} 件作品，来自 {k} 位作者。",
        "将操作提示译为中文：Keep {n} samples and label {k} of them for review.",
        "翻译访谈为英文：我们尝试了 {n} 种材料，仍有 {k} 个问题没有解决。",
        "将报告句子译成中文：Only {k} of the {n} observations support the hypothesis.",
        "翻译公告为英文：{n} 个座位中有 {k} 个暂时关闭，请遵循工作人员安排。",
        "将合同片段译为中文：Deliver {n} units in {k} batches, subject to inspection.",
    ],
    "qa": [
        "解释浏览器缓存的作用，面向 {n} 名读者给出 {k} 个例子。",
        "介绍植物蒸腾作用，结合 {n} 个观察点组织 {k} 个说明。",
        "说明数据库事务为什么有用，用 {k} 个场景解释 {n} 条记录的一致性。",
        "解释声音传播的基本条件，准备给 {n} 人的小组回答 {k} 个疑问。",
        "介绍地图比例尺如何使用，用 {n} 个位置和 {k} 个距离作为例子。",
        "解释文件备份与同步的区别，讨论 {n} 个文件在 {k} 台设备上的管理。",
        "说明团队如何安排任务，针对 {n} 项工作提出 {k} 条建议。",
        "介绍海洋潮汐的形成机制，为 {n} 名学生准备 {k} 个观察建议。",
        "解释网络请求从发出到响应的过程，假设有 {n} 个请求和 {k} 个节点。",
        "说明图书分类的目的，讨论 {n} 本图书如何放入 {k} 个书架。",
        "介绍材料热胀冷缩的现象，设计 {n} 次观察并提出 {k} 个注意点。",
        "解释版本管理为什么记录变更，为 {n} 名协作者给出 {k} 条实践建议。",
    ],
}
REQUIREMENTS = [
    "只需简短回答，使用一个直接例子。",
    "分步骤说明，补充一个容易忽略的限制。",
    "比较两种处理方式，解释适用条件并检查边界。",
    "综合多个限制进行分析，给出反例、边界情况以及可核查的推导。",
]
STYLES = ["使用简洁中文。", "面向初学者解释。", "保留必要的 English 术语。", "按条目组织答案。"]


def generate_records(config: dict) -> tuple[list[HistoricalRecord], dict]:
    settings = config["generator"]
    expected_ids = [f"dev-model-{letter}" for letter in "abcde"]
    if sorted(m["model_id"] for m in config["models"]) != expected_ids:
        raise ValueError("development generator requires exactly dev-model-a through dev-model-e")
    if sorted(settings["profiles"]) != expected_ids:
        raise ValueError("generator profiles must match all five development models")
    variants = settings["variants_per_group"]
    if type(variants) is not int or variants < 4:
        raise ValueError("variants_per_group must be an integer >= 4 to cover difficulty levels")
    for name in ("quality_noise_std", "query_shock_std", "group_effect_std", "latency_log_std", "cost_log_std"):
        finite_number(settings[name], name)
    rng = random.Random(settings["seed"])
    records = []
    for task_type, templates in TEMPLATES.items():
        for template_index, template in enumerate(templates):
            group_id = f"{task_type}-family-{template_index:02d}"
            group_effect = rng.gauss(0, settings["group_effect_std"])
            for variant in range(variants):
                # Difficulty has visible constraints plus unobserved continuous variation.
                level = variant % len(REQUIREMENTS)
                difficulty = min(1.0, max(0.0, (level + rng.uniform(0.05, 0.95)) / 4))
                n, k = 20 + variant * 13 + rng.randrange(10), rng.randrange(2, 18)
                query = (template.format(n=n, k=k) + REQUIREMENTS[level]
                         + STYLES[rng.randrange(len(STYLES))])
                query_id = f"{group_id}-q{variant:03d}"
                query_shock = rng.gauss(0, settings["query_shock_std"])
                # Shared shocks, model noise and nonlinear difficulty preclude deterministic labels.
                for model_id in expected_ids:
                    profile = settings["profiles"][model_id]
                    quality = (profile["ability"] + profile["task_bonus"][task_type]
                               - profile["difficulty_penalty"] * difficulty ** 1.35
                               + group_effect + query_shock
                               + rng.gauss(0, settings["quality_noise_std"]))
                    quality = round(min(1.0, max(0.0, quality)), 6)
                    # Costs depend on input/output effort and model price, never on sampled quality.
                    effort = 0.5 + len(query) / 220 + 0.45 * difficulty
                    cost = profile["base_cost"] * effort * math.exp(rng.gauss(0, settings["cost_log_std"]))
                    latency = (profile["base_latency_ms"] * (0.7 + len(query) / 500 + 0.6 * difficulty)
                               * math.exp(rng.gauss(0, settings["latency_log_std"])))
                    records.append(HistoricalRecord(
                        f"{query_id}--{model_id}", query_id, group_id, query, model_id,
                        quality, round(cost, 8), round(latency, 3), task_type,
                    ))
    validate_history(records, expected_ids, full_coverage=True)
    manifest = {
        "source_kind": "synthetic", "data_version": DATA_VERSION,
        "generator_version": GENERATOR_VERSION, "seed": settings["seed"],
        "quality_score_range": [0.0, 1.0], "cost_unit": "synthetic_cost_unit",
        "latency_unit": "ms", "coverage_kind": "full",
        "quality_definition": "Clipped synthetic utility from model/task ability, nonlinear difficulty, group/query shocks and model noise; not a real response evaluation.",
        "latency_definition": "Simulated complete model invocation duration; excludes router overhead.",
        "observation_count": len(records), "query_count": len(records) // len(expected_ids),
        "group_count": sum(len(t) for t in TEMPLATES.values()), "model_ids": expected_ids,
        "generator_config": settings,
        "notice": "SYNTHETIC DEVELOPMENT DATA. All model outcomes are simulated; no actual models were invoked.",
    }
    return records, manifest


def write_dataset(config: dict, root: Path = ROOT) -> dict:
    rows, manifest = generate_records(config)
    content = "".join(json.dumps(asdict(r), ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode("utf-8")
    manifest["sha256"] = hashlib.sha256(content).hexdigest()
    data_path, meta_path = root / config["data_path"], root / config["manifest_path"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(content)
    meta_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mvp.json")
    args = parser.parse_args()
    manifest = write_dataset(read_config(args.config))
    print(json.dumps({k: manifest[k] for k in ("source_kind", "observation_count", "query_count", "group_count", "sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
