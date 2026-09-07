"""Small, frame-specific payload variants and semantic evidence for Pilot v3."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .payloads_v2 import ContentPayload, generate_content_payload
from .rng import NamespaceRNG
from .semantics_v3 import FRAME_CAPABILITIES, payload_variant_index
from .surface import SemanticComposition


@dataclass(frozen=True)
class VariantText:
    zh: str
    en: str


V = VariantText
FRAME_VARIANTS: dict[str, tuple[VariantText, VariantText]] = {
    "frame-code-scheduler": (
        V("B 与 C 共享独占数据库锁，不能放在同一执行批次。", "B and C share an exclusive database lock and cannot run in the same batch."),
        V("B 与 C 共享独占锁；若 C 首次失败，可重试一次，重试结束前 E 不得启动。", "B and C share an exclusive lock; if C first fails, it may retry once, and E must wait for that retry."),
    ),
    "frame-code-service-debug": (
        V("同一 request_id 随后以 retry=false 再调用一次，预期不产生新写入。", "The same request_id is then called with retry=false and must not create another write."),
        V("写入 retry 键时可能抛出异常；主键已写入的半完成状态必须能够安全恢复。", "Writing the retry key may fail after the primary write; recovery must handle that partial state safely."),
    ),
    "frame-code-sql-report": (
        V("另有一笔 o2 退款在 9 月 1 日获批，不应计入 8 月净额。", "A refund for o2 approved on September 1 must not affect the August net amount."),
        V("`refund_status(refund_id,status)` 将 r2 标为 reversed；只有 approved 退款可以扣减。", "`refund_status(refund_id,status)` marks r2 as reversed; only approved refunds may reduce payment."),
    ),
    "frame-code-api-client": (
        V("服务端可能返回非空页面却重复同一个 next_cursor，客户端必须终止游标循环。", "A nonempty page may repeat the same next_cursor, and the client must stop that cursor loop."),
        V("HTTP 429 的 Retry-After header 与响应体不一致时以 header 为准，且整个请求最多重试两次。", "When an HTTP 429 Retry-After header conflicts with the body, the header wins and the request has two retries total."),
    ),
    "frame-code-refactor": (
        V("rows 也可能是只能遍历一次的 iterator，重构后不得提前耗尽它。", "rows may be a single-pass iterator, which the refactoring must not consume early."),
        V("某行 value 无法转为整数时要记录该行错误并继续处理其余有效行，同时保持有效行顺序。", "If one value cannot be parsed as an integer, record that row error, continue valid rows, and preserve their order."),
    ),
    "frame-code-repository-review": (
        V("测试只 reload service.py 而不 reload cache.py，错误 TTL 仍然存在。", "The test reloads service.py but not cache.py, so the stale TTL remains."),
        V("两个 worker 分别在设置环境变量前后 import 模块，当前会得到不同 TTL；修复后必须显式共享同一配置。", "Two workers import before and after the environment change and currently get different TTLs; the fix must pass one explicit configuration."),
    ),
    "frame-code-data-pipeline": (
        V("来源 B 的时间带 `+08:00`，换算 UTC 后可能落到前一天，去重键使用 UTC 日期。", "Source B includes a +08:00 timestamp that may fall on the prior UTC date; deduplication uses the UTC date."),
        V("来源 A 的 value 缺失时允许来源 B 补值，但日期与标识仍以来源 A 为准，并记录字段级 provenance。", "When source A lacks value, source B may fill it, while A still owns date and ID and field-level provenance is recorded."),
    ),
    "frame-code-performance": (
        V("同一个元素不能使用两次；target=20 且 values=[10,3,10] 时必须返回 True。", "One element cannot be reused; target=20 with values=[10,3,10] must return True."),
        V("输入改为单遍 iterator，内存上限不足以保存全部元素；方案需要说明精确性与外部存储的取舍。", "The input is a single-pass iterator larger than memory; the design must state the exactness and external-storage tradeoff."),
    ),
    "frame-math-equation-system": (
        V("另外要求 x、y 为非负整数且 x≥y，核验解是否满足库存规则。", "Also require nonnegative integer x and y with x at least y, and verify the inventory rule."),
        V("审计要求分别用消元法和矩阵法求解，并说明系数行列式为何保证结果唯一。", "The audit requires both elimination and matrix solutions, plus an explanation of why the coefficient determinant guarantees uniqueness."),
    ),
    "frame-math-probability": (
        V("第一次抽取后，观察者在第二次抽取前额外移走 1 个蓝球。", "After the first draw, an observer removes one blue ball before the second draw."),
        V("第一次抽到的是红球；随后抛公平硬币决定是否放回该红球，但硬币结果不可见，求综合条件概率。", "After the known red first draw, a fair hidden coin decides whether that ball is replaced; find the combined conditional probability."),
    ),
    "frame-math-proof": (
        V("在证明通项后，再推导前 n 项和并用 n=3 核验。", "After proving the term formula, derive the first-n-term sum and check n=3."),
        V("从 n=5 起递推增量改为原来的两倍；判断原命题在哪个范围成立，并给出首个反例。", "From n=5 onward the recurrence increment doubles; identify the valid range of the original claim and its first counterexample."),
    ),
    "frame-math-statistics": (
        V("校准记录表明告警值若有效应减去 2，再比较校准后的均值。", "If the alert value is valid, calibration subtracts 2; also compare the calibrated means."),
        V("告警值可能有效、应校准或应删除；分别计算三种情形并给出不依赖其状态的结论。", "The alert may be valid, calibrated, or removed; calculate all three cases and state conclusions invariant to its status."),
    ),
    "frame-math-geometry": (
        V("同时计算该 L 形区域的外周长，不计挖空区域内部未暴露的边。", "Also calculate the outer perimeter, excluding unexposed edges inside the removed corner."),
        V("左下角还挖去一个与原挖空不重叠的 1m × 2m 矩形，重新计算剩余面积。", "A nonoverlapping 1 m by 2 m rectangle is also removed from the lower-left corner; recompute area."),
    ),
    "frame-math-optimization": (
        V("产品 B 最多生产 4 件，把该上限加入整数模型。", "At most four units of product B may be made; add that bound to the integer model."),
        V("另有材料约束：A 每件耗材 1，B 每件耗材 2，总耗材不超过 12；同时满足工时与耗材约束。", "A second resource limits material: A uses 1, B uses 2, and 12 units are available; satisfy both resource constraints."),
    ),
    "frame-math-word-problem": (
        V("售票结束后最后 2 张成人票每张返还 1 元，另算实际净收入。", "After sales, the final two adult tickets receive a refund of 1 each; also compute net revenue."),
        V("售出后退回 1 张成人票，原始收入不变但现金净额需扣除票价；分别报告售出数和最终持票数。", "One adult ticket is returned after sale; recorded revenue stays original but cash net loses its price. Report sold and retained counts."),
    ),
    "frame-math-chart-analysis": (
        V("复核又发现 Q3 有 5 个单位重复记录，应直接删除后再计算修订增长率。", "Review finds five duplicate Q3 units that must be removed before recomputing revised growth."),
        V("除回溯调整外，Q2 有 4 个单位被误记到 Q1；完成两项修订后再与原增长率比较。", "Besides the retrospective adjustment, four Q2 units were assigned to Q1; apply both corrections before comparison."),
    ),
    "frame-qa-technical-docs": (
        V("[材料D] 日志没有路径；路由表显示 job_type=batch 映射批处理接口，本次 job_type 为 batch。", "[Evidence D] The log lacks a path; the routing table maps job_type=batch to the batch endpoint, and this job records batch."),
        V("[材料D] 网关路径写作 `/login`，但文档规定重试规则以 operation=batch 字段为准；本次 operation 为 batch。", "[Evidence D] The gateway path says `/login`, but the policy uses operation=batch to select retry rules, and this operation is batch."),
    ),
    "frame-qa-evidence-chain": (
        V("[材料D] 窗口负责人随后给出仅限只读流量的确认，写入部署仍未确认。", "[Evidence D] The window owner later approves read-only traffic only; write deployment remains unapproved."),
        V("[材料D] 安全审批在周一到期，而入队记录发生在周二；续期尚无记录。", "[Evidence D] Security approval expired Monday, before Tuesday queue entry, and no renewal is recorded."),
    ),
    "frame-qa-ambiguous-policy": (
        V("[材料D] 另一章节把“最近”用于 14 天窗口，但本页没有交叉引用该章节。", "[Evidence D] Another chapter uses recent for a 14-day window, but this page does not reference that chapter."),
        V("[材料D] 修订记录明确本条款的“最近”为 30 天，并注明该定义自本月一日起生效。", "[Evidence D] A revision defines recent as 30 days for this clause, effective from the first of this month."),
    ),
    "frame-qa-source-comparison": (
        V("[材料D] 两张表换算到 UTC 后，昨日 00:05 的记录落在不同统计日。", "[Evidence D] After UTC conversion, the 00:05 event falls into different reporting days in the two tables."),
        V("[材料D] 技术表还包含一次重放成功，运营表按 request_id 去重后不计该条。", "[Evidence D] Engineering includes one replay success that operations removes by request_id deduplication."),
    ),
    "frame-qa-troubleshooting": (
        V("[材料D] schema 在 mode 变更前 5 分钟已升级，但当时没有流量。", "[Evidence D] The schema changed five minutes before mode, when the service had no traffic."),
        V("[材料D] 用旧 schema 重放同一失败请求成功，用新 schema 重放仍失败。", "[Evidence D] Replaying the same request succeeds with the old schema and still fails with the new schema."),
    ),
    "frame-qa-data-explanation": (
        V("[材料D] 计费报表仍包含 suspended 账户，因此不能直接用仪表盘核对账单。", "[Evidence D] Billing still includes suspended accounts, so the dashboard cannot directly reconcile invoices."),
        V("[材料D] deleted 账户不在两者中；审计问题只针对 active 与 suspended 的范围差异。", "[Evidence D] Deleted accounts appear in neither source; the audit concerns only the active-versus-suspended scope."),
    ),
    "frame-qa-long-policy": (
        V("[材料E] 新修订要求故障至少持续 4 小时，本次修订在截止日前生效。", "[Evidence E] A revision effective before the deadline requires an outage of at least four hours."),
        V("[材料E] 补交还需在系统恢复后的下一个工作日 17:00 前提交；记录显示提交时间为 16:40。", "[Evidence E] Late submission must arrive by 17:00 on the next business day; the record shows 16:40."),
    ),
    "frame-qa-bilingual-terms": (
        V("[材料D] 当前 glossary 明确禁止把 service window 译作“客服窗口”。", "[Evidence D] The current glossary explicitly forbids translating service window as 客服窗口."),
        V("[材料D] 旧工单仍使用“客服窗口”，但页脚标注它引用的是旧版 glossary。", "[Evidence D] An old ticket still uses 客服窗口, but its footer identifies the retired glossary."),
    ),
    "frame-summary-meeting": (
        V("移动端测试后来通过，但缓存告警仍未完成，发布条件因此只满足一部分。", "Mobile testing later passes, but the cache alert remains incomplete, so release conditions are only partly met."),
        V("安全负责人要求把只读发布推迟到告警和兼容性同时通过；产品负责人尚未接受该变更。", "Security asks to delay read-only release until both alerting and compatibility pass; product has not accepted the change."),
    ),
    "frame-summary-research": (
        V("预设亚组没有改善，而探索性亚组显示正向结果；两者必须分开报告。", "The prespecified subgroup shows no improvement while an exploratory subgroup is positive; report them separately."),
        V("注册方案把主要指标定义为第 30 天结果，正文却报告第 14 天结果；作者尚未解释偏离。", "The registry defines the primary endpoint at day 30, while the paper reports day 14 without explaining the deviation."),
    ),
    "frame-summary-incident": (
        V("回滚后读取恢复，但写入直到连接池修复才恢复，影响范围需要分阶段描述。", "Reads recovered after rollback, while writes recovered only after the pool fix; describe impact in phases."),
        V("客户端时钟比服务端快 3 分钟；最终时间线必须统一到服务端时间并标注换算。", "Client clocks are three minutes ahead of the server; normalize the timeline to server time and mark conversions."),
    ),
    "frame-summary-metrics": (
        V("修订说明指出退款值的“万元”标签有误，原始导出实际以千元计；摘要需保留该更正。", "A revision says an earlier refund label incorrectly read 'ten-thousands'; the raw export and figures shown here use thousands. Retain that correction."),
        V("活跃用户表后来补发自然月口径版本；摘要需并列原 31 天滚动口径与新自然月口径。", "A calendar-month active-user table is later issued; compare it with the original rolling 31-day measure."),
    ),
    "frame-summary-conflicting-reports": (
        V("报告B后来说明完成率只统计已排期功能，报告A仍未说明分母。", "Report B later says its rate covers scheduled features only; Report A still lacks a denominator."),
        V("独立审计确认接口变更与环境晚到都发生，但无法量化各自造成的延期天数。", "An independent audit confirms both causes occurred but cannot quantify each cause's delay."),
    ),
    "frame-summary-technical-guide": (
        V("Windows 安装不支持默认 shell 脚本，必须改用 `router.ps1 validate`。", "Windows does not support the default shell script and must use `router.ps1 validate`."),
        V("回滚命令只有在迁移前备份校验通过时可用；备份失败时必须停止升级。", "Rollback is available only after backup verification; a failed backup must stop the upgrade."),
    ),
    "frame-summary-bilingual-report": (
        V("英文更新说明重复 ID 只影响测试环境，中文版本没有环境范围说明。", "The English update limits duplicate IDs to testing; the Chinese version omits the environment scope."),
        V("中文版本的时间戳晚于英文版，但英文页脚自称最终版；版本优先级尚未确定。", "The Chinese timestamp is later, while the English footer calls itself final; precedence remains unresolved."),
    ),
    "frame-summary-executive": (
        V("接口权限风险已有负责人，但完成日期仍取决于外部审批。", "The permission risk has an owner, but completion still depends on external approval."),
        V("扩大试点的批准附带预算上限；当前方案超过上限，是否缩减范围尚未决定。", "Expansion has a budget cap that the current plan exceeds, and scope reduction remains undecided."),
    ),
    "frame-translation-technical": (
        V("`--keep-data` 仅保留用户数据，不保留临时 cache；译文必须保留这一否定范围。", "`--keep-data` preserves user data but not temporary cache; the translation must retain that negative scope."),
        V("正文把恢复动作称为 restore，但命令字面量仍是 `router rollback`；译文要区分概念名称与命令。", "The prose calls the recovery action restore, while the literal command remains `router rollback`; distinguish the concept from the command."),
    ),
    "frame-translation-contract": (
        V("“工作日”不含交付地法定假日，但是否包含周六由附件定义，当前附件缺失。", "Business day excludes public holidays at the delivery location; whether Saturday counts is defined in a missing annex."),
        V("若发生延期，“该方”承担额外费用；“该方”可能指供应商或数据提供方。", "If delay occurs, that party bears the extra cost; that party may mean the supplier or the data provider."),
    ),
    "frame-translation-notice": (
        V("所有时间均为 UTC+8；海外团队不得把 22:00 误作本地时间。", "All times are UTC+8; overseas teams must not interpret 22:00 as local time."),
        V("若维护前 30 分钟错误率超过 5%，窗口取消；取消通知只通过 status page 发布。", "If errors exceed 5% in the 30 minutes before maintenance, the window is canceled through the status page only."),
    ),
    "frame-translation-domain-report": (
        V("附表中的 HV node 专指高压监测点，不是 compute node。", "In the table, HV node specifically means a high-voltage monitoring point, not a compute node."),
        V("术语 baseline 在方法部分指基线电压，在结果部分指对照模型；译文必须按章节区分。", "Baseline means reference voltage in Methods and control model in Results; translate it by section."),
    ),
    "frame-translation-ui-localization": (
        V("复数消息 `{{count}} retries remaining` 必须保留 count 占位符。", "The plural message `{{count}} retries remaining` must preserve the count placeholder."),
        V("按钮标签上限为 12 个字符，错误详情不受该限制；不要为缩短按钮而改写错误含义。", "Button labels have a 12-character limit, but error details do not; shortening a button must not alter the error meaning."),
    ),
    "frame-translation-incident": (
        V("聊天时间使用 UTC，工单使用 UTC+8；合并前先统一时区。", "Chat timestamps use UTC while the ticket uses UTC+8; normalize time zones before merging."),
        V("修复说明称连接池为 confirmed cause，较早工单仅称 hypothesis；译文必须保留来源与确定性变化。", "The repair note calls the pool a confirmed cause, while the earlier ticket says hypothesis; preserve source and certainty changes."),
    ),
    "frame-translation-ambiguous-reference": (
        V("后一句“前者批准后再执行”中的“前者”也可能指网关或 worker。", "In the next sentence, the former must approve before execution, but former may still mean the gateway or worker."),
        V("管理员告诉网关由 worker 更新它的证书；“它的”又可能指管理员、网关或 worker。", "An administrator tells the gateway that the worker will update its certificate; its may refer to any of the three."),
    ),
    "frame-translation-glossary": (
        V("产品名 `Service Window Pro` 不翻译，普通 service window 仍按术语表处理。", "The product name `Service Window Pro` stays untranslated, while ordinary service window follows the glossary."),
        V("新版术语表要求 rollback 作名词时译为“回退点”，作动词时仍译为“回滚”。", "The new glossary translates rollback as 回退点 when it is a noun and 回滚 when it is a verb."),
    ),
}


SC_FORMATS = {
    "frame-code-scheduler": "fields: execution batches / dependency errors / retry boundary",
    "frame-code-service-debug": "sections: root cause / patch / regression tests",
    "frame-code-sql-report": "columns: customer_id / gross / refund / net",
    "frame-code-api-client": "fields: response / error / retry decision",
    "frame-code-refactor": "sections: preserved behavior / change / tests",
    "frame-code-data-pipeline": "columns: output / error / provenance",
    "frame-code-performance": "sections: algorithm / complexity / boundary cases",
    "frame-math-equation-system": "sections: setup / solution / verification",
    "frame-math-probability": "sections: events / probability fraction / verification",
    "frame-math-proof": "sections: claim / base case / induction step",
    "frame-math-geometry": "sections: whole area / cutouts / result",
    "frame-math-optimization": "sections: variables / constraints / optimum",
    "frame-qa-technical-docs": "table columns: claim / evidence / conclusion",
    "frame-qa-ambiguous-policy": "table columns: fact / interpretation / clarification needed",
    "frame-qa-troubleshooting": "table columns: observation / hypothesis / test",
    "frame-qa-bilingual-terms": "table columns: term / source / reason",
    "frame-summary-meeting": "sections: decisions / owners / disagreements / open items",
    "frame-summary-research": "sections: finding / uncertainty / limitations / follow-up",
    "frame-summary-technical-guide": "sections: prerequisite / command / rollback",
    "frame-summary-executive": "sections: conclusion / risk / owner / next action",
    "frame-translation-technical": "columns: source / translation / terminology note",
    "frame-translation-contract": "columns: clause / source / translation / note",
    "frame-translation-notice": "columns: notice item / source / translation",
    "frame-translation-ui-localization": "columns: key / source / translation",
    "frame-translation-glossary": "columns: term / context / translation",
}


_PATTERNS = {
    "AR": r"依赖|算法|推导|证明|归纳|优化|调度|双指针|哈希|方程|分页|矩形|面积|dependency|algorithm|derive|proof|induction|optimi[sz]|schedule|two pointer|hash|equation|pagination|rectangle|area|loop",
    "SQ": r"\d|[=+×/%≥]|概率|面积|均值|利润|收入|总数|方程|probability|area|mean|profit|revenue|total|equation",
    "DE": r"失败|错误|日志|异常|回滚|重复|超时|错误值|fail|error|log|exception|rollback|duplicate|timeout|stale|wrong",
    "DS": r"schema|sql|表|数据|字段|行|聚合|连接|来源|指标|观测|样本|table|data|field|row|aggregate|join|source|metric|dashboard|sample",
    "AU": r"api|http|接口|endpoint|配置|config|命令|command|函数|function|cursor|retry|按钮|button|placeholder|protocol|schema|router\.yaml|router rollback|source a|来源 a|pipeline",
    "FK": r"制度|文档|定义|规则|证据|材料|手册|报告|术语表|结论|风险|决定|policy|document|definition|rule|evidence|manual|report|glossary|finding|risk|decision",
    "FT": r"保留|保持|不得|不改变|不补写|不超出|忠实|统一|转换|修订|回溯|原始|事实|数值|例外|待验证|确定性|限制|尚未批准|摘要|总结|提炼|压缩|综合|preserv|retain|without changing|faithful|normalize|revise|reassign|original|fact|exception|pending|certainty|limit|unapproved|verifiable|directly compar|summary|summari[sz]|synthesi[sz]",
    "MT": r"翻译|译文|目标语言|中英文|术语|本地化|glossary|translat|bilingual|terminology|locali[sz]",
    "AM": r"歧义|可能指|所指|无法唯一|未定义|没有定义|口径没有|范围说明|是否|优先级|全部账户|只显示|冲突|不可直接比较|ambig|referent|may mean|may refer|cannot be resolved|does not define|whether|not yet validated|conflict|precedence|scope (?:is )?(?:missing|omitted|different)|all accounts|only active|not be directly compared",
}


def _segments(payload: ContentPayload) -> list[str]:
    return [segment.strip() for segment in (
        *payload.material.splitlines(), payload.task_instruction, payload.output_contract,
    ) if len(segment.strip()) >= 3 and not segment.strip().startswith("```")]


def _capability_is_aligned(capability: str, composition: SemanticComposition,
                           payload: ContentPayload) -> bool:
    text = "\n".join((payload.material, payload.task_instruction, payload.output_contract))
    if capability not in FRAME_CAPABILITIES[composition.frame.frame_id]:
        return False
    if capability == "MH":
        return payload.passage_count >= 2 and bool(re.search(
            r"依赖|条件|约束|之后|先|链|修订|回滚|比较|综合|验证|收入|冲突|评估|只有|需要|例外|定义|记录|不同|原因|窗口|第一次|第二次|depends|condition|constraint|after|before|chain|revision|rollback|step|compare|integrate|verify|revenue|conflict|assess|requires|needs|only|exception|definition|record|differ|different|cause|window|first|second",
            text, re.IGNORECASE,
        )) or (composition.frame.frame_id.endswith("translation-incident")
               and len(re.findall(r"\d{2}:\d{2}", text)) >= 3) or (
                   composition.frame.frame_id == "frame-math-word-problem"
                   and bool(re.search(
                       r"总数.*收入|共.*总收入|sold.*total revenue|total.*revenue",
                       text, re.IGNORECASE,
                   ))
               )
    if capability == "LC":
        return payload.passage_count >= 3 or bool(re.search(
            r"多段|跨|附录|章节|模块|版本|来源|timeline|appendix|chapter|module|version|source|bilingual",
            text, re.IGNORECASE,
        ))
    if capability == "SC":
        return bool(re.search(
            r"json|schema|字段|列|表|签名|函数|小节|格式|证明|步骤|清单|时间线|table|field|column|function|section|format|proof|checklist|timeline|matrix|contract|resource|guide",
            payload.output_contract, re.IGNORECASE,
        ))
    if capability == "MT" and composition.frame.task == "translation":
        return bool(re.search(r"目标语言|翻译|译文|本地化|translat|locali[sz]|glossary|术语", text, re.IGNORECASE))
    if capability == "FT" and composition.frame.task in {"summary", "translation"}:
        return bool(re.search(_PATTERNS["FT"], text, re.IGNORECASE)) and bool(re.search(r"\d", payload.material))
    return bool(re.search(_PATTERNS[capability], text, re.IGNORECASE))


def build_alignment_evidence(composition: SemanticComposition,
                             payload: ContentPayload) -> tuple[tuple[str, str], ...]:
    segments = _segments(payload)
    evidence: dict[str, str] = {}
    for capability in composition.active_capabilities:
        if not _capability_is_aligned(capability, composition, payload):
            raise ValueError(
                f"invalid capability realization: {composition.frame.frame_id}/{capability}")
        if capability in {"MH", "LC"}:
            selected = payload.material
        elif capability == "SC":
            selected = payload.output_contract
        else:
            pattern = _PATTERNS[capability]
            selected = next(
                (segment for segment in segments if re.search(pattern, segment, re.IGNORECASE)),
                payload.task_instruction,
            )
        if selected not in "\n".join((payload.material, payload.task_instruction,
                                       payload.output_contract)):
            raise ValueError("alignment evidence must be physically present")
        evidence[capability] = selected
    return tuple(sorted(evidence.items()))


def generate_content_payload_v3(composition: SemanticComposition,
                                rng: NamespaceRNG) -> ContentPayload:
    payload = generate_content_payload(composition, rng)
    variant = payload_variant_index(composition.stable_id)
    if variant:
        addition = FRAME_VARIANTS[composition.frame.frame_id][variant - 1]
        if composition.frame.task == "translation":
            if composition.translation_direction == "zh_to_en":
                suffix = addition.zh
            elif composition.translation_direction == "en_to_zh":
                suffix = addition.en
            else:
                suffix = f"[中文补充]\n{addition.zh}\n[English addendum]\n{addition.en}"
            passage_increment = 2 if composition.translation_direction == "bilingual_revision" else 0
        elif composition.language == "en":
            label = "Evidence D" if composition.frame.task == "qa" else "Appendix 5"
            prefix = f"[{label}] " if composition.frame.task in {"qa", "summary"} else "Additional condition: "
            if addition.en.startswith("["):
                prefix = ""
            suffix = prefix + addition.en
            passage_increment = 1 if composition.frame.task in {"qa", "summary"} else 0
        else:
            label = "材料D" if composition.frame.task == "qa" else "附录5"
            prefix = f"[{label}] " if composition.frame.task in {"qa", "summary"} else "补充条件："
            if addition.zh.startswith("["):
                prefix = ""
            suffix = prefix + addition.zh
            passage_increment = 1 if composition.frame.task in {"qa", "summary"} else 0
        payload = replace(
            payload,
            material=f"{payload.material}\n{suffix}",
            structural_markers=tuple(sorted({
                *payload.structural_markers, f"payload_variant:v3-{variant}",
            })),
            passage_count=payload.passage_count + passage_increment,
        )
    else:
        payload = replace(
            payload,
            structural_markers=tuple(sorted({
                *payload.structural_markers, "payload_variant:v3-0",
            })),
        )
        if composition.frame.frame_id == "frame-summary-bilingual-report":
            addition = (
                "[Note 5] Neither language version says whether “current version” "
                "refers to the same build."
                if composition.language == "en"
                else "[说明5] 两种语言材料都没有说明“当前版本”是否指同一次构建。"
            )
            payload = replace(
                payload,
                material=f"{payload.material}\n{addition}",
                passage_count=payload.passage_count + 1,
            )
    if composition.frame.frame_id == "frame-math-proof" and variant == 2:
        instruction = (
            "Identify where the original claim remains valid, prove that range by "
            "induction, and give the first counterexample."
            if composition.language == "en"
            else "判断原命题在哪个范围成立；对成立范围给出归纳证明，并写出首个反例。"
        )
        payload = replace(payload, task_instruction=instruction)
    if "SC" in composition.active_capabilities and not _capability_is_aligned(
            "SC", composition, payload):
        suffix = SC_FORMATS[composition.frame.frame_id]
        payload = replace(payload, output_contract=f"{payload.output_contract}; {suffix}")
    return replace(
        payload,
        capability_evidence=build_alignment_evidence(composition, payload),
    )
