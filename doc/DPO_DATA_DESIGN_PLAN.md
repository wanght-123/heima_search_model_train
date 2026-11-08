# 招聘领域 DPO 数据设计与 Codex Direct v2 发布说明

## 1. 当前落地版本

当前可训练版本为 **Codex Direct v2**，共1000条，拆分为900条训练数据和100条 DPO 评测数据。

1. 输入来自项目 SFT 原始语料的 `input` 和项目 taxonomy 合成输入。
2. **不读取、不复用 SFT 历史 `output`**。
3. chosen 和 rejected 都由 Codex 基于项目提示词、候选集和指定错误类型直接生成。
4. 职位类别唯一取自 `job_cate.json`；其他任务候选唯一取自 `code/data/task_info/label_prompts_v2.md`。
5. 训练输入与冻结测试集已做完全隔离。

当前文件、质检报告和数据生成复现信息见
`code/data/datasets/dpo/codex_direct_v2_full1000/README.md`；可直接执行的 DPO 微调步骤见
[`DPO_FINE_TUNING.md`](DPO_FINE_TUNING.md)。

## 2. 构造边界

### 2.1 明确禁止的数据来源

- 不使用现有 `tasks` / `tasks_fixed` 中的 SFT `output`。
- 不使用 frozen test、diagnostic 或评测报告中的具体样本。
- 不把 SFT 现有错误直接当 rejected。
- 不对现有 chosen 做删词、扩边、换类或改字段以生成新 pair。
- 脚本不产生 chosen/rejected 语义答案。

### 2.2 脚本可以做什么

脚本只允许执行无语义创造的机械工作：

- JSON 可解析性、schema 和字段类型校验。
- taxonomy 路径存在性校验。
- 原文 substring 、哈希去重、训测集隔离和数量统计。
- 格式转换为 LLaMAFactory ShareGPT ranking JSONL。
- 从 SFT 原始语料中只选择 `input` 和 `source_id`，并分配任务/错误类型配额。

脚本不能决定哪个边界更好、哪个类别正确，也不能生成 rejected。

## 3. 后续扩展参考：纯合成多阶段方法

> 本节及后续的 truth ledger 方法是后续纯合成扩量的设计参考，不是当前 Codex Direct v2 的数据来源描述。与第 1–2 节冲突时，以当前落地版本说明为准。

每条 DPO 数据不直接从 prompt 开始写，而是先生成一份不进入训练集的 **真值账本（truth ledger）**。

```json
{
  "case_id": "SYN_EXT_BND_000001",
  "task": "entity_extraction",
  "target_error": "BND_SHORT",
  "difficulty": "hard",
  "scene": "跨境电商推荐算法岗位",
  "facts_to_express": [],
  "facts_to_omit": [],
  "evidence_spans": {},
  "decision_rule": "保留业务域+技术对象，删除结果套话",
  "chosen_claims": [],
  "rejected_error": "删除跨境电商和推荐排序，仅保留模型优化",
  "ambiguity_check": "chosen 在业务规则下可唯一胜出"
}
```

然后分五个隔离阶段完成：

1. **场景设计**：大模型先设计职位、行业、上下文线索、干扰信息和目标错误，不生成答案。
2. **真值建模**：列出该场景中应出现、不应出现、可以标准化和禁止推断的信息。
3. **chosen 生成**：基于真值账本撰写招聘文本和最终正确 JSON，再回填每个非空值的证据位置。
4. **对抗 rejected 生成**：换一个视角，模拟能力较强但对业务规则掌握不稳的模型，只注入一个主错误。
5. **独立裁判**：不查看原生成理由，仅根据 prompt、schema 和 taxonomy 比较 chosen/rejected。无法稳定判定时整条废弃。

这五个阶段都使用大模型能力，但不在同一次生成中一步完成，以减少“自己出题、自己证明”造成的偏差。

## 4. 数据设计总矩阵

第一轮不追求大规模。先由大模型生成约 1,200 条候选 pair，经语义裁判、去重和质量门禁后，保留 **900–1,000 条高质量 DPO 训练数据**。不为凑满 1,000 条放宽标准。

| 主轴 | 候选目标 | 最终目标 | 比例 | 训练目标 |
| --- | ---: | ---: | ---: | --- |
| 实体边界争议 | 420 | 350 | 35% | 简洁、充分、原子化、字段归属清晰 |
| 实体字段/类型混淆 | 180 | 150 | 15% | location/workPlace/company 等抽取类型正确 |
| 标准分类错误 | 300 | 250 | 25% | 岗位 taxonomy 及工时、场所、福利等分类正确 |
| 幻觉与过度推断 | 180 | 150 | 15% | 只输出有文本证据的信息 |
| JSON/schema 错误 | 120 | 100 | 10% | 可解析、键集、类型、空值与纯 JSON 稳定 |
| **合计** | **1,200** | **1,000** | **100%** | |

最终 1,000 条的难度目标为 easy 约 150 条、medium 约 550 条、hard 约 300 条。这里的 hard 不是无法判定的模糊样本，而是 rejected 语义合理、格式正确，但违反一条细粒度业务规则的样本。

## 5. 实体边界争议设计

### 5.1 chosen 标准：最小充分边界

每个实体必须同时满足：

- **最小**：删除不影响标识度的主语、语气词、套话、结果描述和重复修饰。
- **充分**：必须保留能改变语义或类别的对象、行业、技术、年限、范围、程度和否定信息。
- **可证据**：除 `position[1]` 和标准化 `workMode` 外，非空值必须能在输入中逐字定位。
- **原子化**：列表中一个元素只承载一个可独立使用的职责、要求、加分项或福利。
- **归属正确**：不把职责、强制要求、优先条件、背景描述和福利相互混放。

### 5.2 边界错误子类

| 错误码 | 设计方法 | rejected 的迷惑性 |
| --- | --- | --- |
| `BND_LONG_CLAUSE` | 实体包含完整句、目的或结果 | 内容全对，但不够简洁 |
| `BND_SHORT_HEAD` | 只保留中心词 | 看似概括，但丢掉业务域/技术/对象 |
| `BND_MODIFIER_DROP` | 删除年限、程度、学历、语言或软件限定 | 主体仍正确，但门槛变了 |
| `BND_NEIGHBOR_ATTACH` | 把相邻的薪资、地点、福利或说明粘连到实体 | 表面上都与岗位有关 |
| `BND_SHARED_MODIFIER` | 并列项共享修饰语时遗漏或错绑 | 局部字面匹配成立 |
| `MERGE_ATOMS` | 将多个职责/要求合并为长句 | 信息没丢，但无法结构化使用 |
| `SPLIT_CONCEPT` | 将不可分术语或一个条件拆碎 | 每个片段单看都像实体 |
| `POSITION_NOISE` | position 带入招聘词、地点、薪资、急聘等 | 保留了职位原始标题的外观 |
| `POSITION_OVER_GENERALIZE` | `position[1]` 改变专业方向或职级 | 通用名语义相近但不等价 |
| `NEGATION_SCOPE` | 把“不限/无需/不接受”后的内容当正向要求 | 关键词真实出现于原文 |
| `CONDITION_SCOPE` | 将“如果/可/优先/之一”改成强制事实 | 实体本身有文本证据 |
| `DUPLICATE_SEMANTIC` | 同一信息被不同表述重复抽取 | 两个 span 都正确，但输出冗余 |

### 5.3 边界示例

**生成输入**：“岗位职责：负责优化跨境电商推荐排序模型，持续提升点击与转化效果。”

- chosen: `"responsibilities":["优化跨境电商推荐排序模型"]`
- rejected（过长）: `"responsibilities":["负责优化跨境电商推荐排序模型，持续提升点击与转化效果"]`
- rejected（过短）: `"responsibilities":["优化模型"]`

训练时这是两条独立 pair，不在同一 pair 中同时放两种 rejected。

## 6. 实体字段/类型混淆设计

这一类不是 taxonomy 分类，而是同一片段在抽取 schema 中应放入哪个字段。

| 混淆组 | 必须设计的对抗场景 |
| --- | --- |
| `company / workPlace / location` | 公司名含地名；园区名含公司词；区县+大厦+门店同句出现 |
| `position / responsibilities` | 岗位名是动名短语；职责句重复岗位名；一文多岗 |
| `responsibilities / requirements` | “负责”后出现能力要求；“要求”段落中混入日常任务 |
| `requirements / preferredQualifications` | 必须、优先、加分、有则更佳等模态对照 |
| `description / responsibilities` | 团队业务背景与候选人需承担的任务同句出现 |
| `benefits / salary / salaryPayment` | 年终奖、保底薪资、提成、日结、补贴的边界 |
| `schedule / workMode / employmentType` | 远程不等于兼职；弹性不等于线上；夜班不等于临时工 |
| `education / workExperience / requirements` | 学历、专业、年限、证书在同一条件中并列 |

**生成示例**：“工作地点：深圳市南山区；办公场所：科兴科学园 B2 栋。”

- chosen: `location="深圳市南山区"`, `workPlace="科兴科学园B2栋"`
- rejected: `location="深圳市南山区科兴科学园B2栋"`, `workPlace=""`

## 7. 标准分类错误设计

### 7.1 岗位 taxonomy 分类

不从历史 confusion matrix 中抽混淆边。由大模型遍历 taxonomy 的层级结构，根据语义相似度、职责重叠和行业多义性，自主设计混淆对。

每个岗位 pair 必须属于以下一种：

| 错误码 | rejected 设计 |
| --- | --- |
| `CLS_L1_DOMAIN` | 被技术词诱导到另一行业一级类 |
| `CLS_L2_SIBLING` | 一级正确，二级选相邻职能 |
| `CLS_L3_NEAR` | 一、二级正确，叶子岗位近义混淆 |
| `CLS_NON_CANONICAL` | 使用语义正确但 taxonomy 不存在的近义名 |
| `CLS_ILLEGAL_PATH` | 三个名称都存在，但不构成真实父子路径 |
| `CLS_TITLE_ONLY` | 只看岗位标题，忽略职责、产品和技术线索 |
| `CLS_CONTEXT_BAIT` | 被公司行业、福利或工具词诱导，忽略职位核心动作 |
| `CLS_LEVEL_GRANULARITY` | 有充分证据时停在过粗类别，或证据不足时过度细分 |
| `CLS_MULTI_INTENT` | 复合岗位中未按主要职责选类 |
| `CLS_OTHER_ABUSE` | 可明确归类却输出“其他” |

**生成示例**：“职位：汽车线束设计工程师。负责整车低压线束原理图、布置与设计验证。”

- chosen: `汽车 / 汽车研发/制造 / 线束设计`
- rejected: `电子/电气/通信 / 电子/硬件开发 / 电路设计`

rejected 之所以有迷惑性，是因为“低压”、“原理图”和“线束”具有电子语义；但岗位对象是整车线束，应归汽车研发。

### 7.2 工时、场所、福利、用工与结算分类

每个候选类别至少生成以下四种语义形态：

1. 标准名称直接出现。
2. 口语、缩写、数字或时间段间接表达。
3. 与相邻类别的关键词同时出现，需要根据完整语义消歧。
4. 近似但不足以分类，正确结果应为“其他”的边界场景。

要专门设计双向 pair：既要防止“有证据仍分其他”，也要防止“证据不足却强行精细分类”。

## 8. 幻觉实体与过度推断设计

幻觉 rejected 不能是随机增加一个无关实体，必须是根据常识、模板、近邻文本或语义联想很容易补全出来的错误。

| 错误码 | 场景设计 | rejected 示意 |
| --- | --- | --- |
| `HAL_WORLD_KNOWLEDGE` | 知名公司/产品/城市只出现其一 | 根据常识补全另一实体 |
| `HAL_BRAND_PRODUCT` | 说公司发布新品但没说产品名 | 补成该公司最知名产品 |
| `HAL_TEMPLATE_SLOT` | 招聘文本缺少常见的学历/薪资/福利槽位 | 按招聘模板补齐 |
| `HAL_ACRONYM_EXPAND` | 缩写可能对应多个概念 | 无证据展开为某一全称 |
| `HAL_WORK_MODE` | 只有“线上会议/数字化工具” | 推断为远程办公 |
| `HAL_EMPLOYMENT` | 只有“时间灵活/按任务付费” | 直接推断为兼职 |
| `HAL_SALARY_PAYMENT` | 出现日薪/时薪但未说结算周期 | 推断日结/小时结 |
| `HAL_NEGATED` | “不要求本科/不包住” | 将被否定实体当正向值 |
| `HAL_OPTIONAL_AS_FACT` | “可在北京或上海”、“有 Java/Python 之一即可” | 将所有选项当确定事实 |
| `HAL_ADJACENT_LEAK` | 同一输入中列出多岗位或表格多行 | 把相邻岗位实体复制到目标岗位 |
| `HAL_NORMALIZATION` | 原文只说“本科”、“一年经验” | 自行改为“本科及以上”、“1–3年” |
| `HAL_CAUSAL_INFERENCE` | 职责暗示某技能通常必需 | 把隐含能力写成文本明示要求 |

**生成示例**：“要求熟悉至少一种公有云平台，具备容器化部署经验。”

- chosen: `requirements=["熟悉至少一种公有云平台", "具备容器化部署经验"]`
- rejected: 额外输出 `"AWS"` 或 `"阿里云"`

rejected 的云平台名很合理，但没有原文证据，因此是有价值的 hard negative。

## 9. JSON 与 schema 错误设计

JSON 数据不只生成明显无法解析的 rejected。建议 70% 为“语义完全正确，只错一个格式/schema 点”，30% 为真实的非法 JSON。

| 错误码 | rejected 设计 |
| --- | --- |
| `JSON_EXTRA_PROSE` | JSON 前后增加解释、结论或致谢 |
| `JSON_MARKDOWN` | 使用 ```json 围栏 |
| `JSON_WRONG_KEY` | `entities/name/category` 等近义键替代正式键 |
| `JSON_MISSING_KEY` | 删除应保留的空字段 |
| `JSON_EXTRA_KEY` | 增加 reasoning、confidence、level4 等键 |
| `JSON_TYPE` | 字符串/数组/对象互换，如 position 输出为字符串 |
| `JSON_EMPTY_VALUE` | `""` / `[]` 错用 `null`、`{}`、`"N/A"` |
| `JSON_SINGLE_QUOTE` | 使用 Python 字典风格单引号 |
| `JSON_TRAILING_COMMA` | 对象或数组末尾多逗号 |
| `JSON_UNESCAPED` | 引号、换行或反斜杠未转义 |
| `JSON_DUPLICATE_KEY` | 同一键输出两次且值不同 |
| `JSON_TRUNCATED` | 长列表结尾缺少括号或引号 |
| `JSON_MULTI_OBJECT` | 同时输出初稿 JSON 和修正后 JSON |
| `JSON_SCHEMA_CROSS_TASK` | 抽取任务输出分类字段，或分类任务输出抽取全 schema |

格式 pair 中 chosen/rejected 的实体内容必须一致，否则模型无法确定训练的是格式偏好还是语义偏好。

## 10. Pair 的硬性标准

每条 pair 必须同时满足：

1. **一条主偏好**：rejected 只包含一个主错误。
2. **最小差异**：非目标字段、键顺序和 JSON 序列化风格尽量相同。
3. **chosen 可证明**：每个非空抽取值有证据 span；每个分类有判定线索和合法路径。
4. **rejected 可惑惑**：必须能用一句话说明能力不稳的模型为什么会这样答。
5. **偏好唯一**：不依赖个人文风时，chosen 仍能依据业务规则稳定胜出。
6. **不靠表面特征**：避免 chosen 总是更短、总是紧凑 JSON 或总使用固定键顺序而 rejected 不是。
7. **不训练争议**：如果两个答案在现有规则下都可接受，不得强行设为偏好对。

## 11. 大模型生成执行计划

### 阶段 A：规则原子化

将 schema 和 taxonomy 转换为不含任何现有样本的规则卡：

- 每个字段的正定义、反定义和允许输出类型。
- 每个 taxonomy 路径及其与近邻路径的语义区分点。
- 每种错误码的构造要求和禁止情况。

### 阶段 B：50 条标定集

每个主轴先生成 10 条，共 50 条。逐条检查：

- “简洁但不缺重点”是否具有一致标准。
- rejected 是否足够像真实模型错误。
- 分类依据是否充分，是否依赖未写入 prompt 的隐含知识。
- 错误类型之间是否可分。

标定集只用于固化生成规范，不直接训练。

### 阶段 C：1,200 条候选，保留约 1,000 条

- 按“任务 × 错误码 × 难度 × 行业 × 文本形态”制定覆盖卡。
- 每批只生成 20–40 条，批次之间检查模板化和语义重复。
- 每条先产出 truth ledger，再产出训练 pair。
- 同一原文如需训练多个错误，作为同一 `scenario_group` 整体分配，防止训测泄漏。

### 阶段 D：语义裁判

对每条 pair 做三次不同角度的大模型审查：

1. **证据审查**：chosen 中的抽取值是否有原文证据，是否漏掉必要限定词。
2. **反例审查**：rejected 是否只错一点，是否可能由真实模型产生。
3. **盲比审查**：随机交换 A/B 位置，不提供生成理由，重新选择更优答案并说明所依据的规则。

三次结论不一致，或盲比无法稳定选中 chosen，则废弃样本，不做自动修补。

### 阶段 E：训练与扩量

- 使用审查后的 900–1,000 条数据训练一个低学习率、短 epoch 版本。
- 评测既有 frozen test，它可以用于评估，但不能被读取或反向用于构造训练 pair。
- 另外生成一份与训练覆盖卡隔离的 synthetic challenge set，用于检验新构造错误的泛化。
- 本轮只验证这约 1,000 条数据的方向，不预设继续扩量。只有目标错误稳定改善、且非目标任务不退化时，再单独决定是否开启下一轮数据构造。

## 12. 数据存储格式

### 12.1 候选/审查文件

候选文件保留语义质检元数据：

```json
{
  "case_id": "SYN_JOB_CLS_000001",
  "task": "job_name_classification",
  "scenario_group": "SYN_SCENE_000001",
  "target_error": "CLS_L1_DOMAIN",
  "difficulty": "hard",
  "industry": "汽车研发",
  "prompt": {},
  "chosen": {},
  "rejected": {},
  "evidence": [],
  "preference_reason": "根据整车线束的设计对象归入汽车研发",
  "rejected_plausibility": "电气词汇容易诱导到电子硬件",
  "review": {}
}
```

### 12.2 训练文件

最终训练集只输出项目现用的 ShareGPT ranking 结构：

```json
{
  "conversations": [
    {"from": "system", "value": "<正式任务指令>"},
    {"from": "human", "value": "<大模型独立生成的招聘输入>"}
  ],
  "chosen": {"from": "gpt", "value": "<chosen JSON>"},
  "rejected": {"from": "gpt", "value": "<rejected JSON 或目标格式错误>"}
}
```

truth ledger、偏好理由和审查意见不进入 DPO 训练文本。

## 13. 质量验收标准

### 13.1 数据本身

| 指标 | 首轮数据要求 |
| --- | ---: |
| chosen schema/taxonomy 合法率 | 100% |
| chosen 非空抽取值可证据率 | 100%（许可标准化字段除外） |
| chosen/rejected 不相同 | 100% |
| 单主错误率 | ≥95% |
| 盲比 chosen 胜出率 | ≥95% |
| rejected 合理性审查通过率 | ≥90% |
| 语义近重复率 | <5% |
| 任一单一模板占比 | <2% |

### 13.2 训练后

- JSON parse rate 和 schema exact 不得低于 SFT 基线。
- 实体边界同时评估 `strict exact`、关键 token precision、关键 token recall 和 atomicity，不只看字符串变短。
- 幻觉使用 unsupported entity rate，不得以降低正确实体召回为代价。
- 岗位分类同时评估 level1/level2/level3 exact、完整 path exact 和非法路径率。
- 每个错误码单独统计 DPO 前后净修正，避免总分掩盖某一类退化。
- 只有 frozen test 和 synthetic challenge set 都改善，且非目标任务无明显退化，才扩量或上线。

## 14. 里程碑

| 里程碑 | 交付物 | 通过条件 |
| --- | --- | --- |
| M1 规则原子化 | 字段规则卡、taxonomy 区分卡、错误码手册 | 不含现有标注样本 |
| M2 标定 | 50 条标定案例及审查结论 | 边界与偏好规则无明显争议 |
| M3 候选生成 | 约 1,200 条 truth ledger + candidate pair | 覆盖矩阵满足配额 |
| M4 首轮发布 | 900–1,000 条 DPO JSONL + 质检报告 | 达到数据验收阈值，不为凑数放宽标准 |
| M5 首轮微调 | DPO adapter + frozen/challenge 对比报告 | 目标错误改善，守门指标不退化 |
| M6 决策 | 继续使用 SFT、接受 DPO 或进入第二轮的结论 | 以评测结果而非数据量决定 |

## 15. 最终决策

本计划的核心不是用脚本大量制造形式上的 chosen/rejected，而是由大模型主动设计“两个答案都像对的，但只有一个符合业务规则”的场景。

高质量的标准是：

> chosen 简洁、完整、有证据、类别正确、格式稳定；rejected 具有强迷惑性，但只违反一条可清晰解释的业务规则。

如果某条数据需要借助现有标注才能确定 chosen，或必须用规则脚本才能生成 rejected，则它不属于本轮数据范围。
