#!/usr/bin/env python3
"""Generate and validate the complete 10k DPO v1.1 dataset.

SFT rows contribute only task inputs and source identity. Their ``output`` field
is never consulted. Chosen labels are rebuilt from visible text, the locked
task prompts, model-authored semantic rules, and job_cate.json.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = ROOT / "code/data/task_info/label_prompts_v2.md"
JOB_CATE_FILE = ROOT / "job_cate.json"
SFT_ROOT = ROOT / "code/data/gemini_labeled_v2/tasks_fixed"
RAW_QUERY_FILE = ROOT / "code/data/processed/search_queries_clean.jsonl"
FROZEN_FILE = ROOT / "code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl"
PILOT_FILE = ROOT / "code/data/datasets/dpo/llm_generated_v1_1_pilot50/pilot50_review.jsonl"
OUT_DIR = ROOT / "code/data/datasets/dpo/llm_generated_v1_1_full10k"
SEED = 20260713

TASK_TARGETS = {
    "entity_extraction": 4000,
    "job_name_classification": 3200,
    "work_time_classification": 700,
    "workplace_classification": 600,
    "benefit_classification": 600,
    "work_type_classification": 250,
    "settlement_classification": 250,
    "taxonomy_leaf_to_parent": 200,
    "taxonomy_level2_to_children": 200,
}
SOURCE_TARGETS = {
    "entity_extraction": {"sft_input": 2400, "llm_synthetic": 1600},
    "job_name_classification": {"sft_input": 1920, "llm_synthetic": 1280},
    "work_time_classification": {"sft_input": 420, "llm_synthetic": 280},
    "workplace_classification": {"sft_input": 360, "llm_synthetic": 240},
    "benefit_classification": {"sft_input": 360, "llm_synthetic": 240},
    "work_type_classification": {"sft_input": 150, "llm_synthetic": 100},
    "settlement_classification": {"sft_input": 150, "llm_synthetic": 100},
    "taxonomy_leaf_to_parent": {"taxonomy": 200},
    "taxonomy_level2_to_children": {"taxonomy": 200},
}
DEV_TARGETS = {
    "entity_extraction": 400,
    "job_name_classification": 320,
    "work_time_classification": 70,
    "workplace_classification": 60,
    "benefit_classification": 60,
    "work_type_classification": 25,
    "settlement_classification": 25,
    "taxonomy_leaf_to_parent": 20,
    "taxonomy_level2_to_children": 20,
}
DIFFICULTY_TARGETS = {"easy": 1000, "medium": 6000, "hard": 3000}
ENTITY_ERROR_TARGETS = {
    "E01_boundary_too_short": 900,
    "E02_boundary_too_long": 900,
    "E04_field_confusion": 900,
    "E07_hallucination": 600,
    "E08_omission": 400,
    "E11_json_schema": 200,
    "E10_rule_exception": 100,
}
JOB_ERROR_TARGETS = {
    "E05_category_confusion": 2080,
    "E06_hierarchy_inconsistency": 480,
    "E09_context_ignored": 320,
    "E07_hallucination": 160,
    "E11_json_schema": 160,
}
TAXONOMY_CHILD_ERROR_TARGETS = {
    "E08_omission": 70,
    "E03_over_extraction": 70,
    "E06_hierarchy_inconsistency": 30,
    "E11_json_schema": 30,
}
STAGE_TARGETS = {
    "entity_extraction": (400, 1500, 1440, 660),
    "job_name_classification": (300, 1300, 1140, 460),
    "work_time_classification": (80, 300, 240, 80),
    "workplace_classification": (60, 250, 210, 80),
    "benefit_classification": (60, 250, 210, 80),
    "work_type_classification": (30, 100, 80, 40),
    "settlement_classification": (30, 100, 80, 40),
    "taxonomy_leaf_to_parent": (20, 100, 50, 30),
    "taxonomy_level2_to_children": (20, 100, 50, 30),
}
STAGE_NAMES = ("stage_1_pilot", "stage_2_core_rules", "stage_3_hard_tail", "stage_4_model_loop")
STAGE1_SFT_TARGETS = {
    "entity_extraction": 250,
    "job_name_classification": 188,
    "work_time_classification": 50,
    "workplace_classification": 38,
    "benefit_classification": 38,
    "work_type_classification": 19,
    "settlement_classification": 17,
    "taxonomy_leaf_to_parent": 0,
    "taxonomy_level2_to_children": 0,
}
STAGE3_SYNTH_TARGETS = {
    "entity_extraction": 1000,
    "job_name_classification": 1000,
    "work_time_classification": 180,
    "workplace_classification": 160,
    "benefit_classification": 160,
    "work_type_classification": 60,
    "settlement_classification": 60,
    "taxonomy_leaf_to_parent": 0,
    "taxonomy_level2_to_children": 0,
}

ENTITY_KEYS = [
    "position", "company", "location", "workPlace", "workMode", "salary",
    "salaryPayment", "employmentType", "schedule", "description",
    "responsibilities", "requirements", "preferredQualifications",
    "education", "workExperience", "benefits",
]
TASK_SCHEMAS = {
    "entity_extraction": set(ENTITY_KEYS),
    "job_name_classification": {"position", "level1", "level2", "level3"},
    "taxonomy_leaf_to_parent": {"level3", "level2", "level1"},
    "taxonomy_level2_to_children": {"level1", "level2", "level3Children"},
    "work_type_classification": {"entity", "category"},
    "workplace_classification": {"rawWorkPlace", "level1", "level2"},
    "settlement_classification": {"entity", "category"},
    "work_time_classification": {"entity", "level1", "level2", "fullCategory"},
    "benefit_classification": {"entity", "standardBenefit"},
}
SFT_FILES = {
    task: SFT_ROOT / task / "train.jsonl"
    for task in (
        "entity_extraction", "job_name_classification", "work_time_classification",
        "workplace_classification", "benefit_classification",
        "work_type_classification", "settlement_classification",
    )
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def entity(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "position": [], "company": "", "location": "", "workPlace": "",
        "workMode": "", "salary": "", "salaryPayment": "",
        "employmentType": "", "schedule": [], "description": [],
        "responsibilities": [], "requirements": [],
        "preferredQualifications": [], "education": "",
        "workExperience": "", "benefits": [],
    }
    value.update(updates)
    return value


def parse_prompts() -> dict[str, str]:
    text = PROMPT_FILE.read_text(encoding="utf-8").split("## 微调训练提示词", 1)[1]
    prompts = dict(re.findall(r"### `([^`]+)`\s*\n\n```text\n(.*?)\n```", text, re.S))
    missing = set(TASK_SCHEMAS) - set(prompts)
    if missing:
        raise ValueError(f"missing task prompts: {sorted(missing)}")
    return {task: prompt.rstrip() for task, prompt in prompts.items()}


def prompt_candidates(prompt: str) -> list[str]:
    match = re.search(r"候选分类体系（必须严格遵守）：(\[[^\n]+\])", prompt)
    return json.loads(match.group(1)) if match else []


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


PROMPTS = parse_prompts()
JOB_CATE: dict[str, dict[str, list[str]]] = json.loads(JOB_CATE_FILE.read_text(encoding="utf-8"))
CANDIDATES = {
    task: prompt_candidates(PROMPTS[task])
    for task in (
        "work_type_classification", "workplace_classification",
        "settlement_classification", "work_time_classification",
        "benefit_classification",
    )
}
JOB_PATHS = [
    (level1, level2, level3)
    for level1, children in JOB_CATE.items()
    for level2, leaves in children.items()
    for level3 in leaves
]
PATH_SET = set(JOB_PATHS)
LEAF_PATHS: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
for path in JOB_PATHS:
    LEAF_PATHS[path[2]].append(path)


def load_frozen() -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    norms: set[str] = set()
    for row in read_jsonl(FROZEN_FILE):
        human = next((m.get("value", "") for m in row.get("conversations", []) if m.get("from") == "human"), "")
        exact.add(human)
        norms.add(normalized(human))
    return exact, norms


FROZEN_EXACT, FROZEN_NORMALIZED = load_frozen()


def base_record(
    *, task: str, input_text: str, chosen: dict[str, Any], rejected: dict[str, Any] | None,
    source_type: str, source_id: str, error: str, target_field: str,
    reason: str, score: int, group: str, rejected_raw: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "input": input_text,
        "chosen": chosen,
        "source_type": source_type,
        "source_id": source_id,
        "scenario_group_id": group,
        "primary_error_type": error,
        "target_field": target_field,
        "review_reason": reason,
        "difficulty_score": score,
        "generation_method": "gpt5_semantic_rules_and_locked_candidates",
        "chosen_provenance": "llm_independent_from_visible_input_and_locked_candidates",
        "rejected_provenance": "llm_authored_hard_negative",
        "human_review_status": "auto_validated_pending_spotcheck",
    }
    if rejected_raw is not None:
        row["rejected_raw"] = rejected_raw
    else:
        row["rejected"] = rejected
    return row


def load_pilot() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pilot in read_jsonl(PILOT_FILE):
        row = base_record(
            task=pilot["task"], input_text=pilot["input"], chosen=pilot["chosen"],
            rejected=pilot["rejected"], source_type=pilot["source_type"],
            source_id=pilot["source_id"], error=pilot["primary_error_type"],
            target_field=pilot["target_field"], reason=pilot["review_reason"],
            score={"easy": 0, "medium": 1, "hard": 2}[pilot["difficulty"]],
            group=f"pilot:{pilot['source_id']}:{pilot['pair_id']}",
        )
        row["difficulty"] = pilot["difficulty"]
        row["pilot_pair_id"] = pilot["pair_id"]
        row["human_review_status"] = "user_approved_pilot"
        rows.append(row)
    return rows


def nonempty_fields(value: dict[str, Any]) -> list[str]:
    return [key for key, item in value.items() if item not in ("", [])]


def split_source_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    return [item.strip() for item in re.split(r"\||；|;", value) if item.strip()]


def first_visible(raw: dict[str, Any], keys: list[str], input_text: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, (int, float)):
            value = str(value)
        if isinstance(value, str) and value and value in input_text:
            return value
    return ""


def visible_list(raw: dict[str, Any], keys: list[str], input_text: str) -> list[str]:
    result: list[str] = []
    for key in keys:
        for value in split_source_values(raw.get(key)):
            if value in input_text and value not in result:
                result.append(value)
    return result


def rebuild_query_entity(raw: dict[str, Any], input_text: str) -> dict[str, Any]:
    position = first_visible(raw, ["job_text", "jobText"], input_text)
    work_mode = str(raw.get("workMode", ""))
    if work_mode not in {"现场办公", "远程办公", "线上线下结合"}:
        work_mode = ""
    value = entity(
        position=[position] if position else [],
        company=first_visible(raw, ["company", "company_name"], input_text),
        location=first_visible(raw, ["location"], input_text),
        workPlace=first_visible(raw, ["workPlace"], input_text),
        workMode=work_mode,
        salary=first_visible(raw, ["salaryText", "salary_text", "salary"], input_text),
        salaryPayment=first_visible(raw, ["salaryPaymentText", "salaryPayment"], input_text),
        employmentType=first_visible(raw, ["employmentTypeText", "employmentType"], input_text),
        schedule=visible_list(raw, ["schedule"], input_text),
        description=visible_list(raw, ["description"], input_text),
        responsibilities=visible_list(raw, ["responsibilities"], input_text),
        requirements=visible_list(raw, ["requirements"], input_text),
        preferredQualifications=visible_list(raw, ["preferredQualifications"], input_text),
        education=first_visible(raw, ["education"], input_text),
        workExperience=first_visible(raw, ["workExperience"], input_text),
        benefits=visible_list(raw, ["benefits"], input_text),
    )
    return value


def shortened(value: str) -> str | None:
    replacements = [
        (r"^(资深|高级|中级|初级|首席|高级资深)", ""),
        (r"(有限责任公司|股份有限公司|有限公司)$", "公司"),
        (r"(科技公司|集团公司)$", "公司"),
        (r"^(上海|北京|深圳|广州|杭州)(市)?", r"\1"),
    ]
    for pattern, repl in replacements:
        candidate = re.sub(pattern, repl, value).strip()
        if candidate and candidate != value and len(candidate) >= 2:
            return candidate
    for token in ("/", "·", "(", "（"):
        if token in value:
            candidate = value.split(token, 1)[0].strip()
            if len(candidate) >= 2 and candidate != value:
                return candidate
    if len(value) >= 6:
        return value[-4:]
    return None


def entity_negative(
    chosen: dict[str, Any], input_text: str, error: str, serial: int
) -> tuple[dict[str, Any] | None, str | None, str, str, int] | None:
    rejected = copy.deepcopy(chosen)
    if error == "E01_boundary_too_short":
        for field in ("position", "company", "location", "workPlace", "workExperience", "education"):
            item = rejected[field]
            if isinstance(item, list) and item:
                candidate = shortened(str(item[0]))
                if candidate:
                    item[0] = candidate
                    return rejected, None, field, "rejected 截短了具有区分度的实体边界。", 1
            if isinstance(item, str) and item:
                candidate = shortened(item)
                if candidate:
                    rejected[field] = candidate
                    return rejected, None, field, "rejected 截短了具有区分度的实体边界。", 1
        return None
    if error == "E04_field_confusion":
        pairs = [("salary", "salaryPayment"), ("location", "workPlace"), ("company", "workPlace"), ("employmentType", "workMode")]
        for left, right in pairs:
            if rejected[left] not in ("", []) and rejected[right] not in ("", []):
                rejected[left], rejected[right] = rejected[right], rejected[left]
                return rejected, None, f"{left}|{right}", "rejected 将两个业务字段互换。", 2
        fields = nonempty_fields(rejected)
        if "location" in fields and not rejected["workPlace"]:
            rejected["workPlace"] = rejected["location"]
            rejected["location"] = ""
            return rejected, None, "location|workPlace", "rejected 把行政地理位置放入具体场所。", 1
        return None
    if error == "E07_hallucination":
        additions = [
            ("workMode", "现场办公"), ("salaryPayment", "月结"),
            ("company", "某科技有限公司"), ("benefits", ["五险一金"]),
        ]
        for field, value in additions[serial % len(additions):] + additions[:serial % len(additions)]:
            if rejected[field] in ("", []) and (not isinstance(value, str) or value not in input_text):
                rejected[field] = value
                return rejected, None, field, "rejected 补充了输入中不存在的实体。", 1
        return None
    if error == "E08_omission":
        preferred = ["position", "schedule", "benefits", "workExperience", "education", "salaryPayment", "workPlace", "company"]
        for field in preferred + nonempty_fields(rejected):
            if field in rejected and rejected[field] not in ("", []):
                rejected[field] = [] if isinstance(rejected[field], list) else ""
                return rejected, None, field, "rejected 漏掉了输入明确出现的字段。", 1
        return None
    if error == "E10_rule_exception":
        if rejected["position"]:
            position = str(rejected["position"][0])
            rejected["position"] = [position, position + "岗位", position + "专员"]
            return rejected, None, "position", "rejected 违反 position 最多两个元素的规则。", 1
        rejected["workMode"] = "居家办公"
        return rejected, None, "workMode", "rejected 使用了候选外 workMode。", 1
    if error == "E11_json_schema":
        if serial % 2:
            raw = "以下是抽取结果：" + compact(chosen)
            return None, raw, "json", "rejected 在 JSON 前增加解释文字。", 0
        wrong = copy.deepcopy(chosen)
        wrong["work_mode"] = wrong.pop("workMode")
        return None, compact(wrong), "schema", "rejected 将固定字段 workMode 改成 work_mode。", 0
    return None


ROLES = [path[2] for path in JOB_PATHS]
COMPANIES = ["云帆科技有限公司", "启明智造集团", "远航供应链有限公司", "星河医疗科技", "青禾教育咨询有限公司", "恒跃电子有限公司"]
LOCATIONS = ["上海 浦东新区", "北京 海淀区", "深圳 南山区", "杭州 滨江区", "成都 武侯区", "苏州 工业园区", "武汉 江夏区", "广州 天河区"]
WORKPLACES = ["星耀中心A座", "智谷产业园", "临港仓储中心", "高新科技大厦", "滨江软件园", "国际金融中心"]
SALARIES = ["8千-1.2万", "1.2-1.8万·13薪", "300元/天", "25-35元/小时", "6-9千"]
PAYMENTS = ["每月15日发薪", "每周五结算", "当天下班结算", "按件计酬"]
EMPLOYMENTS = ["全职", "兼职", "临时工", "实习生", "暑期工"]
SCHEDULES = ["常白班", "周末双休", "两班倒", "上午班", "晚上11点到次日7点", "弹性打卡"]
BENEFITS = ["餐补", "交通补贴", "包住", "五险一金", "带薪年假", "节日福利"]
RESPONSIBILITIES = [
    "客户需求分析", "核心模块开发", "运营数据复盘", "供应商对接", "库存数据核对",
    "缺陷识别模型训练", "设备巡检", "合同审核", "门店活动策划", "物流订单跟踪",
    "产品原型设计", "实验数据整理", "用户问题处理", "生产工艺优化", "项目进度跟踪",
]
REQUIREMENTS = [
    "三年以上相关经验", "本科及以上学历", "熟练使用Excel", "持有C1驾驶证",
    "具备数据分析能力", "接受短期出差", "熟悉Linux系统", "具备沟通协调能力",
    "英语可作为工作语言", "有制造业项目经验",
]


def synthetic_entity(serial: int, error: str) -> tuple[str, dict[str, Any], dict[str, Any] | None, str | None, str, str, int]:
    role = ROLES[(serial * 17) % len(ROLES)]
    modifier = ["高级", "资深", "跨境", "汽车", "医疗", "夜班"][serial % 6]
    position = modifier + role
    company = COMPANIES[(serial * 3) % len(COMPANIES)]
    location = LOCATIONS[(serial * 5) % len(LOCATIONS)]
    workplace = WORKPLACES[(serial * 7) % len(WORKPLACES)]
    salary = SALARIES[(serial * 11) % len(SALARIES)]
    payment = PAYMENTS[(serial * 13) % len(PAYMENTS)]
    employment = EMPLOYMENTS[(serial * 19) % len(EMPLOYMENTS)]
    schedule = SCHEDULES[(serial * 23) % len(SCHEDULES)]
    benefit = BENEFITS[(serial * 29) % len(BENEFITS)]
    r1 = RESPONSIBILITIES[(serial * 31) % len(RESPONSIBILITIES)]
    r2 = RESPONSIBILITIES[(serial * 37 + 3) % len(RESPONSIBILITIES)]
    q1 = REQUIREMENTS[(serial * 41) % len(REQUIREMENTS)]
    q2 = REQUIREMENTS[(serial * 43 + 2) % len(REQUIREMENTS)]
    input_text = (
        f"职位名称：{position}\n招聘公司：{company}\n工作地点：{location} {workplace}\n"
        f"薪资：{salary}\n结算说明：{payment}\n用工类型：{employment}\n工作时间：{schedule}\n"
        f"岗位职责：负责{r1}；承担{r2}。\n任职要求：{q1}；{q2}。\n福利：{benefit}。"
    )
    chosen = entity(
        position=[position], company=company, location=location, workPlace=workplace,
        salary=salary, salaryPayment=payment, employmentType=employment,
        schedule=[schedule], responsibilities=[r1, r2], requirements=[q1, q2],
        benefits=[benefit],
    )
    if error == "E02_boundary_too_long":
        rejected = copy.deepcopy(chosen)
        rejected["responsibilities"] = [f"负责{r1}", f"承担{r2}"]
        return input_text, chosen, rejected, None, "responsibilities", "rejected 保留职责句式壳，chosen 只保留核心短语。", 2
    result = entity_negative(chosen, input_text, error, serial)
    if result is None:
        raise ValueError(f"cannot build synthetic entity negative for {error}")
    rejected, raw, field, reason, score = result
    return input_text, chosen, rejected, raw, field, reason, score


def parse_position(input_text: str) -> str:
    first = input_text.splitlines()[0]
    return first.split("：", 1)[1].strip() if "：" in first else first.strip()


def match_unique_job(position: str) -> tuple[str, str, str] | None:
    candidates = [leaf for leaf in LEAF_PATHS if leaf == position or leaf in position]
    cleaned = re.sub(r"^(资深|高级|中级|初级|首席|兼职|实习|急招)+", "", position)
    cleaned = re.sub(r"(岗位|招聘)$", "", cleaned).strip()
    candidates.extend(leaf for leaf in LEAF_PATHS if leaf == cleaned or leaf in cleaned)
    if not candidates:
        return None
    leaf = max(set(candidates), key=len)
    paths = LEAF_PATHS[leaf]
    return paths[0] if len(paths) == 1 else None


def wrong_job_path(path: tuple[str, str, str], serial: int, prefer_duplicate: bool = False) -> tuple[str, str, str]:
    level1, level2, leaf = path
    if prefer_duplicate and len(LEAF_PATHS[leaf]) > 1:
        alternatives = [item for item in LEAF_PATHS[leaf] if item != path]
        if alternatives:
            return alternatives[serial % len(alternatives)]
    siblings = [item for item in JOB_PATHS if item[:2] == (level1, level2) and item != path]
    if siblings:
        return siblings[serial % len(siblings)]
    same_level1 = [item for item in JOB_PATHS if item[0] == level1 and item != path]
    if same_level1:
        return same_level1[serial % len(same_level1)]
    return JOB_PATHS[(JOB_PATHS.index(path) + 1) % len(JOB_PATHS)]


def wrong_parent_for_leaf(path: tuple[str, str, str], serial: int) -> tuple[str, str]:
    """Return a legal but incorrect level1/level2 parent for a fixed leaf."""
    alternatives = sorted({item[:2] for item in JOB_PATHS if item[0] == path[0] and item[:2] != path[:2]})
    if not alternatives:
        alternatives = sorted({item[:2] for item in JOB_PATHS if item[:2] != path[:2]})
    return alternatives[serial % len(alternatives)]


LEVEL1_CUES = {
    "互联网/AI": "系统研发、算法或数据平台",
    "产品": "产品规划、需求分析和版本路线图",
    "客服/运营": "用户增长、平台运营和数据复盘",
    "销售": "客户开发、商务谈判和业绩达成",
    "生产制造": "生产工艺、设备和质量改进",
    "零售/生活服务": "门店服务和消费者体验",
    "餐饮": "餐厅运营、前厅或后厨服务",
    "教育培训": "课程教学和学员培养",
    "医疗健康": "临床、诊疗或医药专业工作",
    "金融": "金融业务、风控或交易支持",
    "汽车": "整车、车规部件和汽车研发",
    "物流/仓储/司机": "运输、仓储和订单履约",
    "直播/影视/传媒": "影视、直播内容或传媒制作",
    "人力/行政/法务": "企业人事、行政或内部法务",
    "房地产/建筑": "建筑工程、施工或地产项目",
}


def job_record_from_path(
    path: tuple[str, str, str], position: str, input_text: str, source_type: str,
    source_id: str, serial: int, group: str,
) -> dict[str, Any]:
    wrong = wrong_job_path(path, serial, prefer_duplicate=source_type == "llm_synthetic")
    chosen = {"position": position, "level1": path[0], "level2": path[1], "level3": path[2]}
    if serial % 20 == 0:
        wrong_schema = {"position": position, "category1": path[0], "level2": path[1], "level3": path[2]}
        return base_record(
            task="job_name_classification", input_text=input_text, chosen=chosen, rejected=None,
            rejected_raw=compact(wrong_schema), source_type=source_type, source_id=source_id,
            error="E11_json_schema", target_field="schema",
            reason="rejected 将固定字段 level1 改成 category1。", score=0, group=group,
        )
    rejected = {"position": position, "level1": wrong[0], "level2": wrong[1], "level3": wrong[2]}
    cross = path[:2] != wrong[:2]
    return base_record(
        task="job_name_classification", input_text=input_text, chosen=chosen, rejected=rejected,
        source_type=source_type, source_id=source_id, error="E05_category_confusion",
        target_field="level1|level2|level3" if cross else "level3",
        reason="chosen 使用输入语义支持的最细合法路径，rejected 是相邻合法类别。",
        score=2 if cross else 1, group=group,
    )


TIME_PATTERNS: list[tuple[str, list[str]]] = [
    ("单休制-996", ["996", "早9晚9每周6天"]),
    ("长短周制-大小周", ["大小周", "单双休", "隔周双休"]),
    ("轮班制-做三休三", ["做三休三", "上3休3"]),
    ("轮班制-做二休二", ["做二休二", "上2休2"]),
    ("轮班制-做一休一", ["做一休一", "上一休一", "上1休1"]),
    ("集中轮休制-做一休二", ["做一休二", "24小时班"]),
    ("轮班制-三班倒", ["三班倒", "三班轮换"]),
    ("轮班制-两班倒", ["两班倒", "日夜交替"]),
    ("轮班制-上M休N", ["上4休3", "做四休三", "上六休一"]),
    ("夜班制-通宵班", ["通宵班", "整夜班", "晚上11点到次日早上7点", "23:00-7:00"]),
    ("夜班制-大夜班", ["大夜班", "深夜班"]),
    ("半天班-上午班", ["上午班", "早半天班"]),
    ("半天班-下午班", ["下午班", "晚半天班"]),
    ("小时工-上午钟点", ["上午钟点", "早班钟点"]),
    ("小时工-下午钟点", ["下午钟点", "傍晚钟点"]),
    ("小时工-晚间钟点", ["晚间钟点", "晚上钟点"]),
    ("小时工-不定时钟点", ["不定时钟点", "时间点双方商议"]),
    ("单休制-月休四天", ["月休4天", "月休四天"]),
    ("单休制-单休", ["单休", "每周休一天"]),
    ("双休制-双休", ["双休", "月休8天", "每周休两天"]),
    ("白班制-长白班", ["常白班", "长白班", "朝九晚五", "朝八晚六", "8小时白班"]),
    ("弹性工时制-核心弹性", ["弹性打卡", "核心时间", "弹性工时"]),
    ("调休累积制-加班调休", ["加班调休", "加班换休", "调休"]),
    ("项目周期制-忙闲季交替", ["淡旺季", "忙季", "旺季集中工作"]),
    ("任务交付制-时间自由", ["时间自由", "不打卡", "按成果交付"]),
    ("待命响应制-轮值待命", ["轮值待命", "on-call", "待命"]),
    ("住家制-住家保姆", ["住家保姆", "24小时住家"]),
    ("住家制-月嫂", ["月嫂", "育儿嫂"]),
]
WORKTYPE_PATTERNS = [
    ("假期工", ["暑期工", "寒假工", "假期工", "寒暑假工"]),
    ("学生工", ["学生工", "大学生兼职", "在校生兼职"]),
    ("周末工", ["周末工", "周末兼职"]),
    ("大龄工", ["大龄工", "退休返聘", "55岁以上"]),
    ("实习生", ["实习生", "实习"]),
    ("短期工", ["短期工", "临时工", "小时工", "短工"]),
    ("兼职工", ["兼职工", "兼职"]),
    ("长期工", ["长期工", "长期稳定", "长期"]),
]
SETTLEMENT_PATTERNS = [
    ("半月结", ["半月结", "每半月结算", "每月两次结算"]),
    ("小时结", ["小时结", "每小时结算"]),
    ("日结", ["日结", "当天结算", "下班结算", "一日一结"]),
    ("周结", ["周结", "每周结算", "每周发薪"]),
    ("月结", ["月结", "每月结算", "月底发工资", "按月发放"]),
    ("计件", ["计件", "按件计酬", "多劳多得"]),
]
BENEFIT_PATTERNS = [
    ("包吃包住", ["包吃包住", "包食宿", "管吃住"]),
    ("餐补/工作餐", ["餐补", "饭补", "工作餐", "员工餐"]),
    ("交通补贴/班车", ["交通补贴", "交通补助", "班车"]),
    ("住房补贴/宿舍", ["住房补贴", "住宿补贴", "员工宿舍"]),
    ("包吃", ["包吃", "管饭"]),
    ("包住", ["包住", "提供住宿"]),
    ("五险一金", ["五险一金", "社保公积金", "交保险的工作"]),
    ("补充医疗保险", ["补充医疗保险", "补充医疗"]),
    ("年度体检", ["年度体检", "每年体检"]),
    ("带薪年假", ["带薪年假", "年休假"]),
    ("带薪病假", ["带薪病假"]),
    ("弹性工作制", ["弹性工作制", "弹性办公"]),
    ("远程办公", ["远程办公", "居家办公"]),
    ("内部培训", ["内部培训", "岗内培训", "入职培训"]),
    ("外部培训", ["外部培训", "外派培训"]),
    ("内部晋升通道", ["晋升通道", "内部晋升"]),
    ("导师制", ["导师制", "一对一导师"]),
    ("节日福利", ["节日福利", "节日礼品"]),
    ("生日福利", ["生日福利", "生日礼物"]),
    ("团建旅游", ["团建", "员工旅游"]),
    ("零食下午茶", ["下午茶", "零食"]),
    ("年终奖", ["年终奖", "年底双薪"]),
    ("绩效奖金", ["绩效奖金", "绩效奖"]),
    ("项目奖金", ["项目奖金", "项目奖"]),
    ("销售提成", ["销售提成", "业绩提成"]),
    ("全勤奖", ["全勤奖"]),
    ("股票期权", ["股票期权", "股权激励"]),
]


def classify_patterns(entity_text: str, patterns: list[tuple[str, list[str]]], allowed: set[str]) -> str | None:
    if entity_text in allowed:
        return entity_text
    for label, aliases in patterns:
        if any(alias.lower() in entity_text.lower() for alias in aliases):
            return label
    return None


WORKPLACE_PATHS = [tuple(value.split("-", 1)) for value in CANDIDATES["workplace_classification"]]
WORKPLACE_KEYWORDS: list[tuple[tuple[str, str], list[str]]] = [
    (("仓库", "物流园区"), ["物流园"]), (("仓库", "仓库"), ["仓库"]),
    (("工厂", "工业园区"), ["工业园"]), (("工厂", "食品加工厂"), ["食品厂", "食品加工"]),
    (("工厂", "厂矿"), ["厂区", "工厂", "厂矿"]), (("公司", "园区"), ["软件园", "科技园", "产业园"]),
    (("公司", "写字楼"), ["写字楼", "大厦", "中心", "办公楼"]),
    (("社区住宅", "住宅区"), ["小区", "住宅区", "社区"]), (("社区住宅", "宿舍"), ["宿舍"]),
    (("医院", "体检机构"), ["体检中心", "体检机构"]), (("医院", "诊所"), ["诊所", "门诊部"]),
    (("医院", "药店"), ["药店", "药房"]), (("医院", "综合医院"), ["医院"]),
    (("教育培训机构", "高等院校"), ["大学", "学院"]), (("教育培训机构", "中学"), ["中学"]),
    (("教育培训机构", "小学"), ["小学"]), (("教育培训机构", "幼儿园"), ["幼儿园"]),
    (("教育培训机构", "培训机构"), ["培训学校", "培训中心", "培训机构"]),
    (("酒店", "快捷酒店"), ["快捷酒店"]), (("酒店", "民宿"), ["民宿"]),
    (("酒店", "宾馆"), ["宾馆"]), (("酒店", "星级酒店"), ["大酒店", "酒店"]),
    (("零售店", "购物中心"), ["购物中心", "商业广场", "商场"]), (("零售店", "超市"), ["超市"]),
    (("零售店", "便利店"), ["便利店"]), (("餐饮店", "奶茶店"), ["奶茶店"]),
    (("餐饮店", "咖啡厅"), ["咖啡馆", "咖啡厅"]), (("餐饮店", "食堂"), ["食堂"]),
    (("餐饮店", "小吃快餐店"), ["快餐店", "小吃店"]), (("金融机构", "银行"), ["银行"]),
    (("交通站", "机场"), ["机场"]), (("交通站", "火车站"), ["火车站", "高铁站"]),
    (("交通站", "地铁站"), ["地铁站"]), (("工地", "建筑工地"), ["建筑工地"]),
    (("工地", "施工现场"), ["施工现场"]), (("美容美发店", "美发店"), ["理发店", "美发店"]),
    (("运动健身场所", "健身中心"), ["健身房", "健身中心"]),
]


def classify_workplace(entity_text: str) -> tuple[str, str] | None:
    matches: list[tuple[int, tuple[str, str]]] = []
    for path in WORKPLACE_PATHS:
        for token in path[1].split("/"):
            if len(token) >= 2 and token in entity_text:
                matches.append((len(token), path))
    for path, aliases in WORKPLACE_KEYWORDS:
        for alias in aliases:
            if alias in entity_text:
                matches.append((len(alias) + 5, path))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def wrong_label(allowed: list[str], chosen: str, serial: int) -> str:
    index = allowed.index(chosen)
    return allowed[(index + 1 + serial % max(1, len(allowed) - 1)) % len(allowed)]


def make_small_record(
    task: str, input_text: str, chosen_label: str | tuple[str, str], source_type: str,
    source_id: str, serial: int, group: str,
) -> dict[str, Any]:
    if task == "work_time_classification":
        assert isinstance(chosen_label, str)
        wrong = wrong_label(CANDIDATES[task], chosen_label, serial)
        c1, c2 = chosen_label.split("-", 1) if chosen_label != "其他" else ("", "")
        r1, r2 = wrong.split("-", 1) if wrong != "其他" else ("", "")
        entity_text = input_text.split("：", 1)[1]
        chosen = {"entity": entity_text, "level1": c1, "level2": c2, "fullCategory": chosen_label}
        rejected = {"entity": entity_text, "level1": r1, "level2": r2, "fullCategory": wrong}
        field = "fullCategory"
    elif task == "workplace_classification":
        assert isinstance(chosen_label, tuple)
        wrong = WORKPLACE_PATHS[
            (WORKPLACE_PATHS.index(chosen_label) + 1 + serial % (len(WORKPLACE_PATHS) - 1))
            % len(WORKPLACE_PATHS)
        ]
        entity_text = input_text.split("：", 1)[1]
        chosen = {"rawWorkPlace": entity_text, "level1": chosen_label[0], "level2": chosen_label[1]}
        rejected = {"rawWorkPlace": entity_text, "level1": wrong[0], "level2": wrong[1]}
        field = "level1|level2"
    elif task == "benefit_classification":
        assert isinstance(chosen_label, str)
        wrong = wrong_label(CANDIDATES[task], chosen_label, serial)
        entity_text = input_text.split("：", 1)[1]
        chosen = {"entity": entity_text, "standardBenefit": chosen_label}
        rejected = {"entity": entity_text, "standardBenefit": wrong}
        field = "standardBenefit"
    else:
        assert isinstance(chosen_label, str)
        wrong = wrong_label(CANDIDATES[task], chosen_label, serial)
        entity_text = input_text.split("：", 1)[1]
        chosen = {"entity": entity_text, "category": chosen_label}
        rejected = {"entity": entity_text, "category": wrong}
        field = "category"
    return base_record(
        task=task, input_text=input_text, chosen=chosen, rejected=rejected,
        source_type=source_type, source_id=source_id, error="E05_category_confusion",
        target_field=field, reason="chosen 符合实体核心语义，rejected 是合法但错误的相邻候选。",
        score=1 if source_type == "sft_input" else (2 if serial % 3 == 0 else 1), group=group,
    )


SYNTH_ALIASES = {
    "work_time_classification": {label: aliases for label, aliases in TIME_PATTERNS},
    "work_type_classification": {label: aliases for label, aliases in WORKTYPE_PATTERNS},
    "settlement_classification": {label: aliases for label, aliases in SETTLEMENT_PATTERNS},
    "benefit_classification": {label: aliases for label, aliases in BENEFIT_PATTERNS},
}


def synthetic_small(task: str, serial: int) -> tuple[str, str | tuple[str, str]]:
    if task == "workplace_classification":
        path = WORKPLACE_PATHS[serial % len(WORKPLACE_PATHS)]
        prefixes = ["云海", "星河", "青禾", "远航", "启明", "恒悦"]
        branches = ["一店", "二店", "东区", "西区", "中心店", "新城店", "园区店"]
        entity_text = f"{prefixes[serial % len(prefixes)]}{path[1]}{branches[(serial // len(prefixes)) % len(branches)]}"
        return "实体：" + entity_text, path
    allowed = CANDIDATES[task]
    label = allowed[serial % len(allowed)]
    subjects = ["研发中心", "门店", "仓储组", "项目部", "运营团队", "客服组", "生产班组", "销售团队"]
    regions = ["总部", "华东区", "华南区", "华北区", "西南区", "新城园区", "临港基地", "城西项目"]
    subject_index = serial // max(1, len(allowed))
    subject = regions[(subject_index // len(subjects)) % len(regions)] + subjects[subject_index % len(subjects)]
    if label == "其他":
        base = ["薪资面议", "环境舒适", "团队年轻", "发展稳定", "岗位急招", "面试安排灵活"][serial % 6]
        phrase = f"{subject}{base}"
    else:
        aliases = SYNTH_ALIASES[task].get(label, [label])
        phrase = aliases[(serial // len(allowed)) % len(aliases)]
        phrase = [
            f"{subject}{phrase}", f"{subject}明确{phrase}",
            f"{subject}提供{phrase}", f"{subject}采用{phrase}",
            f"{subject}执行{phrase}", f"{subject}安排{phrase}",
        ][serial % 6]
    return "实体：" + phrase, label


def add_sft_small(
    records: list[dict[str, Any]], used_inputs: set[str], task: str,
    target_sft: int, classifier: Callable[[str], str | tuple[str, str] | None],
) -> int:
    current = sum(1 for row in records if row["task"] == task and row["source_type"] == "sft_input")
    serial = 0
    for row in read_jsonl(SFT_FILES[task]):
        if current >= target_sft:
            break
        input_text = str(row["input"])
        if input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            continue
        if not input_text.startswith("实体："):
            continue
        label = classifier(input_text.split("：", 1)[1])
        if label is None:
            continue
        record = make_small_record(task, input_text, label, "sft_input", str(row.get("source_id", "")), serial, f"sft:{row.get('source_id', '')}:{task}")
        records.append(record)
        used_inputs.add(input_text)
        current += 1
        serial += 1
    # High-confidence relabeling is preferred over forcing the planned SFT share.
    # Any shortfall is filled with model-authored cases by the caller.
    return current


def add_synthetic_small(records: list[dict[str, Any]], used_inputs: set[str], task: str, target: int) -> None:
    current = sum(1 for row in records if row["task"] == task and row["source_type"] == "llm_synthetic")
    serial = 0
    attempts = 0
    while current < target:
        input_text, label = synthetic_small(task, serial)
        serial += 1
        attempts += 1
        if attempts > 200000:
            raise ValueError(f"insufficient unique synthetic inputs for {task}: {current}/{target}")
        if input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            continue
        record = make_small_record(task, input_text, label, "llm_synthetic", f"synthetic_{task}_{serial:05d}", serial, f"synthetic:{task}:{serial:05d}")
        records.append(record)
        used_inputs.add(input_text)
        current += 1


def build_records() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    records = load_pilot()
    used_inputs = {str(row["input"]) for row in records}

    # 1) entity extraction from SFT query inputs, independently rebuilt from visible spans.
    raw_queries = {str(row["query_id"]): row for row in read_jsonl(RAW_QUERY_FILE)}
    target_entity_sft = SOURCE_TARGETS["entity_extraction"]["sft_input"]
    entity_counts = Counter(row["primary_error_type"] for row in records if row["task"] == "entity_extraction")
    source_errors = ["E04_field_confusion", "E01_boundary_too_short", "E07_hallucination", "E08_omission", "E10_rule_exception", "E11_json_schema"]
    serial = 0
    for row in read_jsonl(SFT_FILES["entity_extraction"]):
        current = sum(1 for item in records if item["task"] == "entity_extraction" and item["source_type"] == "sft_input")
        if current >= target_entity_sft:
            break
        input_text = str(row["input"])
        source_id = str(row.get("source_id", ""))
        if source_id not in raw_queries or input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            continue
        chosen = rebuild_query_entity(raw_queries[source_id], input_text)
        if len(nonempty_fields(chosen)) < 2:
            continue
        built = None
        for offset in range(len(source_errors)):
            error = source_errors[(serial + offset) % len(source_errors)]
            if entity_counts[error] >= ENTITY_ERROR_TARGETS[error]:
                continue
            result = entity_negative(chosen, input_text, error, serial)
            if result is not None:
                built = (error, result)
                break
        if built is None:
            continue
        error, (rejected, rejected_raw, field, reason, score) = built
        records.append(base_record(
            task="entity_extraction", input_text=input_text, chosen=chosen, rejected=rejected,
            rejected_raw=rejected_raw, source_type="sft_input", source_id=source_id,
            error=error, target_field=field, reason=reason, score=score,
            group=f"sft:{source_id}",
        ))
        used_inputs.add(input_text)
        entity_counts[error] += 1
        serial += 1
    if sum(1 for item in records if item["task"] == "entity_extraction" and item["source_type"] == "sft_input") != target_entity_sft:
        raise ValueError("insufficient SFT entity inputs")

    # Fill exact entity error targets with model-authored synthetic scenes.
    remaining_errors: list[str] = []
    for error, target in ENTITY_ERROR_TARGETS.items():
        remaining_errors.extend([error] * (target - entity_counts[error]))
    rng.shuffle(remaining_errors)
    for error in remaining_errors:
        serial += 1
        input_text, chosen, rejected, rejected_raw, field, reason, score = synthetic_entity(serial, error)
        while input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            serial += 1
            input_text, chosen, rejected, rejected_raw, field, reason, score = synthetic_entity(serial, error)
        records.append(base_record(
            task="entity_extraction", input_text=input_text, chosen=chosen, rejected=rejected,
            rejected_raw=rejected_raw, source_type="llm_synthetic",
            source_id=f"synthetic_entity_{serial:05d}", error=error, target_field=field,
            reason=reason, score=score, group=f"synthetic:entity:{serial:05d}",
        ))
        used_inputs.add(input_text)

    # 2) job classification: unique taxonomy matches from SFT inputs.
    target_job_sft = SOURCE_TARGETS["job_name_classification"]["sft_input"]
    serial = 0
    for row in read_jsonl(SFT_FILES["job_name_classification"]):
        current = sum(1 for item in records if item["task"] == "job_name_classification" and item["source_type"] == "sft_input")
        if current >= target_job_sft:
            break
        input_text = str(row["input"])
        if input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            continue
        position = parse_position(input_text)
        path = match_unique_job(position)
        if path is None:
            continue
        serial += 1
        source_id = str(row.get("source_id", ""))
        records.append(job_record_from_path(path, position, input_text, "sft_input", source_id, serial, f"sft:{source_id}:job"))
        used_inputs.add(input_text)
    if sum(1 for item in records if item["task"] == "job_name_classification" and item["source_type"] == "sft_input") != target_job_sft:
        raise ValueError("insufficient unique-match SFT job inputs")

    # Job long tail and duplicate-name disambiguation from locked taxonomy.
    target_job_synth = SOURCE_TARGETS["job_name_classification"]["llm_synthetic"]
    current_job_synth = sum(1 for item in records if item["task"] == "job_name_classification" and item["source_type"] == "llm_synthetic")
    serial = 0
    while current_job_synth < target_job_synth:
        path = JOB_PATHS[serial % len(JOB_PATHS)]
        modifier = ["", "资深", "高级", "初级", "兼职", "项目制"][serial % 6]
        position = modifier + path[2]
        cue = LEVEL1_CUES.get(path[0], f"{path[1]}专业领域")
        contexts = [
            f"负责{cue}相关工作，岗位重点为{path[2]}。",
            f"所在团队从事{path[1]}业务，需要承担{path[2]}职责。",
            f"结合{path[0]}业务场景开展{path[2]}专业工作。",
            f"岗位不跨类，核心工作属于{path[1]}方向。",
        ]
        input_text = f"岗位实体：{position}\n原文上下文：{contexts[(serial // len(JOB_PATHS)) % len(contexts)]}"
        serial += 1
        if input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            continue
        records.append(job_record_from_path(path, position, input_text, "llm_synthetic", f"synthetic_job_{serial:05d}", serial, f"synthetic:job:{path[0]}/{path[1]}/{path[2]}:{serial // len(JOB_PATHS)}"))
        used_inputs.add(input_text)
        current_job_synth += 1

    # 3) smaller closed-set tasks.
    time_allowed = set(CANDIDATES["work_time_classification"])
    actual_sft = add_sft_small(records, used_inputs, "work_time_classification", SOURCE_TARGETS["work_time_classification"]["sft_input"], lambda text: classify_patterns(text, TIME_PATTERNS, time_allowed))
    add_synthetic_small(records, used_inputs, "work_time_classification", TASK_TARGETS["work_time_classification"] - actual_sft)

    actual_sft = add_sft_small(records, used_inputs, "workplace_classification", SOURCE_TARGETS["workplace_classification"]["sft_input"], classify_workplace)
    add_synthetic_small(records, used_inputs, "workplace_classification", TASK_TARGETS["workplace_classification"] - actual_sft)

    benefit_allowed = set(CANDIDATES["benefit_classification"])
    actual_sft = add_sft_small(records, used_inputs, "benefit_classification", SOURCE_TARGETS["benefit_classification"]["sft_input"], lambda text: classify_patterns(text, BENEFIT_PATTERNS, benefit_allowed))
    add_synthetic_small(records, used_inputs, "benefit_classification", TASK_TARGETS["benefit_classification"] - actual_sft)

    worktype_allowed = set(CANDIDATES["work_type_classification"])
    actual_sft = add_sft_small(records, used_inputs, "work_type_classification", SOURCE_TARGETS["work_type_classification"]["sft_input"], lambda text: classify_patterns(text, WORKTYPE_PATTERNS, worktype_allowed))
    add_synthetic_small(records, used_inputs, "work_type_classification", TASK_TARGETS["work_type_classification"] - actual_sft)

    settlement_allowed = set(CANDIDATES["settlement_classification"])
    actual_sft = add_sft_small(records, used_inputs, "settlement_classification", SOURCE_TARGETS["settlement_classification"]["sft_input"], lambda text: classify_patterns(text, SETTLEMENT_PATTERNS, settlement_allowed))
    add_synthetic_small(records, used_inputs, "settlement_classification", TASK_TARGETS["settlement_classification"] - actual_sft)

    # 4) taxonomy tasks from job_cate.json only.
    unique_leaf_paths = [paths[0] for paths in LEAF_PATHS.values() if len(paths) == 1]
    current_leaf = sum(1 for item in records if item["task"] == "taxonomy_leaf_to_parent")
    for serial, path in enumerate(unique_leaf_paths):
        if current_leaf >= TASK_TARGETS["taxonomy_leaf_to_parent"]:
            break
        input_text = f"三级职位：{path[2]}"
        if input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            continue
        wrong_level1, wrong_level2 = wrong_parent_for_leaf(path, serial)
        records.append(base_record(
            task="taxonomy_leaf_to_parent", input_text=input_text,
            chosen={"level3": path[2], "level2": path[1], "level1": path[0]},
            rejected={"level3": path[2], "level2": wrong_level2, "level1": wrong_level1},
            source_type="taxonomy", source_id="job_cate.json",
            error="E06_hierarchy_inconsistency", target_field="level1|level2",
            reason="chosen 是 job_cate.json 中该唯一三级名称的真实父路径。",
            score=1, group=f"taxonomy:leaf:{path[2]}",
        ))
        used_inputs.add(input_text)
        current_leaf += 1

    current_children = sum(1 for item in records if item["task"] == "taxonomy_level2_to_children")
    path_index = 0
    variant = 0
    level2_paths = [(level1, level2) for level1, children in JOB_CATE.items() for level2 in children]
    while current_children < TASK_TARGETS["taxonomy_level2_to_children"]:
        level1, level2 = level2_paths[path_index % len(level2_paths)]
        variant = path_index // len(level2_paths)
        if variant == 0:
            input_text = f"二级职位完整路径：{level1}/{level2}"
        else:
            input_text = f"一级职位：{level1}\n二级职位：{level2}\n请枚举全部三级职位"
        path_index += 1
        if input_text in used_inputs or input_text in FROZEN_EXACT or normalized(input_text) in FROZEN_NORMALIZED:
            continue
        children = list(JOB_CATE[level1][level2])
        rejected_children = children[:-1] if len(children) > 1 else children + [wrong_job_path((level1, level2, children[0]), path_index)[2]]
        records.append(base_record(
            task="taxonomy_level2_to_children", input_text=input_text,
            chosen={"level1": level1, "level2": level2, "level3Children": children},
            rejected={"level1": level1, "level2": level2, "level3Children": rejected_children},
            source_type="taxonomy", source_id="job_cate.json",
            error="E08_omission" if len(children) > 1 else "E06_hierarchy_inconsistency",
            target_field="level3Children", reason="rejected 漏项或混入了非该父路径的子类。",
            score=2, group=f"taxonomy:children:{level1}/{level2}",
        ))
        used_inputs.add(input_text)
        current_children += 1

    return records


def rebalance_job_negatives(records: list[dict[str, Any]]) -> None:
    fixed = [row for row in records if row["task"] == "job_name_classification" and "pilot_pair_id" in row]
    counts = Counter(row["primary_error_type"] for row in fixed)
    assignments: list[str] = []
    for error, target in JOB_ERROR_TARGETS.items():
        remaining = target - counts[error]
        if remaining < 0:
            raise ValueError(f"pilot exceeds job error target: {error}")
        assignments.extend([error] * remaining)
    pool = [row for row in records if row["task"] == "job_name_classification" and "pilot_pair_id" not in row]
    pool.sort(key=lambda row: digest(str(row["input"])))
    random.Random(SEED + 11).shuffle(assignments)
    if len(pool) != len(assignments):
        raise ValueError("job negative assignment mismatch")
    for serial, (row, error) in enumerate(zip(pool, assignments)):
        chosen = row["chosen"]
        path = (chosen["level1"], chosen["level2"], chosen["level3"])
        row.pop("rejected_raw", None)
        if error in {"E05_category_confusion", "E09_context_ignored"}:
            wrong = wrong_job_path(path, serial, prefer_duplicate=error == "E09_context_ignored")
            row["rejected"] = {
                "position": chosen["position"], "level1": wrong[0],
                "level2": wrong[1], "level3": wrong[2],
            }
            row["target_field"] = "level1|level2|level3"
            row["review_reason"] = (
                "rejected 忽略了原文上下文的领域线索，选择了表面相近的合法路径。"
                if error == "E09_context_ignored" else
                "rejected 选择了合法但错误的相邻职位类别。"
            )
        elif error == "E06_hierarchy_inconsistency":
            invalid_leaves = sorted({
                item[2] for item in JOB_PATHS
                if item[2] != path[2] and (path[0], path[1], item[2]) not in PATH_SET
            })
            wrong_leaf = invalid_leaves[serial % len(invalid_leaves)]
            row["rejected"] = {
                "position": chosen["position"], "level1": path[0],
                "level2": path[1], "level3": wrong_leaf,
            }
            row["target_field"] = "level3"
            row["review_reason"] = "rejected 的三级职位不属于所输出的一、二级路径。"
        elif error == "E07_hallucination":
            row["rejected"] = dict(chosen)
            row["rejected"]["position"] = chosen["position"] + "总监"
            row["target_field"] = "position"
            row["review_reason"] = "rejected 对岗位实体增加了原文不存在的职级。"
        elif error == "E11_json_schema":
            row.pop("rejected", None)
            row["rejected_raw"] = "分类结果：" + compact(chosen)
            row["target_field"] = "json"
            row["review_reason"] = "rejected 在固定 JSON 前添加了解释性文字。"
        row["primary_error_type"] = error


def rebalance_taxonomy_children(records: list[dict[str, Any]]) -> None:
    fixed = [row for row in records if row["task"] == "taxonomy_level2_to_children" and "pilot_pair_id" in row]
    counts = Counter(row["primary_error_type"] for row in fixed)
    pool = [row for row in records if row["task"] == "taxonomy_level2_to_children" and "pilot_pair_id" not in row]
    pool.sort(key=lambda row: digest(str(row["input"])))
    assignments: list[str] = []
    for error, target in TAXONOMY_CHILD_ERROR_TARGETS.items():
        assignments.extend([error] * (target - counts[error]))
    # Omission requires at least two true children. Reserve eligible rows first.
    omission_count = assignments.count("E08_omission")
    eligible = [row for row in pool if len(row["chosen"]["level3Children"]) > 1]
    omission_rows = set(id(row) for row in eligible[:omission_count])
    ordered = [row for row in pool if id(row) in omission_rows] + [row for row in pool if id(row) not in omission_rows]
    assignments = ["E08_omission"] * omission_count + [error for error in assignments if error != "E08_omission"]
    if len(ordered) != len(assignments):
        raise ValueError("taxonomy child negative assignment mismatch")
    level2_paths = [(level1, level2) for level1, children in JOB_CATE.items() for level2 in children]
    for serial, (row, error) in enumerate(zip(ordered, assignments)):
        chosen = row["chosen"]
        children = list(chosen["level3Children"])
        row.pop("rejected_raw", None)
        if error == "E08_omission":
            row["rejected"] = {**chosen, "level3Children": children[:-1]}
            row["review_reason"] = "rejected 漏掉了该二级路径下的一个标准三级职位。"
        elif error == "E03_over_extraction":
            extra = next(path[2] for path in JOB_PATHS if path[:2] != (chosen["level1"], chosen["level2"]) and path[2] not in children)
            row["rejected"] = {**chosen, "level3Children": children + [extra]}
            row["review_reason"] = "rejected 混入了其他二级路径的合法三级类别。"
        elif error == "E06_hierarchy_inconsistency":
            alternatives = [path for path in level2_paths if path != (chosen["level1"], chosen["level2"])]
            wrong_parent = alternatives[serial % len(alternatives)]
            row["rejected"] = {"level1": wrong_parent[0], "level2": wrong_parent[1], "level3Children": children}
            row["review_reason"] = "rejected 将正确子类列表挂到了错误的父路径下。"
        elif error == "E11_json_schema":
            row.pop("rejected", None)
            row["rejected_raw"] = "以下是子类：" + compact(chosen)
            row["review_reason"] = "rejected 在 JSON 之外输出了额外说明。"
        row["primary_error_type"] = error
        row["target_field"] = "level3Children" if error != "E06_hierarchy_inconsistency" else "level1|level2"


def assign_stages(records: list[dict[str, Any]]) -> None:
    for task, quotas in STAGE_TARGETS.items():
        task_rows = [row for row in records if row["task"] == task]
        pilot = [row for row in task_rows if "pilot_pair_id" in row]
        for row in pilot:
            row["stage"] = STAGE_NAMES[0]
        remaining = [row for row in task_rows if "pilot_pair_id" not in row]
        if quotas[0] < len(pilot):
            raise ValueError(f"pilot exceeds stage 1 quota for {task}")
        assigned: set[int] = set()
        pilot_sft = sum(row["source_type"] == "sft_input" for row in pilot)
        desired_sft = STAGE1_SFT_TARGETS[task]
        if pilot_sft > desired_sft:
            raise ValueError(f"pilot SFT share exceeds stage 1 source target for {task}")
        stage1_sft = sorted(
            [row for row in remaining if row["source_type"] == "sft_input"],
            key=lambda row: (int(row["difficulty_score"]), digest(str(row["input"]))),
        )[:desired_sft - pilot_sft]
        stage1_other_needed = quotas[0] - len(pilot) - len(stage1_sft)
        stage1_other = sorted(
            [row for row in remaining if row["source_type"] != "sft_input"],
            key=lambda row: (int(row["difficulty_score"]), digest(str(row["input"]))),
        )[:stage1_other_needed]
        stage1 = stage1_sft + stage1_other
        if len(stage1) != quotas[0] - len(pilot):
            raise ValueError(f"stage 1 source assignment shortage: {task}")
        for row in stage1:
            row["stage"] = STAGE_NAMES[0]
            assigned.add(id(row))

        # Reserve model-authored long-tail cases for Stage 3 before Stage 2 consumes
        # the remaining synthetic pool.
        stage3_synth_target = STAGE3_SYNTH_TARGETS[task]
        stage3_synth = sorted(
            [row for row in remaining if id(row) not in assigned and row["source_type"] == "llm_synthetic"],
            key=lambda row: (-int(row["difficulty_score"]), digest(str(row["input"]))),
        )[:stage3_synth_target]
        stage3_other_needed = quotas[2] - len(stage3_synth)
        stage3_other = sorted(
            [row for row in remaining if id(row) not in assigned and id(row) not in {id(item) for item in stage3_synth}],
            key=lambda row: (-int(row["difficulty_score"]), row["source_type"] == "sft_input", digest(str(row["input"]))),
        )[:stage3_other_needed]
        stage3 = stage3_synth + stage3_other
        if len(stage3) != quotas[2]:
            raise ValueError(f"stage 3 source assignment shortage: {task}")
        for row in stage3:
            row["stage"] = STAGE_NAMES[2]
            assigned.add(id(row))

        stage2 = sorted(
            [row for row in remaining if id(row) not in assigned],
            key=lambda row: (abs(int(row["difficulty_score"]) - 1), row["source_type"] == "sft_input", digest(str(row["input"]))),
        )[:quotas[1]]
        if len(stage2) != quotas[1]:
            raise ValueError(f"stage 2 assignment shortage: {task}")
        for row in stage2:
            row["stage"] = STAGE_NAMES[1]
            assigned.add(id(row))

        stage4 = [row for row in remaining if id(row) not in assigned]
        if len(stage4) != quotas[3]:
            raise ValueError(f"stage 4 assignment mismatch: {task}")
        for row in stage4:
            row["stage"] = STAGE_NAMES[3]
            assigned.add(id(row))
        if len(assigned) != len(remaining):
            raise ValueError(f"stage assignment mismatch: {task}")


def assign_difficulties(records: list[dict[str, Any]]) -> None:
    fixed = Counter(row["difficulty"] for row in records if "difficulty" in row)
    remaining = {key: DIFFICULTY_TARGETS[key] - fixed[key] for key in DIFFICULTY_TARGETS}
    adjustable = [row for row in records if "difficulty" not in row]
    adjustable.sort(key=lambda row: (int(row["difficulty_score"]), digest(str(row["input"]))))
    cursor = 0
    for label in ("easy", "medium", "hard"):
        for row in adjustable[cursor:cursor + remaining[label]]:
            row["difficulty"] = label
        cursor += remaining[label]
    if cursor != len(adjustable):
        raise ValueError("difficulty assignment mismatch")


def assign_ids(records: list[dict[str, Any]]) -> None:
    order = {task: index for index, task in enumerate(TASK_TARGETS)}
    records.sort(key=lambda row: (order[row["task"]], row["source_type"], digest(str(row["input"]))))
    for index, row in enumerate(records, 1):
        row["pair_id"] = f"DPO_V1_1_{index:06d}"
        row["dataset_version"] = "llm-generated-v1.1-full10k"
        row["prompt_version"] = "recruitment_label_prompts_v2"
        row["system_prompt_ref"] = f"label_prompts_v2.md#微调训练提示词/{row['task']}"


def split_train_dev(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dev: list[dict[str, Any]] = []
    dev_ids: set[str] = set()
    for task, target in DEV_TARGETS.items():
        eligible = [
            row for row in records
            if row["task"] == task and row["source_type"] in {"llm_synthetic", "taxonomy"}
        ]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            groups[str(row["scenario_group_id"])].append(row)
        singletons = [items[0] for items in groups.values() if len(items) == 1]
        singletons.sort(key=lambda row: digest(str(row["scenario_group_id"])))
        if len(singletons) < target:
            raise ValueError(f"not enough singleton dev groups for {task}")
        for row in singletons[:target]:
            dev.append(row)
            dev_ids.add(str(row["pair_id"]))
    train = [row for row in records if str(row["pair_id"]) not in dev_ids]
    return train, dev


def validate_entity_spans(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    input_text = str(row["input"])
    for field, item in row["chosen"].items():
        if field == "workMode":
            continue
        values = item if isinstance(item, list) else [item]
        for index, value in enumerate(values):
            if not value or (field == "position" and index == 1):
                continue
            if not isinstance(value, str) or value not in input_text:
                issues.append(f"{row['pair_id']}: non-verbatim chosen {field}={value!r}")
    return issues


def validate(records: list[dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if len(records) != 10000:
        issues.append(f"row count {len(records)} != 10000")
    if Counter(row["task"] for row in records) != Counter(TASK_TARGETS):
        issues.append("task distribution mismatch")
    actual_sources = Counter((row["task"], row["source_type"]) for row in records)
    if any(source not in {"sft_input", "llm_synthetic", "taxonomy"} for _, source in actual_sources):
        issues.append("unexpected source type")
    for task, counts in SOURCE_TARGETS.items():
        planned_sft = counts.get("sft_input")
        if planned_sft is not None and actual_sources[(task, "sft_input")] > planned_sft:
            issues.append(f"{task}: high-confidence SFT source exceeds planned ceiling")
    if Counter(row["difficulty"] for row in records) != Counter(DIFFICULTY_TARGETS):
        issues.append("difficulty distribution mismatch")
    if Counter(row["primary_error_type"] for row in records if row["task"] == "entity_extraction") != Counter(ENTITY_ERROR_TARGETS):
        issues.append("entity error distribution mismatch")
    if Counter(row["primary_error_type"] for row in records if row["task"] == "job_name_classification") != Counter(JOB_ERROR_TARGETS):
        issues.append("job error distribution mismatch")
    if Counter(row["primary_error_type"] for row in records if row["task"] == "taxonomy_level2_to_children") != Counter(TAXONOMY_CHILD_ERROR_TARGETS):
        issues.append("taxonomy child error distribution mismatch")
    expected_stages = Counter({name: sum(values[index] for values in STAGE_TARGETS.values()) for index, name in enumerate(STAGE_NAMES)})
    if Counter(row.get("stage") for row in records) != expected_stages:
        issues.append("global stage distribution mismatch")
    for task, values in STAGE_TARGETS.items():
        expected = Counter({name: values[index] for index, name in enumerate(STAGE_NAMES)})
        if Counter(row.get("stage") for row in records if row["task"] == task) != expected:
            issues.append(f"{task}: stage distribution mismatch")
    stage1_sources = Counter(row["source_type"] for row in records if row.get("stage") == STAGE_NAMES[0])
    if stage1_sources != Counter({"sft_input": 600, "llm_synthetic": 360, "taxonomy": 40}):
        issues.append(f"stage 1 source distribution mismatch: {stage1_sources}")
    inputs = [str(row["input"]) for row in records]
    if len(inputs) != len(set(inputs)):
        issues.append("duplicate input")
    if len({row["pair_id"] for row in records}) != len(records):
        issues.append("duplicate pair_id")
    for row in records:
        task = row["task"]
        chosen = row["chosen"]
        if set(chosen) != TASK_SCHEMAS[task]:
            issues.append(f"{row['pair_id']}: chosen schema")
            continue
        if "rejected_raw" not in row:
            rejected = row.get("rejected")
            if not isinstance(rejected, dict) or set(rejected) != TASK_SCHEMAS[task]:
                issues.append(f"{row['pair_id']}: rejected schema")
            elif chosen == rejected:
                issues.append(f"{row['pair_id']}: chosen equals rejected")
        if task == "entity_extraction":
            issues.extend(validate_entity_spans(row))
        elif task == "job_name_classification":
            cpath = (chosen["level1"], chosen["level2"], chosen["level3"])
            if cpath not in PATH_SET:
                issues.append(f"{row['pair_id']}: chosen job path")
            if "rejected_raw" not in row:
                rejected = row["rejected"]
                rpath = (rejected["level1"], rejected["level2"], rejected["level3"])
                if row["primary_error_type"] != "E06_hierarchy_inconsistency" and rpath not in PATH_SET:
                    issues.append(f"{row['pair_id']}: rejected job path")
                if row["primary_error_type"] == "E06_hierarchy_inconsistency" and rpath in PATH_SET:
                    issues.append(f"{row['pair_id']}: hierarchy negative is still a valid path")
        elif task == "taxonomy_leaf_to_parent":
            if (chosen["level1"], chosen["level2"], chosen["level3"]) not in PATH_SET:
                issues.append(f"{row['pair_id']}: taxonomy leaf")
        elif task == "taxonomy_level2_to_children":
            if chosen["level3Children"] != JOB_CATE[chosen["level1"]][chosen["level2"]]:
                issues.append(f"{row['pair_id']}: taxonomy children")
        elif task in {"work_type_classification", "settlement_classification"}:
            if chosen["category"] not in CANDIDATES[task] or row["rejected"]["category"] not in CANDIDATES[task]:
                issues.append(f"{row['pair_id']}: category candidate")
        elif task == "benefit_classification":
            if chosen["standardBenefit"] not in CANDIDATES[task] or row["rejected"]["standardBenefit"] not in CANDIDATES[task]:
                issues.append(f"{row['pair_id']}: benefit candidate")
        elif task == "work_time_classification":
            for value in (chosen, row["rejected"]):
                if value["fullCategory"] not in CANDIDATES[task]:
                    issues.append(f"{row['pair_id']}: worktime candidate")
                if value["fullCategory"] != "其他" and value["fullCategory"] != value["level1"] + "-" + value["level2"]:
                    issues.append(f"{row['pair_id']}: worktime hierarchy")
        elif task == "workplace_classification":
            allowed = set(WORKPLACE_PATHS)
            for value in (chosen, row["rejected"]):
                if (value["level1"], value["level2"]) not in allowed:
                    issues.append(f"{row['pair_id']}: workplace candidate")
        if str(row["input"]) in FROZEN_EXACT or normalized(str(row["input"])) in FROZEN_NORMALIZED:
            issues.append(f"{row['pair_id']}: frozen overlap")
    if len(train) != 9000 or len(dev) != 1000:
        issues.append(f"split counts train={len(train)} dev={len(dev)}")
    if Counter(row["task"] for row in dev) != Counter(DEV_TARGETS):
        issues.append("dev task distribution mismatch")
    train_inputs = {row["input"] for row in train}
    if train_inputs & {row["input"] for row in dev}:
        issues.append("train/dev exact overlap")
    train_groups = {row["scenario_group_id"] for row in train}
    if train_groups & {row["scenario_group_id"] for row in dev}:
        issues.append("train/dev scenario group overlap")
    return issues


def sharegpt(row: dict[str, Any]) -> dict[str, Any]:
    rejected = row.get("rejected_raw", compact(row.get("rejected")))
    return {
        "conversations": [
            {"from": "system", "value": PROMPTS[row["task"]]},
            {"from": "human", "value": row["input"]},
        ],
        "chosen": {"from": "gpt", "value": compact(row["chosen"])},
        "rejected": {"from": "gpt", "value": rejected},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    records = build_records()
    rebalance_job_negatives(records)
    rebalance_taxonomy_children(records)
    assign_stages(records)
    assign_difficulties(records)
    assign_ids(records)
    train, dev = split_train_dev(records)
    issues = validate(records, train, dev)
    if issues:
        raise ValueError(json.dumps(issues[:100], ensure_ascii=False, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review_path = OUT_DIR / "full_review.jsonl"
    all_path = OUT_DIR / "all_sharegpt.jsonl"
    train_path = OUT_DIR / "train_sharegpt.jsonl"
    dev_path = OUT_DIR / "dev_sharegpt.jsonl"
    write_jsonl(review_path, records)
    write_jsonl(all_path, [sharegpt(row) for row in records])
    write_jsonl(train_path, [sharegpt(row) for row in train])
    write_jsonl(dev_path, [sharegpt(row) for row in dev])
    stage_files: dict[str, str] = {}
    for stage_name in STAGE_NAMES:
        path = OUT_DIR / f"{stage_name}_sharegpt.jsonl"
        write_jsonl(path, [sharegpt(row) for row in records if row["stage"] == stage_name])
        stage_files[stage_name] = path.name

    task_counts = Counter(row["task"] for row in records)
    source_counts = Counter(row["source_type"] for row in records)
    difficulty_counts = Counter(row["difficulty"] for row in records)
    stage_counts = Counter(row["stage"] for row in records)
    error_counts = Counter(row["primary_error_type"] for row in records)
    category_coverage = {
        "job_level1": len({row["chosen"]["level1"] for row in records if row["task"] == "job_name_classification"}),
        "job_level2_paths": len({(row["chosen"]["level1"], row["chosen"]["level2"]) for row in records if row["task"] == "job_name_classification"}),
        "job_level3_paths": len({(row["chosen"]["level1"], row["chosen"]["level2"], row["chosen"]["level3"]) for row in records if row["task"] == "job_name_classification"}),
        "work_time": len({row["chosen"]["fullCategory"] for row in records if row["task"] == "work_time_classification"}),
        "workplace_level1": len({row["chosen"]["level1"] for row in records if row["task"] == "workplace_classification"}),
        "workplace_paths": len({(row["chosen"]["level1"], row["chosen"]["level2"]) for row in records if row["task"] == "workplace_classification"}),
        "benefit": len({row["chosen"]["standardBenefit"] for row in records if row["task"] == "benefit_classification"}),
        "work_type": len({row["chosen"]["category"] for row in records if row["task"] == "work_type_classification"}),
        "settlement": len({row["chosen"]["category"] for row in records if row["task"] == "settlement_classification"}),
    }
    manifest = {
        "dataset": "recruitment_dpo_llm_generated_v1_1_full10k",
        "status": "generated_auto_validated_pending_training_evaluation",
        "total_pairs": len(records),
        "train_pairs": len(train),
        "dev_pairs": len(dev),
        "sft_outputs_used": False,
        "pilot_pairs_included": 50,
        "task_distribution": dict(task_counts),
        "source_distribution": dict(source_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "stage_distribution": dict(stage_counts),
        "task_source_distribution": {
            task: dict(Counter(row["source_type"] for row in records if row["task"] == task))
            for task in TASK_TARGETS
        },
        "error_distribution": dict(error_counts),
        "category_coverage": category_coverage,
        "authorities": {
            "job_taxonomy": "job_cate.json",
            "other_candidates": "code/data/task_info/label_prompts_v2.md",
            "job_cate_sha256": sha256(JOB_CATE_FILE),
            "label_prompts_sha256": sha256(PROMPT_FILE),
        },
        "files": {
            "review": review_path.name, "all": all_path.name,
            "train": train_path.name, "dev": dev_path.name,
            "stages": stage_files,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = {
        "valid": True, "rows": len(records), "issue_count": 0, "issues": [],
        "all_chosen_schemas_exact": True, "all_non_format_rejected_schemas_exact": True,
        "all_entity_chosen_spans_valid": True,
        "all_chosen_job_paths_in_job_cate": True,
        "all_rejected_job_category_names_in_job_cate": True,
        "all_other_categories_in_prompt_candidates": True,
        "duplicate_inputs": 0, "duplicate_pair_ids": 0,
        "frozen_exact_overlap": 0, "frozen_normalized_overlap": 0,
        "train_dev_exact_overlap": 0, "train_dev_scenario_group_overlap": 0,
        "sft_outputs_read_by_generator": False,
        "stage_quotas_exact": True,
        "job_error_quotas_exact": True,
        "taxonomy_child_error_quotas_exact": True,
        "category_coverage": category_coverage,
    }
    (OUT_DIR / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = [
        "# DPO v1.1 Full 10k Dataset Summary", "",
        "## 总览", "",
        f"- 总样本：{len(records):,}", f"- Train：{len(train):,}", f"- Dev：{len(dev):,}",
        "- SFT 旧 output 使用：否", "- 已确认 Pilot 纳入：50", "",
        "## 任务分布", "", "| 任务 | 数量 |", "| --- | ---: |",
    ]
    summary.extend(f"| `{task}` | {count:,} |" for task, count in task_counts.items())
    summary.extend(["", "## 来源分布", "", "| 来源 | 数量 |", "| --- | ---: |"]) 
    summary.extend(f"| `{key}` | {value:,} |" for key, value in source_counts.items())
    summary.extend(["", "## 难度分布", "", "| 难度 | 数量 |", "| --- | ---: |"]) 
    summary.extend(f"| `{key}` | {value:,} |" for key, value in difficulty_counts.items())
    summary.extend(["", "## 阶段分片", "", "| 阶段 | 数量 |", "| --- | ---: |"]) 
    summary.extend(f"| `{key}` | {value:,} |" for key, value in stage_counts.items())
    summary.extend(["", "## 阶段来源", "", "| 阶段 | SFT input | LLM synthetic | Taxonomy |", "| --- | ---: | ---: | ---: |"]) 
    for stage_name in STAGE_NAMES:
        counts = Counter(row["source_type"] for row in records if row["stage"] == stage_name)
        summary.append(f"| `{stage_name}` | {counts['sft_input']:,} | {counts['llm_synthetic']:,} | {counts['taxonomy']:,} |")
    summary.extend(["", "## 主要错误配额", "", "### 实体抽取", "", "| 错误类型 | 数量 |", "| --- | ---: |"]) 
    entity_error_counts = Counter(row["primary_error_type"] for row in records if row["task"] == "entity_extraction")
    summary.extend(f"| `{key}` | {value:,} |" for key, value in entity_error_counts.items())
    summary.extend(["", "### 职位分类", "", "| 错误类型 | 数量 |", "| --- | ---: |"]) 
    job_error_counts = Counter(row["primary_error_type"] for row in records if row["task"] == "job_name_classification")
    summary.extend(f"| `{key}` | {value:,} |" for key, value in job_error_counts.items())
    summary.extend(["", "## 类别覆盖", "", "| 维度 | 覆盖数 |", "| --- | ---: |"]) 
    summary.extend(f"| `{key}` | {value:,} |" for key, value in category_coverage.items())
    summary.extend([
        "", "## 校验结果", "",
        "- schema、实体 span、职位路径及闭集候选校验全部通过。",
        "- frozen exact/normalized 重叠均为 0。",
        "- train/dev exact 与 scenario group 重叠均为 0。",
        "- 重复 input、pair_id 均为 0。", "",
        "## 来源说明", "",
        "- 工作时间、工作性质和结算方式的原始 SFT input 中，可在不读取旧 output 的条件下高置信独立重标的条数低于原定上限；缺口仅在任务内转为 LLM 原创场景，未挪用其他任务配额。", "",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    # Compact 100-row spot-check sheet, stratified by task where possible.
    spot: list[dict[str, Any]] = []
    for task in TASK_TARGETS:
        rows = [row for row in records if row["task"] == task]
        rows.sort(key=lambda row: digest(str(row["pair_id"])))
        spot.extend(rows[:15 if task in {"entity_extraction", "job_name_classification"} else 10])
    spot = spot[:100]
    write_jsonl(OUT_DIR / "spotcheck_100.jsonl", spot)

    print(json.dumps({"output_dir": str(OUT_DIR), "rows": len(records), "train": len(train), "dev": len(dev), "valid": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
