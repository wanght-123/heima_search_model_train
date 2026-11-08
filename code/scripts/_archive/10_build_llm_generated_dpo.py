#!/usr/bin/env python3
"""Mechanically expand and validate LLM-authored compact DPO records.

This program does not invent samples or labels. Semantic inputs and preference
pairs live in data/datasets/dpo/llm_generated_v1/*.jsonl and were authored by
the model; this file only applies the project's fixed schemas and serialization.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/datasets/dpo/llm_generated_v1"
PROMPT_DOC = ROOT / "data/task_info/label_prompts_v2.md"
TAXONOMY = ROOT.parent / "job_cate.json"

ENTITY_SCHEMA = {
    "position": [], "company": "", "location": "", "workPlace": "",
    "workMode": "", "salary": "", "salaryPayment": "",
    "employmentType": "", "schedule": [], "description": [],
    "responsibilities": [], "requirements": [],
    "preferredQualifications": [], "education": "", "workExperience": "",
    "benefits": [],
}
PROMPT_TASKS = [
    "entity_extraction", "job_name_classification", "taxonomy_leaf_to_parent",
    "taxonomy_level2_to_children", "work_time_classification",
    "workplace_classification", "benefit_classification",
    "settlement_classification", "work_type_classification",
]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                yield lineno, json.loads(line)


def task_prompts():
    text = PROMPT_DOC.read_text(encoding="utf-8").split("## 微调训练提示词", 1)[1]
    result = dict(re.findall(r'^### `([^`]+)`\s*\n\n```text\n(.*?)\n```', text, re.M | re.S))
    missing = set(PROMPT_TASKS) - set(result)
    if missing: raise ValueError(f"missing prompts in label_prompts_v2.md: {sorted(missing)}")
    return {task: result[task] for task in PROMPT_TASKS}


def job_paths_from_prompt_doc():
    text = PROMPT_DOC.read_text(encoding="utf-8").split("## 微调训练提示词", 1)[0]
    block = re.search(r'^### `job_name_classification`\s*\n\n```text\n(.*?)\n```', text, re.M | re.S).group(1)
    values = json.loads(re.search(r'候选分类体系（必须严格遵守）：(\[.*\])', block).group(1))
    return {tuple(value.split("-", 2)) for value in values}


def closed_candidates(prompts):
    result = {}
    for task in ("work_type_classification", "workplace_classification", "settlement_classification", "work_time_classification", "benefit_classification"):
        match = re.search(r'候选分类体系（必须严格遵守）：(\[.*?\])', prompts[task], re.S)
        result[task] = set(json.loads(match.group(1)))
    return result


def entity(delta=None, base=None):
    out = copy.deepcopy(ENTITY_SCHEMA)
    out.update(base or {})
    out.update(delta or {})
    return out


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def variant(obj, name):
    """Render an explicitly authored negative format variant."""
    good = compact(obj)
    if name == "extra_prose": return "以下是抽取结果：" + good
    if name == "markdown": return "```json\n" + good + "\n```"
    if name == "single_quotes": return str(obj)
    if name == "trailing_comma": return good[:-1] + ",}"
    if name == "truncated": return good[:-1]
    if name == "multi_object": return good + "\n" + good
    if name == "unescaped_newline": return good.replace("\\n", "\n") + "\n说明"
    if name == "null_empty_values":
        x = {k: (None if v in ("", []) else v) for k, v in obj.items()}; return compact(x)
    if name == "null_value":
        x = dict(obj); x[next(reversed(x))] = None; return compact(x)
    if name == "null_level":
        x = dict(obj); x["level2"] = None; return compact(x)
    if name == "duplicate_key":
        k = next(iter(obj)); return good[:-1] + ',"' + k + '":null}'
    key_alias = {
        "wrong_key_company": ("company", "companyName"),
        "wrong_key_work_mode": ("workMode", "work_mode"),
        "wrong_key_requirements": ("requirements", "requirement"),
        "wrong_key_responsibilities": ("responsibilities", "duties"),
        "wrong_key_preferred": ("preferredQualifications", "preferred"),
        "wrong_key_education": ("education", "degree"),
        "wrong_key_experience": ("workExperience", "experience"),
        "wrong_key_position": ("position", "jobName"),
        "wrong_key_workplace": ("workPlace", "workplace"),
        "wrong_key_standard_benefit": ("standardBenefit", "benefit"),
        "wrong_key_category": ("category", "label"),
        "wrong_key_entity": ("entity", "text"),
        "wrong_key_raw": ("rawWorkPlace", "entity"),
        "wrong_key_full_category": ("fullCategory", "category"),
        "wrong_key_levels": ("level1", "category1"),
    }
    if name in key_alias:
        old, new = key_alias[name]; x = dict(obj)
        if old in x: x[new] = x.pop(old)
        else: x[new] = ""
        return compact(x)
    missing = {
        "missing_empty_keys": [k for k, v in obj.items() if v in ("", [])],
        "missing_level1": ["level1"], "missing_level2": ["level2"],
        "missing_level3": ["level3"], "missing_full_category": ["fullCategory"],
        "missing_position": ["position"], "missing_raw": ["rawWorkPlace"],
    }
    if name in missing:
        x = dict(obj)
        for k in missing[name]: x.pop(k, None)
        return compact(x)
    if name.startswith("wrong_type_"):
        x = dict(obj)
        token = name.removeprefix("wrong_type_")
        fields = {
            "salary_array": "salary", "schedule_string": "schedule",
            "responsibilities_string": "responsibilities", "benefits_string": "benefits",
            "label_array": "category", "category_array": "category", "level_array": "level1",
        }
        k = "standardBenefit" if token == "label_array" and "standardBenefit" in x else fields.get(token, token)
        if k in x: x[k] = [x[k]] if isinstance(x[k], str) else "、".join(x[k])
        return compact(x)
    if name in {"extra_key_category", "extra_key_confidence", "extra_reasoning_key", "extra_level3", "extra_level4"}:
        x = dict(obj); k = {"extra_key_category":"extraCategory","extra_key_confidence":"confidence","extra_reasoning_key":"reasoning","extra_level3":"level3","extra_level4":"level4"}[name]
        x[k] = 0.98 if k == "confidence" else "模型补充"; return compact(x)
    if name == "schema_cross_task": return compact({"entity": "", "category": "其他"})
    raise ValueError(f"unknown JSON variant: {name}")


def classification(task, fmt, case):
    if fmt == "job_path":
        if len(case) == 10:
            title, context, *rest = case
            case = [f"岗位实体：{title}\n原文上下文：{context}", *rest]
        inp, pos, a, b, c, ra, rb, rc, diff = case
        return inp, {"position":pos,"level1":a,"level2":b,"level3":c}, {"position":pos,"level1":ra,"level2":rb,"level3":rc}, diff
    if fmt == "entity_category":
        ent, chosen, rejected, diff = case
        a, b = chosen.split("-", 1) if chosen != "其他" else ("", "")
        ra, rb = rejected.split("-", 1) if rejected != "其他" else ("", "")
        return "实体："+ent, {"entity":ent,"level1":a,"level2":b,"fullCategory":chosen}, {"entity":ent,"level1":ra,"level2":rb,"fullCategory":rejected}, diff
    if fmt == "workplace":
        ent, a, b, ra, rb, diff = case
        return "实体："+ent, {"rawWorkPlace":ent,"level1":a,"level2":b}, {"rawWorkPlace":ent,"level1":ra,"level2":rb}, diff
    if fmt == "entity_label":
        ent, chosen, rejected, diff = case
        key = "standardBenefit" if task == "benefit_classification" else "category"
        return "实体："+ent, {"entity":ent,key:chosen}, {"entity":ent,key:rejected}, diff
    if fmt == "entity_label_mixed":
        task, ent, chosen, rejected, diff = case
        return task, "实体："+ent, {"entity":ent,"category":chosen}, {"entity":ent,"category":rejected}, diff
    if fmt == "taxonomy_leaf":
        leaf, a, b, ra, rb, diff = case
        return "三级职位："+leaf, {"level3":leaf,"level2":b,"level1":a}, {"level3":leaf,"level2":rb,"level1":ra}, diff
    if fmt == "taxonomy_children":
        a, b, chosen, rejected, diff = case
        inp = f"二级职位完整路径：{a}/{b}"
        return inp, {"level1":a,"level2":b,"level3Children":chosen}, {"level1":a,"level2":b,"level3Children":rejected}, diff
    raise ValueError((task, fmt))


def expand_series(row):
    out = []
    fmt, task = row.get("format"), row["task"]
    for idx, case in enumerate(row["cases"], 1):
        rid = f'{row["series"]}_{idx:03d}'
        if fmt in {"json_variants", "json_variants_job", "json_variants_mixed"}:
            if fmt == "json_variants_mixed": task0, inp, chosen, variants, diff = case
            else: inp, chosen, variants, diff = case; task0 = task
            if fmt == "json_variants": chosen = entity(chosen)
            for j, v in enumerate(variants):
                out.append(dict(id=f"{rid}_{chr(97+j)}", task=task0, bucket=row["bucket"], input=inp,
                                chosen=chosen, rejected_raw=variant(chosen, v), difficulty=diff,
                                error=row["error"], reason=row["reason_rule"]))
            continue
        if fmt:
            if fmt == "entity_category" and row.get("categories"):
                ent, c, r, diff = case
                case = [ent, row["categories"][c], row["categories"][r], diff]
            result = classification(task, fmt, case)
            if fmt == "entity_label_mixed": task0, inp, chosen, rejected, diff = result
            else: inp, chosen, rejected, diff = result; task0 = task
        else:
            inp, c, r, diff = case; task0 = task
            if row.get("shape") == "object": chosen, rejected = entity(c), entity(r)
            else:
                chosen = entity({row["target"]: c}); rejected = entity({row["target"]: r})
        out.append(dict(id=rid, task=task0, bucket=row["bucket"], input=inp, chosen=chosen,
                        rejected=rejected, difficulty=diff, error=row["error"], reason=row["reason_rule"]))
    return out


def load_split(split):
    rows = []
    for path in sorted(SRC.glob(f"{split}*.jsonl")):
        if path.name in {"train.jsonl", "eval.jsonl"}: continue
        for _, row in read_jsonl(path):
            if "series" in row: rows.extend(expand_series(row))
            else:
                rows.append(dict(id=row["id"], task=row["task"], bucket=row["bucket"], input=row["input"],
                                 chosen=entity(row.get("chosen"), row.get("base")),
                                 rejected=entity(row.get("rejected"), row.get("base")),
                                 difficulty=row["difficulty"], error=row["error"], reason=row["reason"]))
    corrections = json.loads((SRC / "corrections.json").read_text(encoding="utf-8"))
    for row in rows:
        fix = corrections.get(row["id"], {})
        if "chosen" in fix:
            row["chosen"].update(fix["chosen"])
        if "rejected" in fix:
            row["rejected"].update(fix["rejected"])
        row.update({k:v for k,v in fix.items() if k not in {"chosen","rejected"}})
    quotas = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))[f"{split}_quota"]
    kept, counts = [], Counter()
    for row in rows:
        if counts[row["bucket"]] < quotas[row["bucket"]]:
            kept.append(row); counts[row["bucket"]] += 1
    if counts != Counter(quotas): raise ValueError(f"{split} quota mismatch: {counts} != {quotas}")
    return kept


def final_record(row, prompts):
    chosen = compact(row["chosen"])
    rejected = row["rejected_raw"] if "rejected_raw" in row else compact(row["rejected"])
    return {
        "conversations": [{"from":"system","value":prompts[row["task"]]}, {"from":"human","value":row["input"]}],
        "chosen": {"from":"gpt","value":chosen}, "rejected": {"from":"gpt","value":rejected},
    }


def valid_job_paths():
    return job_paths_from_prompt_doc()


def validate(train, ev):
    issues = []
    paths = valid_job_paths()
    prompts = task_prompts()
    candidates = closed_candidates(prompts)
    schemas = {
        "job_name_classification": {"position","level1","level2","level3"},
        "work_time_classification": {"entity","level1","level2","fullCategory"},
        "workplace_classification": {"rawWorkPlace","level1","level2"},
        "benefit_classification": {"entity","standardBenefit"},
        "settlement_classification": {"entity","category"},
        "work_type_classification": {"entity","category"},
        "taxonomy_leaf_to_parent": {"level1","level2","level3"},
        "taxonomy_level2_to_children": {"level1","level2","level3Children"},
    }
    for split, rows in (("train",train),("eval",ev)):
        ids = [x["id"] for x in rows]
        if len(ids) != len(set(ids)): issues.append(f"{split}: duplicate ids")
        for x in rows:
            if compact(x["chosen"]) == x.get("rejected_raw", compact(x.get("rejected",{}))): issues.append(f'{x["id"]}: chosen equals rejected')
            if x["task"] == "entity_extraction" and set(x["chosen"]) != set(ENTITY_SCHEMA): issues.append(f'{x["id"]}: entity schema')
            if x["task"] in schemas and set(x["chosen"]) != schemas[x["task"]]: issues.append(f'{x["id"]}: classification schema')
            if x["task"] == "entity_extraction":
                for field, value in x["chosen"].items():
                    values = value if isinstance(value, list) else [value]
                    for i, text in enumerate(values):
                        if not text or field == "workMode" or (field == "position" and i == 1): continue
                        if text not in x["input"]: issues.append(f'{x["id"]}: non-verbatim {field}={text}')
            if x["task"] == "job_name_classification":
                p = tuple(x["chosen"][k] for k in ("level1","level2","level3"))
                if p not in paths: issues.append(f'{x["id"]}: invalid chosen job path {p}')
                if x["bucket"] == "standard_classification":
                    rp = tuple(x["rejected"][k] for k in ("level1","level2","level3"))
                    if rp not in paths: issues.append(f'{x["id"]}: rejected job path outside prompt candidates {rp}')
            if x["task"] in {"work_type_classification","settlement_classification"}:
                if x["chosen"]["category"] not in candidates[x["task"]]: issues.append(f'{x["id"]}: chosen category outside prompt')
            if x["task"] == "benefit_classification" and x["chosen"]["standardBenefit"] not in candidates[x["task"]]:
                issues.append(f'{x["id"]}: chosen benefit outside prompt')
            if x["task"] == "work_time_classification" and x["chosen"]["fullCategory"] not in candidates[x["task"]]:
                issues.append(f'{x["id"]}: chosen work time outside prompt')
            if x["task"] == "workplace_classification":
                pair = x["chosen"]["level1"]+"-"+x["chosen"]["level2"]
                allowed_l1 = {v.split("-",1)[0] for v in candidates[x["task"]]}
                if x["chosen"]["level1"] not in allowed_l1 or (pair not in candidates[x["task"]] and x["chosen"]["level2"] != x["chosen"]["rawWorkPlace"]):
                    issues.append(f'{x["id"]}: chosen workplace outside prompt')
            if x["task"] == "taxonomy_leaf_to_parent":
                p = tuple(x["chosen"][k] for k in ("level1","level2","level3"))
                if p not in paths: issues.append(f'{x["id"]}: invalid leaf parent path {p}')
            if x["task"] == "taxonomy_level2_to_children":
                a, b = x["chosen"]["level1"], x["chosen"]["level2"]
                expected = [p[2] for p in paths if p[:2] == (a,b)]
                if set(expected) != set(x["chosen"]["level3Children"]) or len(expected) != len(x["chosen"]["level3Children"]):
                    issues.append(f'{x["id"]}: incomplete taxonomy children')
    overlap = set(x["input"] for x in train) & set(x["input"] for x in ev)
    if overlap: issues.append(f"exact train/eval overlap: {len(overlap)}")
    return issues


def main():
    prompts = task_prompts(); train, ev = load_split("train"), load_split("eval")
    issues = validate(train, ev)
    for split, rows in (("train",train),("eval",ev)):
        with (SRC / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(final_record(row,prompts),ensure_ascii=False)+"\n")
    normalize = lambda s: re.sub(r"[\W_]+", "", s).lower()
    train_norm = {normalize(x["input"]) for x in train}
    buckets = {}
    for x in train_norm: buckets.setdefault(x[:4], set()).add(x)
    near = 0
    for row in ev:
        y = normalize(row["input"])
        near += sum(x != y and difflib.SequenceMatcher(None,x,y,autojunk=False).ratio() >= 0.92 for x in buckets.get(y[:4],()))
    report = {
        "prompt_source":"code/data/task_info/label_prompts_v2.md#微调训练提示词",
        "prompt_sha256":{task:hashlib.sha256(prompt.encode()).hexdigest() for task,prompt in prompts.items()},
        "job_candidate_path_count":len(job_paths_from_prompt_doc()),
        "closed_candidate_counts":{task:len(values) for task,values in closed_candidates(prompts).items()},
        "train_count":len(train), "eval_count":len(ev),
        "train_buckets":Counter(x["bucket"] for x in train), "eval_buckets":Counter(x["bucket"] for x in ev),
        "train_tasks":Counter(x["task"] for x in train), "eval_tasks":Counter(x["task"] for x in ev),
        "exact_input_overlap":len(set(x["input"] for x in train)&set(x["input"] for x in ev)),
        "normalized_input_overlap":sum(normalize(x["input"]) in train_norm for x in ev),
        "near_duplicate_pairs_ge_0.92":near,
        "issues":issues,
    }
    (SRC/"validation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if issues: raise SystemExit(1)


if __name__ == "__main__": main()
