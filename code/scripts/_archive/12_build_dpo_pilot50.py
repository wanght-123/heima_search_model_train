#!/usr/bin/env python3
"""Build the 50-pair DPO v1.1 review pilot without reading SFT outputs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = ROOT / "code/data/task_info/label_prompts_v2.md"
JOB_CATE_FILE = ROOT / "job_cate.json"
SFT_ROOT = ROOT / "code/data/gemini_labeled_v2/tasks_fixed"
OUT_DIR = ROOT / "code/data/datasets/dpo/llm_generated_v1_1_pilot50"
FROZEN_FILE = ROOT / "code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl"

TASK_FILES = {
    "entity_extraction": SFT_ROOT / "entity_extraction/train.jsonl",
    "job_name_classification": SFT_ROOT / "job_name_classification/train.jsonl",
    "work_time_classification": SFT_ROOT / "work_time_classification/train.jsonl",
    "workplace_classification": SFT_ROOT / "workplace_classification/train.jsonl",
    "benefit_classification": SFT_ROOT / "benefit_classification/train.jsonl",
    "work_type_classification": SFT_ROOT / "work_type_classification/train.jsonl",
    "settlement_classification": SFT_ROOT / "settlement_classification/train.jsonl",
}

EXPECTED_TASKS = {
    "entity_extraction": 20,
    "job_name_classification": 16,
    "work_time_classification": 4,
    "workplace_classification": 3,
    "benefit_classification": 3,
    "work_type_classification": 1,
    "settlement_classification": 1,
    "taxonomy_leaf_to_parent": 1,
    "taxonomy_level2_to_children": 1,
}
EXPECTED_SOURCES = {"sft_input": 30, "llm_synthetic": 18, "taxonomy": 2}
EXPECTED_DIFFICULTIES = {"easy": 5, "medium": 30, "hard": 15}

ENTITY_KEYS = [
    "position",
    "company",
    "location",
    "workPlace",
    "workMode",
    "salary",
    "salaryPayment",
    "employmentType",
    "schedule",
    "description",
    "responsibilities",
    "requirements",
    "preferredQualifications",
    "education",
    "workExperience",
    "benefits",
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


def entity(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "position": [],
        "company": "",
        "location": "",
        "workPlace": "",
        "workMode": "",
        "salary": "",
        "salaryPayment": "",
        "employmentType": "",
        "schedule": [],
        "description": [],
        "responsibilities": [],
        "requirements": [],
        "preferredQualifications": [],
        "education": "",
        "workExperience": "",
        "benefits": [],
    }
    value.update(updates)
    return value


def extraction_spec(
    *,
    source_id: str,
    chosen: dict[str, object],
    rejected_updates: dict[str, object],
    error: str,
    field: str,
    difficulty: str,
    reason: str,
    input_text: str | None = None,
    sample_id: str | None = None,
) -> dict[str, object]:
    rejected = copy.deepcopy(chosen)
    rejected.update(rejected_updates)
    return {
        "task": "entity_extraction",
        "source_id": source_id,
        "source_sample_id": sample_id,
        "input": input_text,
        "chosen": chosen,
        "rejected": rejected,
        "primary_error_type": error,
        "target_field": field,
        "difficulty": difficulty,
        "review_reason": reason,
    }


def classification_spec(
    *,
    task: str,
    source_id: str,
    chosen: dict[str, object],
    rejected: dict[str, object],
    error: str,
    field: str,
    difficulty: str,
    reason: str,
    input_text: str | None = None,
    sample_id: str | None = None,
) -> dict[str, object]:
    return {
        "task": task,
        "source_id": source_id,
        "source_sample_id": sample_id,
        "input": input_text,
        "chosen": chosen,
        "rejected": rejected,
        "primary_error_type": error,
        "target_field": field,
        "difficulty": difficulty,
        "review_reason": reason,
    }


def job(position: str, level1: str, level2: str, level3: str) -> dict[str, str]:
    return {"position": position, "level1": level1, "level2": level2, "level3": level3}


SPECS: list[dict[str, object]] = [
    # entity_extraction: 12 SFT inputs
    extraction_spec(
        source_id="sq_000595",
        sample_id="EXT_f79373f4113e",
        chosen=entity(
            position=["韩语/朝鲜语翻译"],
            company="XX公司",
            location="上海浦东",
            salaryPayment="每月结算",
            schedule=["弹性打卡"],
        ),
        rejected_updates={"workMode": "现场办公"},
        error="E07_hallucination",
        field="workMode",
        difficulty="medium",
        reason="弹性打卡属于工作时间表达，不能据此补充现场办公。",
    ),
    extraction_spec(
        source_id="sq_006839",
        sample_id="EXT_3cda02d08e53",
        chosen=entity(workPlace="仓库", salaryPayment="月底发工资", schedule=["常白班"]),
        rejected_updates={"position": ["仓库"], "workPlace": ""},
        error="E04_field_confusion",
        field="position|workPlace",
        difficulty="medium",
        reason="仓库是场所而不是岗位名称。",
    ),
    extraction_spec(
        source_id="sq_003966",
        sample_id="EXT_920aaaa19f7f",
        chosen=entity(position=["资深外贸跟单", "外贸跟单"], company="华为科技"),
        rejected_updates={"company": "华为"},
        error="E01_boundary_too_short",
        field="company",
        difficulty="medium",
        reason="公司实体应保留输入中具有标识度的完整名称。",
    ),
    extraction_spec(
        source_id="sq_002401",
        sample_id="EXT_cd72c2dfafba",
        chosen=entity(position=["影视发行"], workExperience="3-5年", benefits=["餐补"]),
        rejected_updates={"position": ["影视"]},
        error="E01_boundary_too_short",
        field="position",
        difficulty="easy",
        reason="影视只是领域，影视发行才是完整岗位实体。",
    ),
    extraction_spec(
        source_id="job_028904",
        sample_id="EXT_b7c56afdc5e1",
        chosen=entity(
            position=["催收员"],
            company="洛阳昊苍科技有限公司",
            location="洛阳",
            salary="7.5千-1.5万·13薪",
            responsibilities=["电话催收", "信息核实", "了解还款困难原因", "提供解决方案", "记录催收过程", "整理相关资料", "配合团队完成催收目标", "提升回款效率", "客户满意度"],
            requirements=["沟通能力", "心理素质", "办公软件操作", "数据分析能力", "工作认真负责", "执行力", "团队协作精神"],
            preferredQualifications=["有相关经验者优先"],
            education="大专及以上学历",
        ),
        rejected_updates={
            "responsibilities": [
                "负责逾期账款的电话催收及信息核实,确保客户按时还款",
                "了解还款困难原因",
                "提供解决方案",
                "记录催收过程",
                "整理相关资料",
                "配合团队完成催收目标",
                "提升回款效率",
                "客户满意度",
            ]
        },
        error="E02_boundary_too_long",
        field="responsibilities",
        difficulty="hard",
        reason="首项应保留电话催收和信息核实两个职责核心，不保留整句结果描述。",
    ),
    extraction_spec(
        source_id="job_000007",
        sample_id="EXT_ef8458f74b90",
        chosen=entity(
            position=["Java 开发专家-ToB 产品研发方向", "Java 开发专家"],
            company="字节跳动",
            location="上海 上海杨浦区",
            workPlace="抖音新江湾广场",
            responsibilities=["大数据研发治理平台设计与研发", "技术架构设计及落地", "理解客户需求", "设计合理技术方案", "客户需求实现及交付", "产品技术创新"],
            requirements=["精通 Java 语言", "JVM", "IO", "多线程", "并发", "网络"],
        ),
        rejected_updates={
            "responsibilities": [
                "参与字节大数据研发治理平台设计与研发,面向ToB场景主导高可扩展技术架构设计及落地",
                "理解客户需求",
                "设计合理技术方案",
                "客户需求实现及交付",
                "产品技术创新",
            ]
        },
        error="E02_boundary_too_long",
        field="responsibilities",
        difficulty="hard",
        reason="chosen 拆分平台研发与技术架构职责，rejected 把两个核心动作合并为整句。",
    ),
    extraction_spec(
        source_id="job_000005",
        sample_id="EXT_36b07a77c36a",
        chosen=entity(
            position=["Java Web 应用开发工程师"],
            company="OOCL",
            location="上海 上海虹口区",
            workPlace="宝华商业广场(四川北路商业街店)",
            responsibilities=[
                "项目开发",
                "需求分析",
                "系统设计",
                "代码编写",
                "单元测试",
                "集成发布",
                "测试用例评审",
                "系统性能测试",
                "代码审查",
                "重构",
                "项目回顾总结",
                "经验分享",
            ],
            requirements=["计算机、软件工程、信息技术相关专业"],
            education="全日制大学本科及以上学历",
        ),
        rejected_updates={
            "responsibilities": [
                "项目开发",
                "需求分析",
                "系统设计",
                "负责数字化系统开发,包括代码编写、单元测试和集成发布等",
                "测试用例评审",
                "系统性能测试",
                "代码审查",
                "重构",
                "项目回顾总结",
                "经验分享",
            ]
        },
        error="E02_boundary_too_long",
        field="responsibilities",
        difficulty="hard",
        reason="代码编写、单元测试和集成发布应拆成简洁可识别项。",
    ),
    extraction_spec(
        source_id="job_000011",
        sample_id="EXT_d48d0aea1cec",
        chosen=entity(
            position=["Java Back-end 개발자 모집", "Java Back-end 개발자"],
            company="成都恩希云科技",
            location="成都 成都武侯区",
            workPlace="太平洋保险金融大厦D8",
            responsibilities=["개발 리딩", "기술 교류", "글로벌 개발", "백엔드", "공통 플랫폼 개발"],
            requirements=["Java", "Web", "C/C++", "Mobile"],
        ),
        rejected_updates={"position": ["Java Back-end 개발자 모집", "Java后端开发工程师"]},
        error="E10_rule_exception",
        field="position[1]",
        difficulty="hard",
        reason="position[1] 不得把源语言岗位名翻译成另一种语言。",
    ),
    extraction_spec(
        source_id="job_007708",
        sample_id="EXT_cd5e49732f86",
        chosen=entity(
            position=["后勤助理"],
            company="合肥养正生物科技有限公司",
            location="杭州 上城区",
            salary="5.5-6.5千",
            responsibilities=[
                "采购计划制定",
                "供应商对接",
                "入库验收管理",
                "办公环境维护",
                "水电设施巡检",
                "保洁监督",
                "绿植养护安排",
                "员工活动",
                "会议接待",
                "考勤统计",
                "档案归档工作",
                "食堂膳食供应",
                "福利发放流程",
                "突发行政事务",
                "协调跨部门资源",
            ],
            requirements=[
                "熟悉办公软件操作",
                "沟通协调能力",
                "突发事件处理能力",
                "服务意识强",
                "工作细致严谨",
                "责任心强",
                "无不良职业记录",
            ],
            preferredQualifications=["行政管理、工商管理或相关专业优先", "持有C1驾照并熟练驾驶者优先考虑"],
            education="大专及以上学历",
            workExperience="1年以上企业后勤或行政助理工作经验",
        ),
        rejected_updates={
            "responsibilities": [
                "负责公司行政物资的采购计划制定、供应商对接及入库验收管理",
                "办公环境维护",
                "水电设施巡检",
                "保洁监督",
                "绿植养护安排",
                "员工活动",
                "会议接待",
                "考勤统计",
                "档案归档工作",
                "食堂膳食供应",
                "福利发放流程",
                "突发行政事务",
                "协调跨部门资源",
            ]
        },
        error="E02_boundary_too_long",
        field="responsibilities",
        difficulty="hard",
        reason="首项应拆出采购计划、供应商对接和验收管理，不保留整句。",
    ),
    extraction_spec(
        source_id="job_002029",
        sample_id="EXT_445e6fab08ae",
        chosen=entity(
            position=["成品仓库(发货统计管理)", "成品仓库管理"],
            company="常州苏格电气有限公司",
            location="常州 新北区",
            salary="5-6千",
            responsibilities=["发货流程管理", "跟踪和统计发货数据", "整理并汇报发货情况", "维护仓库库存记录", "协助盘点工作"],
            requirements=["熟悉仓库操作流程", "熟练使用办公软件", "仓库管理系统", "工作细致认真", "责任心强", "沟通能力", "团队协作精神"],
            preferredQualifications=["物流、仓储、管理等相关专业优先"],
            education="大专及以上学历",
            workExperience="1年以上仓库或相关岗位工作经验",
        ),
        rejected_updates={"position": ["成品仓库"]},
        error="E01_boundary_too_short",
        field="position",
        difficulty="medium",
        reason="仅保留成品仓库会丢失发货统计管理这一岗位标识。",
    ),
    extraction_spec(
        source_id="job_031701",
        sample_id="EXT_5b546bbfe2e1",
        chosen=entity(
            position=["物流专员/助理"],
            company="广东鸿福林人力资源有限公司",
            location="重庆",
            salary="7-9千·13薪",
            schedule=["弹性工作时间"],
            responsibilities=["物流订单的接收", "审核", "系统录入", "安排车辆调度", "货物发运", "跟踪在途状态", "处理异常物流事件", "维护承运商档案", "物流数据统计", "编制报表", "费用对账结算"],
            requirements=["基础数据分析能力", "熟悉物流运作流程及相关法律法规", "能接受短期出差"],
            preferredQualifications=["物流管理、供应链或相关专业优先"],
            education="大专及以上学历",
            workExperience="1 年以上物流操作或助理经验",
        ),
        rejected_updates={"schedule": []},
        error="E08_omission",
        field="schedule",
        difficulty="medium",
        reason="输入明确出现弹性工作时间，不能漏抽。",
    ),
    extraction_spec(
        source_id="job_008477",
        sample_id="EXT_c22f3179ae96",
        chosen=entity(
            position=["ERP成本/财务顾问"],
            company="广东风华高新科技股份有限公司",
            location="肇庆 端州区",
            salary="1.5-2.5万·13薪",
            requirements=["财务管理", "成本管理", "信息化方案设计", "金蝶星瀚ERP系统实施和维护经验", "ERP方案能力", "项目管理能力", "数据分析能力", "职业素养", "保密意识"],
            workExperience="至少两个制造型企业ERP实施经验",
        ),
        rejected_updates={"workMode": "现场办公"},
        error="E07_hallucination",
        field="workMode",
        difficulty="medium",
        reason="输入没有出现现场、远程或混合办公方式。",
    ),
    # entity_extraction: 8 LLM-authored inputs
    extraction_spec(
        source_id="synthetic_entity_001",
        input_text="职位名称：推荐算法工程师\n岗位职责：负责优化跨境电商推荐排序模型，提升点击率与转化率。",
        chosen=entity(position=["推荐算法工程师"], responsibilities=["优化跨境电商推荐排序模型", "提升点击率", "转化率"]),
        rejected_updates={"responsibilities": ["优化推荐模型", "提升点击率", "转化率"]},
        error="E01_boundary_too_short",
        field="responsibilities",
        difficulty="medium",
        reason="优化推荐模型丢失跨境电商和排序两个关键限定，且不是原文片段。",
    ),
    extraction_spec(
        source_id="synthetic_entity_002",
        input_text="职位名称：热管理算法工程师\n任职要求：三年以上新能源汽车电池热管理算法经验，熟练使用MATLAB，接受短期出差。",
        chosen=entity(position=["热管理算法工程师"], requirements=["三年以上新能源汽车电池热管理算法经验", "熟练使用MATLAB", "接受短期出差"], workExperience="三年以上新能源汽车电池热管理算法经验"),
        rejected_updates={"requirements": ["电池热管理算法经验", "熟练使用MATLAB", "接受短期出差"]},
        error="E01_boundary_too_short",
        field="requirements",
        difficulty="hard",
        reason="要求短语不能丢失三年以上和新能源汽车两个识别条件。",
    ),
    extraction_spec(
        source_id="synthetic_entity_003",
        input_text="职位名称：行政前台\n云帆科技在杭州滨江区星耀中心A座招聘行政前台。",
        chosen=entity(position=["行政前台"], company="云帆科技", location="杭州滨江区", workPlace="星耀中心A座"),
        rejected_updates={"workPlace": "杭州滨江区星耀中心A座"},
        error="E04_field_confusion",
        field="location|workPlace",
        difficulty="medium",
        reason="工作场所不应把行政地理位置一并吞入。",
    ),
    extraction_spec(
        source_id="synthetic_entity_004",
        input_text="招聘装卸工，薪资300元/天，每周五发放。",
        chosen=entity(position=["装卸工"], salary="300元/天", salaryPayment="每周五发放"),
        rejected_updates={"salaryPayment": "300元/天"},
        error="E04_field_confusion",
        field="salary|salaryPayment",
        difficulty="medium",
        reason="元/天是薪资单位，每周五发放才是结算信息。",
    ),
    extraction_spec(
        source_id="synthetic_entity_005",
        input_text="职位名称：设备维修工\n公司提供午餐和住宿。",
        chosen=entity(position=["设备维修工"], benefits=["午餐", "住宿"]),
        rejected_updates={"benefits": ["包吃包住"]},
        error="E07_hallucination",
        field="benefits",
        difficulty="medium",
        reason="抽取任务必须保留原文，不能改写成标准福利词。",
    ),
    extraction_spec(
        source_id="synthetic_entity_006",
        input_text="门店招聘暑期兼职生，需到店工作。",
        chosen=entity(employmentType="暑期兼职生", workMode="现场办公", workPlace="门店"),
        rejected_updates={"employmentType": "现场办公", "workMode": ""},
        error="E04_field_confusion",
        field="employmentType|workMode",
        difficulty="medium",
        reason="暑期兼职生是用工类型，到店工作才对应现场办公。",
    ),
    extraction_spec(
        source_id="synthetic_entity_007",
        input_text="团队专注于工业视觉检测。岗位负责缺陷识别模型训练。",
        chosen=entity(description=["团队专注于工业视觉检测"], responsibilities=["缺陷识别模型训练"]),
        rejected_updates={"description": ["团队专注于工业视觉检测", "缺陷识别模型训练"], "responsibilities": []},
        error="E04_field_confusion",
        field="description|responsibilities",
        difficulty="medium",
        reason="团队背景与岗位职责需要分字段表达。",
    ),
    extraction_spec(
        source_id="synthetic_entity_008",
        input_text="用户搜索词：工资高环境好",
        chosen=entity(),
        rejected_updates={"salary": "工资高"},
        error="E07_hallucination",
        field="salary",
        difficulty="easy",
        reason="工资高不是明确薪资范围或金额，应保持 salary 为空。",
    ),
    # job_name_classification: 10 SFT inputs
    classification_spec(
        task="job_name_classification",
        source_id="job_023389",
        sample_id="JOB_NAME_CLASSIFICATION_42ed8ea6ba97",
        chosen=job("临床项目经理", "医疗健康", "临床试验", "临床项目经理"),
        rejected=job("临床项目经理", "项目管理", "项目管理", "临床项目经理"),
        error="E05_category_confusion",
        field="level1|level2",
        difficulty="hard",
        reason="创新药、CRA、GCP 和研究中心语境明确指向医疗健康下的临床试验。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="job_031701",
        sample_id="JOB_NAME_CLASSIFICATION_8c7c89b56623",
        chosen=job("物流专员/助理", "物流/仓储/司机", "物流/运输", "物流专员"),
        rejected=job("物流专员/助理", "物流/仓储/司机", "仓储", "仓库管理员"),
        error="E05_category_confusion",
        field="level2|level3",
        difficulty="medium",
        reason="核心工作是订单、运输协调和在途跟踪，不是仓库库存管理。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="job_008477",
        sample_id="JOB_NAME_CLASSIFICATION_a32849763c16",
        chosen=job("ERP成本/财务顾问", "互联网/AI", "技术项目管理", "实施顾问"),
        rejected=job("ERP成本/财务顾问", "财务/审计/税务", "其他财务岗位", "财务顾问"),
        error="E05_category_confusion",
        field="level1|level2|level3",
        difficulty="hard",
        reason="语境重点是 ERP 系统实施、维护和方案，不是一般财务咨询。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="sq_002401",
        sample_id="JOB_NAME_CLASSIFICATION_65e3a1ae9856",
        chosen=job("影视发行岗位", "直播/影视/传媒", "影视", "影视发行"),
        rejected=job("影视发行岗位", "直播/影视/传媒", "影视", "影视策划"),
        error="E05_category_confusion",
        field="level3",
        difficulty="medium",
        reason="岗位实体明确为影视发行，不能混为前期影视策划。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="job_007708",
        sample_id="JOB_NAME_CLASSIFICATION_4ac8b12a458b",
        chosen=job("后勤助理", "人力/行政/法务", "行政", "后勤"),
        rejected=job("后勤助理", "人力/行政/法务", "行政", "行政专员/助理"),
        error="E05_category_confusion",
        field="level3",
        difficulty="medium",
        reason="物资、环境和物料、食堂和突发行政事务以企业后勤为核心。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="job_028904",
        sample_id="JOB_NAME_CLASSIFICATION_be6c4cfc9ffc",
        chosen=job("催收员", "金融", "中后台", "催收员"),
        rejected=job("催收员", "销售", "金融销售", "信用卡销售"),
        error="E05_category_confusion",
        field="level1|level2|level3",
        difficulty="medium",
        reason="工作目标是逾期账款催收，不是销售信用卡。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="sq_012796",
        sample_id="JOB_NAME_CLASSIFICATION_b7d12eb8bc0e",
        chosen=job("旅游产品经理", "酒店/旅游", "旅游服务", "旅游产品经理"),
        rejected=job("旅游产品经理", "产品", "产品经理", "产品经理"),
        error="E05_category_confusion",
        field="level1|level2|level3",
        difficulty="hard",
        reason="岗位实体含明确旅游领域，应选择旅游服务下的最细三级类别。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="sq_005198",
        sample_id="JOB_NAME_CLASSIFICATION_06b89f26073f",
        chosen=job("资深临床监查员", "医疗健康", "临床试验", "临床监查员"),
        rejected=job("资深临床监查员", "医疗健康", "临床试验", "临床项目经理"),
        error="E05_category_confusion",
        field="level3",
        difficulty="medium",
        reason="临床监查员与临床项目经理职责层级不同，实体已明确监查员。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="sq_000595",
        sample_id="JOB_NAME_CLASSIFICATION_15ece59bbf75",
        chosen=job("韩语/朝鲜语翻译", "咨询/翻译/法律", "翻译", "韩语/朝鲜语翻译"),
        rejected=job("韩语/朝鲜语翻译", "教育培训", "教师", "其他外语教师"),
        error="E05_category_confusion",
        field="level1|level2|level3",
        difficulty="easy",
        reason="输入是翻译岗位，没有教学语义。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="job_002029",
        sample_id="JOB_NAME_CLASSIFICATION_a4ad217eb9a1",
        chosen=job("成品仓库(发货统计管理)", "物流/仓储/司机", "仓储", "仓库管理员"),
        rejected=job("成品仓库(发货统计管理)", "物流/仓储/司机", "配送理货", "配/理/拣/发货"),
        error="E05_category_confusion",
        field="level2|level3",
        difficulty="medium",
        reason="职责覆盖库存记录、盘点和完整发货流程管理，不只是配拣发货。",
    ),
    # job_name_classification: 6 LLM-authored inputs
    classification_spec(
        task="job_name_classification",
        source_id="synthetic_job_001",
        input_text="岗位实体：AI产品经理\n原文上下文：负责大模型应用产品规划、需求分析和版本路线图，协调算法与研发团队落地。",
        chosen=job("AI产品经理", "产品", "产品经理", "AI产品经理"),
        rejected=job("AI产品经理", "互联网/AI", "人工智能", "算法工程师"),
        error="E05_category_confusion",
        field="level1|level2|level3",
        difficulty="medium",
        reason="职责是产品规划和需求管理，不是算法研发。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="synthetic_job_002",
        input_text="岗位实体：汽车电子工程师\n原文上下文：负责整车控制器硬件方案设计、车规器件选型与台架验证，服务乘用车研发项目。",
        chosen=job("汽车电子工程师", "汽车", "汽车研发/制造", "汽车电子工程师"),
        rejected=job("汽车电子工程师", "生产制造", "汽车研发/制造", "汽车电子工程师"),
        error="E05_category_confusion",
        field="level1",
        difficulty="hard",
        reason="整车、车规和乘用车研发上下文应落在汽车一级类。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="synthetic_job_003",
        input_text="岗位实体：跨境电商运营\n原文上下文：负责亚马逊店铺投放、Listing优化、站内活动和销售数据复盘。",
        chosen=job("跨境电商运营", "客服/运营", "电商运营", "跨境电商运营"),
        rejected=job("跨境电商运营", "客服/运营", "电商运营", "国内电商运营"),
        error="E05_category_confusion",
        field="level3",
        difficulty="medium",
        reason="亚马逊和 Listing 明确属于跨境电商。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="synthetic_job_004",
        input_text="岗位实体：直播运营\n原文上下文：负责直播间排期、投流、用户互动策略和成交转化复盘，不承担摄像或导播工作。",
        chosen=job("直播运营", "客服/运营", "内容运营", "直播运营"),
        rejected=job("直播运营", "直播/影视/传媒", "直播", "直播运营"),
        error="E05_category_confusion",
        field="level1|level2",
        difficulty="hard",
        reason="投流、互动策略和转化复盘是运营职责，不是直播制作。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="synthetic_job_005",
        input_text="岗位实体：法务专员\n原文上下文：审核业务合同、处理劳动争议并协助外部律师推进诉讼材料。",
        chosen=job("法务专员", "人力/行政/法务", "法律服务", "法务专员/助理"),
        rejected=job("法务专员", "咨询/翻译/法律", "法律服务", "律师助理"),
        error="E05_category_confusion",
        field="level1|level3",
        difficulty="medium",
        reason="这是企业内部法务岗位，不是律所律师助理。",
    ),
    classification_spec(
        task="job_name_classification",
        source_id="synthetic_job_006",
        input_text="岗位实体：自然语言处理算法工程师\n原文上下文：负责中文分词、文本分类、信息抽取和大模型评测算法研发。",
        chosen=job("自然语言处理算法工程师", "互联网/AI", "人工智能", "自然语言处理算法"),
        rejected=job("自然语言处理算法工程师", "互联网/AI", "人工智能", "算法工程师"),
        error="E05_category_confusion",
        field="level3",
        difficulty="hard",
        reason="分词、文本分类和信息抽取支持最细粒度 NLP 类别。",
    ),
    # work_time_classification
    classification_spec(
        task="work_time_classification",
        source_id="sq_000595",
        sample_id="WORK_TIME_CLASSIFICATION_d0a0b6ebee5f",
        chosen={"entity": "弹性打卡", "level1": "弹性工时制", "level2": "核心弹性", "fullCategory": "弹性工时制-核心弹性"},
        rejected={"entity": "弹性打卡", "level1": "任务交付制", "level2": "时间自由", "fullCategory": "任务交付制-时间自由"},
        error="E05_category_confusion",
        field="fullCategory",
        difficulty="medium",
        reason="弹性打卡仍有考勤约束，不等于以任务交付为准的时间自由。",
    ),
    classification_spec(
        task="work_time_classification",
        source_id="sq_006839",
        sample_id="WORK_TIME_CLASSIFICATION_b66f06e0c99e",
        chosen={"entity": "常白班", "level1": "白班制", "level2": "长白班", "fullCategory": "白班制-长白班"},
        rejected={"entity": "常白班", "level1": "轮班制", "level2": "两班倒", "fullCategory": "轮班制-两班倒"},
        error="E05_category_confusion",
        field="fullCategory",
        difficulty="easy",
        reason="常白班是固定白班，不包含早晚两班轮换。",
    ),
    classification_spec(
        task="work_time_classification",
        source_id="synthetic_time_001",
        input_text="实体：仅上午8点到12点，下午不排班",
        chosen={"entity": "仅上午8点到12点，下午不排班", "level1": "半天班", "level2": "上午班", "fullCategory": "半天班-上午班"},
        rejected={"entity": "仅上午8点到12点，下午不排班", "level1": "小时工", "level2": "上午钟点", "fullCategory": "小时工-上午钟点"},
        error="E05_category_confusion",
        field="fullCategory",
        difficulty="medium",
        reason="输入描述固定半天班次，没有小时计酬或钟点工语义。",
    ),
    classification_spec(
        task="work_time_classification",
        source_id="synthetic_time_002",
        input_text="实体：每天晚上11点到次日早上7点",
        chosen={"entity": "每天晚上11点到次日早上7点", "level1": "夜班制", "level2": "通宵班", "fullCategory": "夜班制-通宵班"},
        rejected={"entity": "每天晚上11点到次日早上7点", "level1": "夜班制", "level2": "大夜班", "fullCategory": "夜班制-大夜班"},
        error="E05_category_confusion",
        field="fullCategory",
        difficulty="hard",
        reason="23:00 至次日 7:00 覆盖整夜，符合通宵班定义。",
    ),
    # workplace_classification
    classification_spec(
        task="workplace_classification",
        source_id="sq_006839",
        sample_id="WORKPLACE_CLASSIFICATION_fe493a3106ac",
        chosen={"rawWorkPlace": "仓库", "level1": "仓库", "level2": "仓库"},
        rejected={"rawWorkPlace": "仓库", "level1": "工厂", "level2": "厂矿"},
        error="E05_category_confusion",
        field="level1|level2",
        difficulty="medium",
        reason="仓库直接命中仓库-仓库，不应泛化成生产工厂。",
    ),
    classification_spec(
        task="workplace_classification",
        source_id="job_000002",
        sample_id="WORKPLACE_CLASSIFICATION_d7d931e67334",
        chosen={"rawWorkPlace": "用友软件园东区19A", "level1": "公司", "level2": "园区"},
        rejected={"rawWorkPlace": "用友软件园东区19A", "level1": "公司", "level2": "写字楼"},
        error="E05_category_confusion",
        field="level2",
        difficulty="medium",
        reason="实体明确包含软件园，应优先按公司园区分类，而不是泛化为写字楼。",
    ),
    classification_spec(
        task="workplace_classification",
        source_id="synthetic_workplace_001",
        input_text="实体：环球影城大酒店",
        chosen={"rawWorkPlace": "环球影城大酒店", "level1": "酒店", "level2": "星级酒店"},
        rejected={"rawWorkPlace": "环球影城大酒店", "level1": "旅游景点", "level2": "游乐园"},
        error="E05_category_confusion",
        field="level1|level2",
        difficulty="hard",
        reason="实体中心词是大酒店，不能因品牌含环球影城而分类为游乐园。",
    ),
    # benefit_classification
    classification_spec(
        task="benefit_classification",
        source_id="sq_017604",
        sample_id="BENEFIT_CLASSIFICATION_c0688771e844",
        chosen={"entity": "交通补贴", "standardBenefit": "交通补贴/班车"},
        rejected={"entity": "交通补贴", "standardBenefit": "住房补贴/宿舍"},
        error="E05_category_confusion",
        field="standardBenefit",
        difficulty="medium",
        reason="交通补贴应归交通补贴/班车。",
    ),
    classification_spec(
        task="benefit_classification",
        source_id="sq_002401",
        sample_id="BENEFIT_CLASSIFICATION_9ed7c8fe84ab",
        chosen={"entity": "餐补", "standardBenefit": "餐补/工作餐"},
        rejected={"entity": "餐补", "standardBenefit": "包吃"},
        error="E05_category_confusion",
        field="standardBenefit",
        difficulty="medium",
        reason="餐补是补助，不等于用人方承诺包吃。",
    ),
    classification_spec(
        task="benefit_classification",
        source_id="synthetic_benefit_001",
        input_text="实体：管吃住",
        chosen={"entity": "管吃住", "standardBenefit": "包吃包住"},
        rejected={"entity": "管吃住", "standardBenefit": "包吃"},
        error="E01_boundary_too_short",
        field="standardBenefit",
        difficulty="medium",
        reason="管吃住同时包含饮食和住宿，不能漏掉住宿权益。",
    ),
    # work_type_classification
    classification_spec(
        task="work_type_classification",
        source_id="sq_012796",
        sample_id="WORK_TYPE_CLASSIFICATION_b782bb82fb97",
        chosen={"entity": "兼职", "category": "兼职工"},
        rejected={"entity": "兼职", "category": "短期工"},
        error="E05_category_confusion",
        field="category",
        difficulty="easy",
        reason="兼职描述工作安排，不必然表示短期用工。",
    ),
    # settlement_classification
    classification_spec(
        task="settlement_classification",
        source_id="sq_000595",
        sample_id="SETTLEMENT_CLASSIFICATION_a8033371207f",
        chosen={"entity": "每月结算", "category": "月结"},
        rejected={"entity": "每月结算", "category": "日结"},
        error="E05_category_confusion",
        field="category",
        difficulty="medium",
        reason="每月结算明确属于月结。",
    ),
    # taxonomy tasks
    classification_spec(
        task="taxonomy_leaf_to_parent",
        source_id="job_cate.json",
        input_text="三级职位：韩语/朝鲜语翻译",
        chosen={"level3": "韩语/朝鲜语翻译", "level2": "翻译", "level1": "咨询/翻译/法律"},
        rejected={"level3": "韩语/朝鲜语翻译", "level2": "教师", "level1": "教育培训"},
        error="E06_hierarchy_inconsistency",
        field="level1|level2",
        difficulty="medium",
        reason="job_cate.json 中该唯一三级名称的父路径是咨询/翻译/法律-翻译。",
    ),
    classification_spec(
        task="taxonomy_level2_to_children",
        source_id="job_cate.json",
        input_text="二级职位完整路径：互联网/AI/人工智能",
        chosen={
            "level1": "互联网/AI",
            "level2": "人工智能",
            "level3Children": [
                "图像算法",
                "自然语言处理算法",
                "大模型算法",
                "数据挖掘",
                "规控算法",
                "SLAM算法",
                "推荐算法",
                "搜索算法",
                "语音算法",
                "风控算法",
                "高性能计算工程师",
                "算法工程师",
                "算法研究员",
                "机器学习",
                "深度学习",
                "自动驾驶系统工程师",
                "数据标注/AI训练师",
            ],
        },
        rejected={
            "level1": "互联网/AI",
            "level2": "人工智能",
            "level3Children": [
                "图像算法",
                "自然语言处理算法",
                "数据挖掘",
                "规控算法",
                "SLAM算法",
                "推荐算法",
                "搜索算法",
                "语音算法",
                "风控算法",
                "高性能计算工程师",
                "算法工程师",
                "算法研究员",
                "机器学习",
                "深度学习",
                "自动驾驶系统工程师",
                "数据标注/AI训练师",
            ],
        },
        error="E08_omission",
        field="level3Children",
        difficulty="hard",
        reason="rejected 漏掉 job_cate.json 中的大模型算法子类。",
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip()


def load_frozen_inputs() -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    normalized: set[str] = set()
    with FROZEN_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for message in row.get("conversations", []):
                if message.get("from") == "human":
                    value = str(message.get("value", ""))
                    exact.add(value)
                    normalized.add(normalized_text(value))
                    break
    return exact, normalized


def extract_training_prompts() -> dict[str, str]:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    section = text.split("## 微调训练提示词", 1)[1]
    prompts: dict[str, str] = {}
    pattern = re.compile(r"### `([^`]+)`\s*\n\n```text\n(.*?)\n```", re.S)
    for task, prompt in pattern.findall(section):
        prompts[task] = prompt.rstrip() + "\n"
    missing = set(TASK_SCHEMAS) - set(prompts)
    if missing:
        raise ValueError(f"missing training prompts: {sorted(missing)}")
    return prompts


def load_sft_inputs() -> dict[str, tuple[str, str, str]]:
    """Read only sample_id/source_id/input; deliberately ignore every output field."""
    wanted = {str(spec["source_sample_id"]) for spec in SPECS if spec.get("source_sample_id")}
    found: dict[str, tuple[str, str, str]] = {}
    for task, path in TASK_FILES.items():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if sample_id in wanted:
                    found[sample_id] = (task, str(row.get("source_id", "")), str(row["input"]))
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"missing SFT inputs: {sorted(missing)}")
    return found


def candidate_list(prompt: str) -> list[str]:
    match = re.search(r"候选分类体系（必须严格遵守）：(\[[^\n]+\])", prompt)
    if not match:
        return []
    return json.loads(match.group(1))


def validate_entity_spans(input_text: str, value: dict[str, object]) -> list[str]:
    issues: list[str] = []
    for key, item in value.items():
        if key == "workMode":
            continue
        values = item if isinstance(item, list) else [item]
        for index, part in enumerate(values):
            if not part:
                continue
            if key == "position" and index == 1:
                continue
            if not isinstance(part, str) or part not in input_text:
                issues.append(f"chosen {key}[{index}] not found in input: {part!r}")
    return issues


def validate_pair(
    record: dict[str, object], prompts: dict[str, str], job_cate: dict[str, dict[str, list[str]]]
) -> list[str]:
    task = str(record["task"])
    chosen = record["chosen"]
    rejected = record["rejected"]
    assert isinstance(chosen, dict) and isinstance(rejected, dict)
    issues: list[str] = []
    expected_schema = TASK_SCHEMAS[task]
    if set(chosen) != expected_schema:
        issues.append(f"chosen schema mismatch: {sorted(chosen)}")
    if set(rejected) != expected_schema:
        issues.append(f"rejected schema mismatch: {sorted(rejected)}")
    if chosen == rejected:
        issues.append("chosen equals rejected")

    input_text = str(record["input"])
    if task == "entity_extraction":
        issues.extend(validate_entity_spans(input_text, chosen))
    elif task == "job_name_classification":
        l1, l2, l3 = chosen["level1"], chosen["level2"], chosen["level3"]
        if l1 not in job_cate or l2 not in job_cate[l1] or l3 not in job_cate[l1][l2]:
            issues.append(f"chosen job path outside job_cate.json: {l1}/{l2}/{l3}")
        r1, r2, r3 = rejected["level1"], rejected["level2"], rejected["level3"]
        if r1 not in job_cate or r2 not in job_cate[r1] or r3 not in job_cate[r1][r2]:
            issues.append(f"rejected job path outside job_cate.json: {r1}/{r2}/{r3}")
    elif task == "taxonomy_leaf_to_parent":
        l1, l2, l3 = chosen["level1"], chosen["level2"], chosen["level3"]
        if l1 not in job_cate or l2 not in job_cate[l1] or l3 not in job_cate[l1][l2]:
            issues.append("chosen leaf parent path outside job_cate.json")
    elif task == "taxonomy_level2_to_children":
        l1, l2 = chosen["level1"], chosen["level2"]
        if l1 not in job_cate or l2 not in job_cate[l1]:
            issues.append("chosen level2 path outside job_cate.json")
        elif chosen["level3Children"] != job_cate[l1][l2]:
            issues.append("chosen children differ from job_cate.json")
    elif task in {"work_type_classification", "settlement_classification"}:
        allowed = set(candidate_list(prompts[task]))
        if chosen["category"] not in allowed or rejected["category"] not in allowed:
            issues.append("category outside prompt candidates")
    elif task == "benefit_classification":
        allowed = set(candidate_list(prompts[task]))
        if chosen["standardBenefit"] not in allowed or rejected["standardBenefit"] not in allowed:
            issues.append("benefit outside prompt candidates")
    elif task == "work_time_classification":
        allowed = set(candidate_list(prompts[task]))
        for side, value in (("chosen", chosen), ("rejected", rejected)):
            if value["fullCategory"] not in allowed:
                issues.append(f"{side} work time outside prompt candidates")
            if value["fullCategory"] == "其他":
                if value["level1"] or value["level2"]:
                    issues.append(f"{side} other must have empty parents")
            elif value["fullCategory"] != f"{value['level1']}-{value['level2']}":
                issues.append(f"{side} work time hierarchy mismatch")
    elif task == "workplace_classification":
        pairs = {tuple(item.split("-", 1)) for item in candidate_list(prompts[task])}
        for side, value in (("chosen", chosen), ("rejected", rejected)):
            if (value["level1"], value["level2"]) not in pairs:
                issues.append(f"{side} workplace path outside prompt candidates")
    return issues


def main() -> None:
    prompts = extract_training_prompts()
    job_cate = json.loads(JOB_CATE_FILE.read_text(encoding="utf-8"))
    sft_inputs = load_sft_inputs()

    records: list[dict[str, object]] = []
    for index, spec in enumerate(SPECS, 1):
        record = copy.deepcopy(spec)
        sample_id = record.pop("source_sample_id", None)
        inline_input = record.get("input")
        if sample_id:
            task, source_id, input_text = sft_inputs[str(sample_id)]
            if task != record["task"] or source_id != record["source_id"]:
                raise ValueError(f"source identity mismatch for {sample_id}")
            if inline_input is not None:
                raise ValueError(f"SFT-backed record must not embed input: {sample_id}")
            source_type = "sft_input"
            record["input"] = input_text
        elif record["source_id"] == "job_cate.json":
            source_type = "taxonomy"
        else:
            source_type = "llm_synthetic"

        record.update(
            {
                "pair_id": f"DPO_V1_1_P50_{index:04d}",
                "dataset_version": "llm-generated-v1.1-pilot50",
                "prompt_version": "recruitment_label_prompts_v2",
                "system_prompt_ref": f"label_prompts_v2.md#微调训练提示词/{record['task']}",
                "source_type": source_type,
                "generation_method": "gpt5_independent_relabel",
                "chosen_provenance": "llm_independent_from_input_and_locked_candidates",
                "rejected_provenance": "llm_authored_hard_negative",
                "human_review_status": "pending_user_confirmation",
            }
        )
        records.append(record)

    task_counts = Counter(str(row["task"]) for row in records)
    source_counts = Counter(str(row["source_type"]) for row in records)
    difficulty_counts = Counter(str(row["difficulty"]) for row in records)
    error_counts = Counter(str(row["primary_error_type"]) for row in records)

    issues: list[dict[str, object]] = []
    if dict(task_counts) != EXPECTED_TASKS:
        issues.append({"scope": "dataset", "issue": "task quota mismatch", "actual": dict(task_counts)})
    if dict(source_counts) != EXPECTED_SOURCES:
        issues.append({"scope": "dataset", "issue": "source quota mismatch", "actual": dict(source_counts)})
    if dict(difficulty_counts) != EXPECTED_DIFFICULTIES:
        issues.append({"scope": "dataset", "issue": "difficulty quota mismatch", "actual": dict(difficulty_counts)})
    pair_ids = [str(row["pair_id"]) for row in records]
    inputs = [str(row["input"]) for row in records]
    if len(pair_ids) != len(set(pair_ids)):
        issues.append({"scope": "dataset", "issue": "duplicate pair_id"})
    if len(inputs) != len(set(inputs)):
        issues.append({"scope": "dataset", "issue": "duplicate input"})
    frozen_exact, frozen_normalized = load_frozen_inputs()
    exact_overlap = sorted(value for value in inputs if value in frozen_exact)
    normalized_overlap = sorted(value for value in inputs if normalized_text(value) in frozen_normalized)
    if exact_overlap:
        issues.append({"scope": "dataset", "issue": "frozen exact input overlap", "values": exact_overlap})
    if normalized_overlap:
        issues.append({"scope": "dataset", "issue": "frozen normalized input overlap", "values": normalized_overlap})
    for row in records:
        for issue in validate_pair(row, prompts, job_cate):
            issues.append({"scope": row["pair_id"], "issue": issue})

    if issues:
        raise ValueError(json.dumps(issues, ensure_ascii=False, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review_path = OUT_DIR / "pilot50_review.jsonl"
    sharegpt_path = OUT_DIR / "pilot50_sharegpt.jsonl"
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )

    sharegpt_rows = []
    for row in records:
        task = str(row["task"])
        sharegpt_rows.append(
            {
                "conversations": [
                    {"from": "system", "value": prompts[task].rstrip()},
                    {"from": "human", "value": row["input"]},
                ],
                "chosen": {"from": "gpt", "value": json.dumps(row["chosen"], ensure_ascii=False, separators=(",", ":"))},
                "rejected": {"from": "gpt", "value": json.dumps(row["rejected"], ensure_ascii=False, separators=(",", ":"))},
            }
        )
    sharegpt_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in sharegpt_rows),
        encoding="utf-8",
    )

    manifest = {
        "dataset": "recruitment_dpo_llm_generated_v1_1_pilot50",
        "status": "pending_user_confirmation",
        "total_pairs": len(records),
        "sft_outputs_used": False,
        "chosen_generation": "gpt5_independent_relabel",
        "task_distribution": dict(task_counts),
        "source_distribution": dict(source_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "error_distribution": dict(error_counts),
        "authorities": {
            "job_taxonomy": "job_cate.json",
            "other_task_candidates": "code/data/task_info/label_prompts_v2.md",
            "job_cate_sha256": sha256(JOB_CATE_FILE),
            "label_prompts_sha256": sha256(PROMPT_FILE),
        },
        "files": {"review": review_path.name, "sharegpt": sharegpt_path.name},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    validation = {
        "valid": True,
        "rows": len(records),
        "issue_count": 0,
        "issues": [],
        "all_chosen_rejected_json_objects": True,
        "all_schemas_exact": True,
        "all_chosen_entity_spans_valid": True,
        "all_job_paths_in_job_cate": True,
        "all_other_categories_in_prompt_candidates": True,
        "chosen_not_equal_rejected": True,
        "duplicate_pair_ids": 0,
        "duplicate_inputs": 0,
        "frozen_exact_input_overlap": 0,
        "frozen_normalized_input_overlap": 0,
        "sft_outputs_read_by_generator": False,
    }
    (OUT_DIR / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_lines = [
        "# DPO v1.1 Pilot 50 Review Summary",
        "",
        "> 本批次仅供用户确认；所有 `human_review_status` 均为 `pending_user_confirmation`。",
        "",
        "## 总览",
        "",
        f"- 样本数：{len(records)}",
        "- SFT 旧 output 使用：否",
        "- chosen：大模型依据原始 input 与锁定候选独立生成",
        "- 职位类别：仅 `job_cate.json`",
        "- 其余分类候选：仅 `label_prompts_v2.md` 对应任务",
        "",
        "## 任务分布",
        "",
        "| 任务 | 数量 |",
        "| --- | ---: |",
    ]
    summary_lines.extend(f"| `{task}` | {count} |" for task, count in task_counts.items())
    summary_lines.extend(["", "## 来源分布", "", "| 来源 | 数量 |", "| --- | ---: |"]) 
    summary_lines.extend(f"| `{name}` | {count} |" for name, count in source_counts.items())
    summary_lines.extend(["", "## 难度分布", "", "| 难度 | 数量 |", "| --- | ---: |"]) 
    summary_lines.extend(f"| `{name}` | {count} |" for name, count in difficulty_counts.items())
    summary_lines.extend(["", "## 错误类型分布", "", "| 错误类型 | 数量 |", "| --- | ---: |"]) 
    summary_lines.extend(f"| `{name}` | {count} |" for name, count in error_counts.items())
    summary_lines.extend(
        [
            "",
            "## 自动校验",
            "",
            "- 50/50 chosen 与 rejected schema 正确。",
            "- 20/20 实体抽取 chosen 通过原文 span 校验（允许的 position[1]/workMode 例外除外）。",
            "- 18/18 职位分类与 taxonomy chosen 通过 `job_cate.json` 校验。",
            "- 其余分类 chosen/rejected 均属于对应提示词候选。",
            "- 重复 pair_id：0；重复 input：0；冻结集 exact/normalized input 重叠：0。",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    review_lines = [
        "# DPO v1.1 Pilot 50 人工确认表",
        "",
        "> 该文件用于逐条确认，尚未标记为正式训练数据。",
        "",
    ]
    current_task = None
    for row in records:
        if row["task"] != current_task:
            current_task = row["task"]
            review_lines.extend([f"## `{current_task}`", ""])
        review_lines.extend(
            [
                f"### {row['pair_id']}",
                "",
                f"- 来源：`{row['source_type']}` / `{row['source_id']}`",
                f"- 难度：`{row['difficulty']}`",
                f"- 错误类型：`{row['primary_error_type']}`",
                f"- 目标字段：`{row['target_field']}`",
                f"- 偏好理由：{row['review_reason']}",
                "",
                "输入：",
                "",
                "```text",
                str(row["input"]),
                "```",
                "",
                "Chosen：",
                "",
                "```json",
                json.dumps(row["chosen"], ensure_ascii=False, indent=2),
                "```",
                "",
                "Rejected：",
                "",
                "```json",
                json.dumps(row["rejected"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    (OUT_DIR / "pilot50_review.md").write_text("\n".join(review_lines), encoding="utf-8")

    print(json.dumps({"output_dir": str(OUT_DIR), "rows": len(records), "valid": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
