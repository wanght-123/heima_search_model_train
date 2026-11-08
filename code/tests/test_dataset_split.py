"""数据切分模块的最小测试。"""

from job_finetune_course.dataset_split import save_splits, source_group, split_by_source


def test_same_source_always_in_same_split(tmp_path) -> None:
    rows = [
        {
            "sample_id": f"sample-{index}-{task}",
            "source_id": f"source-{index}",
            "task": task,
        }
        for index in range(100)
        for task in ("entity_extraction", "job_name_classification")
    ]
    splits = split_by_source(rows)
    location = {}
    for split, split_rows in splits.items():
        for row in split_rows:
            group = source_group(row)
            assert group not in location or location[group] == split
            location[group] = split
    assert sum(len(split_rows) for split_rows in splits.values()) == len(rows)
    manifest = save_splits(tmp_path / "split", splits, seed=20260614)
    assert manifest["source_group_leakage"] == 0
    assert sum(manifest["split_rows"].values()) == len(rows)
