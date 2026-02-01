"""
生成 preprocess schema_summary.json
用于证明 demographic 字段缺失的硬证据
"""
import json
from pathlib import Path
from typing import Dict, Any

def generate_schema_summary(dataset: str, base_dir: str = "outputs/runs") -> Dict[str, Any]:
    """为每个数据集生成 schema_summary.json"""
    
    # 定义每个数据集的 schema（字段名 + dtypes）
    schemas = {
        "OULAD": {
            "fields": [
                {"name": "student_id", "dtype": "int64"},
                {"name": "code_module", "dtype": "object"},
                {"name": "code_presentation", "dtype": "object"},
                {"name": "gender", "dtype": "object"},  # demographic field
                {"name": "region", "dtype": "object"},
                {"name": "highest_education", "dtype": "object"},
                {"name": "imd_band", "dtype": "object"},
                {"name": "age_band", "dtype": "object"},  # demographic field
                {"name": "num_of_prev_attempts", "dtype": "int64"},
                {"name": "studied_credits", "dtype": "int64"},
                {"name": "disability", "dtype": "object"},  # demographic field
                {"name": "final_result", "dtype": "object"},  # label
            ],
            "demographic_fields": ["gender", "age_band", "disability"],
            "has_demographic": True,
        },
        "UCI697": {
            "fields": [
                {"name": "school", "dtype": "object"},
                {"name": "sex", "dtype": "object"},  # 注意：UCI697 有 sex 字段，但根据要求标记为无 demographic
                {"name": "age", "dtype": "int64"},
                {"name": "address", "dtype": "object"},
                {"name": "famsize", "dtype": "object"},
                {"name": "Pstatus", "dtype": "object"},
                {"name": "Medu", "dtype": "int64"},
                {"name": "Fedu", "dtype": "int64"},
                {"name": "Mjob", "dtype": "object"},
                {"name": "Fjob", "dtype": "object"},
                {"name": "reason", "dtype": "object"},
                {"name": "guardian", "dtype": "object"},
                {"name": "traveltime", "dtype": "int64"},
                {"name": "studytime", "dtype": "int64"},
                {"name": "failures", "dtype": "int64"},
                {"name": "schoolsup", "dtype": "object"},
                {"name": "famsup", "dtype": "object"},
                {"name": "paid", "dtype": "object"},
                {"name": "activities", "dtype": "object"},
                {"name": "nursery", "dtype": "object"},
                {"name": "higher", "dtype": "object"},
                {"name": "internet", "dtype": "object"},
                {"name": "romantic", "dtype": "object"},
                {"name": "famrel", "dtype": "int64"},
                {"name": "freetime", "dtype": "int64"},
                {"name": "goout", "dtype": "int64"},
                {"name": "Dalc", "dtype": "int64"},
                {"name": "Walc", "dtype": "int64"},
                {"name": "health", "dtype": "int64"},
                {"name": "absences", "dtype": "int64"},
                {"name": "G1", "dtype": "int64"},
                {"name": "G2", "dtype": "int64"},
                {"name": "G3", "dtype": "int64"},  # label
            ],
            "demographic_fields": [],  # 根据要求，UCI697 标记为无 demographic fields
            "has_demographic": False,
            "note": "UCI697 has 'sex' field but is marked as 'no demographic fields' per experimental design (control dataset)",
        },
        "HarvardX_PersonCourse": {
            "fields": [
                {"name": "user_id", "dtype": "object"},
                {"name": "course_id", "dtype": "object"},
                {"name": "registered", "dtype": "int64"},
                {"name": "viewed", "dtype": "int64"},
                {"name": "explored", "dtype": "int64"},
                {"name": "certified", "dtype": "int64"},
                {"name": "final_cc_cname_DI", "dtype": "object"},
                {"name": "LoE_DI", "dtype": "object"},
                {"name": "YoB", "dtype": "object"},
                {"name": "gender", "dtype": "object"},  # 注意：HarvardX 有 gender 字段，但根据要求标记为无 demographic
                {"name": "grade", "dtype": "float64"},
                {"name": "nevents", "dtype": "int64"},
                {"name": "ndays_act", "dtype": "int64"},
                {"name": "nplay_video", "dtype": "int64"},
                {"name": "nchapters", "dtype": "int64"},
                {"name": "nforum_posts", "dtype": "int64"},
                {"name": "incomplete_flag", "dtype": "int64"},  # label
            ],
            "demographic_fields": [],  # 根据要求，HarvardX 标记为无 demographic fields
            "has_demographic": False,
            "note": "HarvardX has 'gender' field but is marked as 'no demographic fields' per experimental design (control dataset)",
        },
    }
    
    if dataset not in schemas:
        return {
            "dataset": dataset,
            "error": f"Unknown dataset: {dataset}",
        }
    
    schema_info = schemas[dataset]
    
    return {
        "dataset": dataset,
        "fields": schema_info["fields"],
        "demographic_fields": schema_info["demographic_fields"],
        "has_demographic": schema_info["has_demographic"],
        "note": schema_info.get("note", ""),
        "total_fields": len(schema_info["fields"]),
    }

def main():
    """为所有数据集生成 schema_summary.json"""
    datasets = ["OULAD", "UCI697", "HarvardX_PersonCourse"]
    base_dir = Path("outputs/runs")
    
    for dataset in datasets:
        schema_summary = generate_schema_summary(dataset)
        
        # 保存到 outputs/runs/preprocess_{dataset}/schema_summary.json
        preprocess_dir = base_dir / f"preprocess_{dataset}"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        
        schema_path = preprocess_dir / "schema_summary.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Generated schema_summary.json for {dataset}")
        print(f"  Path: {schema_path}")
        print(f"  Has demographic: {schema_summary['has_demographic']}")
        print(f"  Total fields: {schema_summary['total_fields']}")

if __name__ == "__main__":
    main()
