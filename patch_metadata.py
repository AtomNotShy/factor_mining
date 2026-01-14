import json
from pathlib import Path

metadata_path = Path("data/backtests/metadata.json")
if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    modified = False
    for run_id, record in data.items():
        if record.get("strategy_name") == "unknown":
            # 根据 symbol 或其他特征判断，这里我们由于最近都在跑 us_etf_momentum，且 symbol 匹配
            if "QQQ" in record.get("symbol", ""):
                record["strategy_name"] = "us_etf_momentum"
                modified = True
    
    if modified:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Successfully patched metadata.json")
    else:
        print("No 'unknown' strategies found to patch.")
else:
    print("metadata.json not found.")
