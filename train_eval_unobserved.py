import os
import sys
import json
import pandas as pd

# Portable: this folder (wherever the script actually lives), not a fixed session path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_eval_v2 import JIHU, select_features, run_case, log

# "미관측 지점 추정" 프레이밍: RWIS 관측소 "자기 자신"의 실시간 기상 센서 실측값은
# 입력에서 제외한다. 왜냐하면 실제 RWIS가 없는 지점에서는 이런 값 자체를 가질 수 없기
# 때문이다. 위경도/고도, 외부(AWS/ASOS/안개)까지의 거리, 시간 주기성은 어떤 지점에서든
# 항상 구할 수 있는 정보이므로 유지한다. 외부(AWS/ASOS/안개) 관측값과 교량/터널 구조
# 데이터도 그대로 유지한다 (이미 각각 asos1_/asos2_/aws1_/aws2_/aws3_/fog1_/struct_
# 접두어가 붙어 있어 RWIS 백본 컬럼과 구분됨).
SELF_SENSOR_COLS = {
    "시정 (m)", "습도 (%)", "일 누적 강수량 (mm)", "적설 (cm)", "기온 (C)",
    "1분 풍향 (Degree)", "1분 풍속 (m/s)", "최대순간풍향 (Degree)", "최대순간풍속 (m/s)",
    "해면기압 (hPa)", "운고1", "운고2", "운고3", "운량",
    "기온_결측", "습도_결측",
}


def unobserved_feats(case_name):
    df = pd.read_parquet(f"{JIHU}/{case_name}_dataset.parquet")
    feats = select_features(df)
    feats2 = [c for c in feats if c not in SELF_SENSOR_COLS]
    removed = sorted(set(feats) - set(feats2))
    log(f"[{case_name}] 원래 {len(feats)}개 -> 자체 센서값 제외 후 {len(feats2)}개")
    log(f"  제외된 컬럼({len(removed)}개):", removed)
    return feats2


if __name__ == "__main__":
    # case4 only: case3 (organizer's raw, unprocessed data) is out of scope for submission.
    out = {}
    for case in ["case4"]:
        feats = unobserved_feats(case)
        out[case] = run_case(case, feature_subset=feats)
    with open(f"{JIHU}/eval_results_unobserved.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log("\nsaved eval_results_unobserved.json")
