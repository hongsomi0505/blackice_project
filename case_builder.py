"""Builds case4_dataset.parquet (the team's submission dataset) by merging 신지후's individually
processed feature files together with a structural (교량/터널/고도/ice_rate) summary file.

HOW TO RUN
----------
1. Put these files flat in DATA_DIR (default: the same folder as this script):
     RWIS_features.csv        (지후 가공 - RWIS 백본)
     ASOS_features.csv        (지후 가공 - ASOS 2슬롯)
     AWS_features.csv         (지후 가공 - AWS 3슬롯)
     목표안개_features.csv     (지후 가공 - 안개, 구글드라이브 원본 파일명 그대로)
     station_spatial_mapping.csv   (관측소 <-> AWS/ASOS/FOG 매핑표)
     final_station_analysis.csv    (관측소별 구조적 피처: 고도/교량·터널 거리/ice_rate 등)
2. Run:  python case_builder.py
   This writes case4_dataset.parquet into OUT_DIR (default: same folder as this script).
3. Then run train_eval_oofstack_case34_v2.py from the same folder.

case3(주최측 원본 그대로, 전처리 없음)은 제출 범위에서 제외되어 이 스크립트는 case4만 만든다.
"""
import os
import gc
import pandas as pd
import numpy as np

DROP_FEAT_COLS = {"지점명", "위도", "경도", "관측 지점명", "관측소 위도", "관측소 경도",
                   "고속도로", "관측소 종류", "관측소 고도 (m)",
                   "hour_sin", "hour_cos", "doy_sin", "doy_cos", "winter_season"}

# ====== EDIT THESE TWO IF YOUR DATA LIVES SOMEWHERE ELSE ======
# DATA_DIR: folder containing the 6 input files listed in the module docstring above.
# OUT_DIR : where case4_dataset.parquet is written (train_eval_oofstack_case34_v2.py reads
#           it from here too, via the JIHU env var / default in train_eval_v2.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CASE_BUILDER_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))
OUT_DIR = os.environ.get("JIHU_DATA_DIR", _HERE)
# ================================================================

RWIS_TIME = "관측시각 (KST)"
RWIS_ID = "관측 지점번호"
RWIS_NAME = "관측 지점명"


def log(*a):
    print(*a, flush=True)


def mem_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1
    return -1


def downcast(df, exclude):
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
    return df


def strip_cols(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def load_backbone():
    rwis = pd.read_csv(f"{DATA_DIR}/RWIS_features.csv", encoding="utf-8-sig")
    strip_cols(rwis)
    rwis[RWIS_TIME] = pd.to_datetime(rwis[RWIS_TIME])
    rwis[RWIS_NAME] = rwis[RWIS_NAME].str.strip()
    rwis[RWIS_ID] = pd.to_numeric(rwis[RWIS_ID], errors="coerce").astype("Int64")
    # missing indicators for RWIS's own 기온/습도 (known 100% gap at 삼봉/대곡)
    rwis["기온_결측"] = rwis["기온 (C)"].isna().astype(int)
    rwis["습도_결측"] = rwis["습도 (%)"].isna().astype(int)
    log("backbone rows:", len(rwis), "cols:", rwis.shape[1])
    return rwis


def load_mapping():
    m = pd.read_csv(f"{DATA_DIR}/station_spatial_mapping.csv", encoding="utf-8-sig")
    strip_cols(m)
    for c in ["RWIS_ID", "AWS_1_ID", "AWS_2_ID", "AWS_3_ID", "ASOS_1_ID", "ASOS_2_ID", "FOG_1_ID"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("Int64")
    return m


def check_unique_key(df, id_col, time_col, label):
    dup = df.duplicated(subset=[id_col, time_col]).sum()
    log(f"[{label}] duplicate (id,time) rows: {dup} / {len(df)}")
    if dup:
        raise SystemExit(f"{label}: duplicate keys found, aborting")


def join_slot(base, ext, ext_id_col, ext_time_col, slot_id_col, prefix, hourly=False):
    # Uses fixed internal key names (_ext_id/_ext_jt/_base_id/_base_jt) throughout so that
    # source files whose id/time columns happen to share RWIS's own column names (e.g. the
    # fog file also uses "관측 지점번호"/"관측시각 (KST)") can never silently collide with
    # and overwrite the backbone's own columns during merge.
    ext[ext_time_col] = pd.to_datetime(ext[ext_time_col])
    ext[ext_id_col] = pd.to_numeric(ext[ext_id_col], errors="coerce").astype("Int64")
    ext_jt = ext[ext_time_col].dt.floor("h") if hourly else ext[ext_time_col]
    base_jt = base[RWIS_TIME].dt.floor("h") if hourly else base[RWIS_TIME]

    feat_cols = [c for c in ext.columns if c not in (ext_id_col, ext_time_col) and c not in DROP_FEAT_COLS]
    ext_small = ext[feat_cols].copy()
    ext_small["_ext_id"] = ext[ext_id_col].values
    ext_small["_ext_jt"] = ext_jt.values
    downcast(ext_small, exclude={"_ext_id", "_ext_jt"})
    ext_small = ext_small.rename(columns={c: f"{prefix}_{c}" for c in feat_cols})

    check_unique_key(ext_small, "_ext_id", "_ext_jt", f"{prefix} source")
    log(f"[{prefix}] source mem after downcast: {ext_small.memory_usage(deep=True).sum()/1e6:.1f} MB, rss={mem_mb():.0f}MB")

    base["_base_id"] = base[slot_id_col].values
    base["_base_jt"] = base_jt.values

    before = len(base)
    merged = base.merge(
        ext_small, left_on=["_base_id", "_base_jt"], right_on=["_ext_id", "_ext_jt"], how="left"
    )
    del ext_small
    base.drop(columns=["_base_id", "_base_jt"], inplace=True)
    if len(merged) != before:
        raise SystemExit(f"{prefix}: row count changed after merge ({before} -> {len(merged)})")
    merged = merged.drop(columns=["_base_id", "_base_jt", "_ext_id", "_ext_jt"])
    cov = merged[f"{prefix}_{feat_cols[0]}"].notna().mean()
    log(f"[{prefix}] merged. coverage (non-null first feat col): {cov:.3f}, rows={len(merged)}, cols={merged.shape[1]}, rss={mem_mb():.0f}MB")
    assert RWIS_ID in merged.columns, f"{prefix}: {RWIS_ID} lost after merge!"
    del base
    gc.collect()
    return merged


def load_structural():
    s = pd.read_csv(f"{DATA_DIR}/final_station_analysis.csv", encoding="utf-8-sig")
    strip_cols(s)
    s[RWIS_ID] = pd.to_numeric(s[RWIS_ID], errors="coerce").astype("Int64")
    keep = [RWIS_ID, "고도", "nearest_bridge_m", "nearest_tunnel_m",
            "bridge_adjacent", "tunnel_adjacent", "ice_rate", "temp_mean", "night_temp_mean"]
    s = s[[c for c in keep if c in s.columns]].rename(
        columns={c: f"struct_{c}" for c in keep if c != RWIS_ID}
    )
    return s


def build_case4():
    log(f"\n===== building case4 (data from {DATA_DIR}) =====")
    rwis = load_backbone()
    mapping = load_mapping()
    rwis = rwis.merge(mapping, left_on=RWIS_ID, right_on="RWIS_ID", how="left", suffixes=("", "_map"))
    log("after mapping merge:", len(rwis))

    # ASOS (지후 가공, 2개 슬롯, 시간단위)
    asos = pd.read_csv(f"{DATA_DIR}/ASOS_features.csv", encoding="utf-8-sig")
    strip_cols(asos)
    for i, slot_col in enumerate(["ASOS_1_ID", "ASOS_2_ID"], start=1):
        rwis = join_slot(rwis, asos, "지점", "시간", slot_col, f"asos{i}", hourly=True)
    del asos

    # AWS (지후 가공, 3개 슬롯, 10분단위)
    aws = pd.read_csv(f"{DATA_DIR}/AWS_features.csv", encoding="utf-8-sig")
    strip_cols(aws)
    # The uploaded AWS CSV ends with a truncated record ("2024", no station ID).
    aws["시간(KST)"] = pd.to_datetime(aws["시간(KST)"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    valid_aws = aws["시간(KST)"].notna() & pd.to_numeric(aws["지점 번호"], errors="coerce").notna()
    log(f"[AWS] excluded invalid station/time keys: {(~valid_aws).sum()}")
    aws = aws.loc[valid_aws]
    for i, slot_col in enumerate(["AWS_1_ID", "AWS_2_ID", "AWS_3_ID"], start=1):
        rwis = join_slot(rwis, aws, "지점 번호", "시간(KST)", slot_col, f"aws{i}", hourly=False)
    del aws

    # FOG (지후 가공, 안개 피처) -- 구글드라이브 원본 파일명이 "목표안개_features.csv"라서
    # (Claude가 작업 중 내부적으로 fog_features.csv로 바꿔 부른 것일 뿐, 실제로는 같은 파일)
    # 그대로 원본 파일명으로 읽는다. 다른 이름으로 바꾸지 말고 Drive에서 받은 그대로 두면 됨.
    fog = pd.read_csv(f"{DATA_DIR}/목표안개_features.csv", encoding="utf-8-sig")
    strip_cols(fog)
    rwis = join_slot(rwis, fog, RWIS_ID, RWIS_TIME, "FOG_1_ID", "fog1", hourly=False)
    del fog

    # 구조적 피처 (고도/교량·터널 거리/과거 ice_rate) -- "제공 데이터 외 추가 데이터":
    # 대회 요강(제출 형식)에 따라 활용 근거(관련 논문·선행연구)를 제출물에 함께 제시할 것.
    struct = load_structural()
    rwis = rwis.merge(struct, on=RWIS_ID, how="left")
    log("after structural merge:", len(rwis), "cols:", rwis.shape[1])

    out_path = f"{OUT_DIR}/case4_dataset.parquet"
    os.makedirs(OUT_DIR, exist_ok=True)
    rwis.to_parquet(out_path, index=False)
    log(f"saved -> {out_path} ({rwis.shape[0]} rows, {rwis.shape[1]} cols)")
    return out_path


if __name__ == "__main__":
    build_case4()
