# Case 4 CatBoost 스태킹 Optuna 튜닝

## Windows VS Code에서 실행

이 ZIP과 원래의 `data.zip`을 같은 상위 폴더에 풀고 VS Code에서 상위 폴더를 엽니다. 압축을 푼 뒤 `blackice_optuna`와 `data` 폴더가 같은 위치에 있어야 합니다. Windows 탐색기의 압축 풀기 기능이 동명의 폴더를 한 겹 더 만들었으면 안쪽 폴더를 옮깁니다.

| 상위 폴더에 둘 폴더 | 내부 파일 예시 |
| --- | --- |
| `blackice_optuna` | `case_builder.py`, `tune_optuna.py`, `requirements.txt` |
| `data` | `RWIS_features.csv`, `AWS_features.csv`, `ASOS_features.csv`, `목표안개_features.csv`, `station_spatial_mapping.csv`, `final_station_analysis.csv` |

VS Code의 PowerShell 터미널에서 실행합니다. Python 3.12를 사용하며 가상환경의 Python을 경로로 직접 실행합니다.

```powershell
cd .\blackice_optuna
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe case_builder.py
.\.venv\Scripts\python.exe tune_optuna.py --trials 2 --threads 2 --out optuna_check
```

`case_builder.py`가 생성한 `case4_dataset.parquet`는 다음 실행에 다시 사용합니다. `--trials 2`는 기존 설정 1회와 새 조합 1회를 평가합니다. 최종 탐색을 시작하려면 아래 명령을 실행합니다. 명령을 실행할 때마다 새 탐색이 시작됩니다.

```powershell
.\.venv\Scripts\python.exe tune_optuna.py --trials 20 --threads 4
```

1 trial에 CatBoost를 14회 학습하므로 전체 데이터로 실행하면 시간이 걸립니다. `--threads`는 CatBoost의 학습 스레드 수입니다. 데이터나 결과 경로를 따로 쓰려면 `tune_optuna.py --help`를 실행합니다. 데이터 폴더가 다른 위치에 있다면 빌드 전에 PowerShell에서 `$env:CASE_BUILDER_DATA_DIR = 'C:\경로\data'`로 지정할 수 있습니다.

## 튜닝과 평가 방식

- 1단계는 기존 Case 4의 기본 변수, 2단계는 1단계 확률과 RWIS·ASOS 추가 변수 및 관측 가능 개수를 사용합니다.
- 두 단계의 `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`를 각각 탐색합니다. 기존 값은 두 단계 모두 `450`, `5`, `0.05`, `8`이며 첫 trial에서 비교 기준으로 평가합니다.
- 학습 관측소 14곳을 관측소 기준으로 4개 폴드로 나눕니다. 2단계 학습용 1단계 확률도 각 검증 관측소를 제외한 내부 교차검증으로 만듭니다. 기존 파라미터와 새 조합을 같은 교차검증 방식으로 비교합니다.
- 관측소·날짜별 10분 예측 확률의 최댓값에 임계값을 적용합니다. 두 겨울 기간의 일별 F1을 같은 비중으로 평균한 값을 최대화합니다. 임계값은 원본 스크립트의 후보 목록에서 trial마다 고릅니다.
- 부곡교, 삼봉, 해전은 최종 성능 평가에 사용합니다. 이 세 곳의 라벨은 파라미터나 임계값 선택에 사용하지 않습니다.
- 제공된 AWS CSV의 마지막 행은 시각과 지점번호가 불완전해 제외합니다. 매핑에는 있지만 첨부 CSV에 없는 AWS 관측소의 입력은 기존 병합 방식대로 결측값으로 남습니다.

## 결과 확인

| 출력 파일 | 내용 |
| --- | --- |
| `optuna_summary.json` | 기존 설정과 최적 설정의 OOF F1, 단계별 최적값, 임계값, 최종 세 곳의 F1 |
| `optuna_trials.csv` | trial별 파라미터, OOF F1, 임계값 |
| `holdout_predictions.csv` | 세 관측소의 10분별 실제 라벨, 예측 확률, 이진 예측 |
| `stage1.cbm`, `stage2.cbm` | 최적 설정으로 14개 관측소를 사용해 재학습한 두 모델 |

첨부 데이터 전체로 Parquet 생성을 확인했고, 실제 데이터의 일부와 축소한 트리 수를 사용해 2개 trial, 홀드아웃 평가, 결과 저장까지 확인했습니다. 전체 데이터로 최적 파라미터를 탐색한 성능 결과는 포함하지 않습니다.
