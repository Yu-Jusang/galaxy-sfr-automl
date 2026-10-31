# 기기 4대에 실험 나눠 돌리기

모든 기기가 같은 코드, 같은 분할, 같은 5-fold를 쓴다. 각자 맡은 작업을 돌리고 결과 zip만 모으면 그대로 합쳐진다.
작업 목록과 배정은 `scripts/jobs.py` 한 곳에 있다(`python scripts/jobs.py list`).

| 기기 | 역할 | 맡은 작업 | 예상 시간 |
|---|---|---|---|
| A | 조정용 노트북(Windows, GPU 없음) | 단일 모델 확장, 선형·커널·KNN 20개(튜닝 포함), AutoGluon 변수 축(F0·F3·F5L·F5S), 최종 앙상블·스태킹 | 약 10.5 h |
| B | Apple Silicon Mac (MPS) | 딥러닝·커널·사전학습 12개(TabM, RealMLP×2, xRFM, TabPFN, Tweedie MLP × F5/F5LS), TabM·RealMLP 튜닝 | 약 11 h |
| C | CPU, RAM 16 GB 이상 | nested Optuna 튜닝: LightGBM 50회, XGBoost 40회, CatBoost 25회 (변수 묶음 포함 여부도 튜닝), AutoGluon 제곱근 목표·스택 2층 | 약 12 h |
| D | CUDA GPU 있으면 최고(Colab Pro, 연구실 서버) | AutoGluon 전처리 축(F5LS, 전체 변수, Tweedie, 로그 목표, 거대 은하 가중치), extreme_quality, (선택) 4시간판 | 약 9.5 h (+4 h) |

## 1. 준비 (한 번, 약 20분)

```bash
git clone https://github.com/Yu-Jusang/galaxy-sfr-automl.git
cd galaxy-sfr-automl
conda create -n sfr python=3.11 -y
conda activate sfr
pip install "autogluon.tabular[all,tabarena]==1.4.0" xrfm optuna
```

- 원본 CSV(`Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv`)를 `data/`에 넣는다. 저장소에는 없다.
- 첫 실행 때 변수 파일(`data/features_all.parquet`, 159열)을 만드는 데 3–5분 걸린다. 변수는 코드가 CSV에서 자동으로 계산하므로 모든 기기에서 같다.
- Mac은 LightGBM용으로 `brew install libomp`가 필요할 수 있다. 딥러닝 모델은 Apple GPU(MPS)를 자동으로 쓴다.
- CUDA GPU 기기는 PyTorch CUDA판을 깐다: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

## 2. 실행 (명령 한 줄)

```bash
python scripts/jobs.py run B     # 자기 기기 문자(A/B/C/D)
```

- 끝난 작업은 건너뛰므로, 중간에 꺼져도 같은 명령으로 이어서 돌린다(진행 중이던 작업 하나만 처음부터).
- 스레드 수는 `SFR_N_THREADS` 환경변수로 바꾼다(기본 14).
- 컴퓨터가 잠들지 않게 전원을 연결하고 덮개를 열어 둔다(Mac은 `caffeinate -i python scripts/jobs.py run B`).

## 3. 결과 보내기

```bash
python scripts/jobs.py pack B    # results/pack_B.zip 생성
```

`results/pack_<기기>.zip`을 A 담당에게 직접 보낸다. 이 zip의 `oof` 파일에는 학습 은하의 정답 SFR이 들어 있으므로 공개된 곳(GitHub 등)에 올리지 않는다.

## 4. 합치기 (A만)

```bash
python scripts/jobs.py unpack pack_B.zip pack_C.zip pack_D.zip
python scripts/a04_ensemble.py   # 앙상블, CV 기준 최종 선택, 검증셋 1회 평가
```

## 지킬 것
- `src/`, `scripts/` 코드는 고치지 않는다(패키지 설치 문제만 고친다). 오류가 나면 로그(`results/logs/J??_<기기>.log`)를 그대로 공유한다.
- 검증셋 점수를 보고 설정을 바꾸지 않는다. 선택은 CV로만 한다.
- 저장소에 push하지 않는다(A 담당만 push).
