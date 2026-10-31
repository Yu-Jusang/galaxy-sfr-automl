# 기기 4대에 실험 나눠 돌리기 (2026-10-31 개정)

모든 기기가 **같은 저장소 코드, 대회 양식의 고정 80/20 분할, 같은 11층 층화 5-fold(seed 42)** 를 쓴다.
각자 맡은 작업을 돌리고 결과 zip만 모으면 그대로 합쳐진다. 작업 목록과 배정은 `scripts/jobs.py` 한 곳에 있다.

| 기기 | 권장 담당 | 맡은 작업 | 예상 시간 |
|---|---|---|---|
| A | 조정용 노트북 (Windows, GPU 없음) | 단일 모델(F5C·F5LS), 선형·커널·KNN 30개(튜닝 포함), nested 변수 선택, AutoGluon 로그·제곱근 목표, 최종 앙상블 | 약 10.5 h |
| B | Apple Silicon Mac (MPS) | 딥러닝·커널·사전학습 18개(TabM, RealMLP×2, xRFM, TabPFN, Tweedie MLP × F5/F5C/F5LS), TabM·RealMLP 튜닝 | 약 11 h |
| C | CPU 기기, RAM 16 GB 이상 | LightGBM·XGBoost·CatBoost nested 튜닝(변수 묶음 포함 여부까지), AutoGluon 거대 은하 가중치 | 약 10.5 h |
| D | CUDA GPU (Colab Pro, 연구실 서버) | AutoGluon 변수 축(F5C, F5LS, F5X, FULL, T2CORE), Tweedie, extreme_quality | 약 11 h |
| X | 먼저 끝난 기기 | (선택) AutoGluon F0·F3, 4시간판 | 3–7 h |

## 0. 조원 2의 변수 모듈을 쓰려면 (FULL, T2CORE 세트)

조원 2의 `sfr_features.py`(424개 변수)를 저장소의 `src/` 폴더에 넣는다. 그러면 첫 실행 때 자동으로 `t2_` 접두사를 붙여 변수 파일에 추가한다.
모듈은 타깃과 ID가 빠진 13개 입력 열만 받는다. 파일이 없으면 FULL, T2CORE 관련 작업은 건너뛰고 나머지는 그대로 돈다.
**모든 기기가 같은 `sfr_features.py`를 써야 한다**(A 담당이 저장소에 올린다).

## 1. 준비 (한 번, 약 20분)

```bash
git clone https://github.com/Yu-Jusang/galaxy-sfr-automl.git
cd galaxy-sfr-automl
git log --oneline | head -1          # 모든 기기가 같은 커밋인지 확인
conda create -n sfr python=3.11 -y
conda activate sfr
pip install "autogluon.tabular[all,tabarena]==1.4.0" xrfm optuna
```

- 원본 CSV(`Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv`)를 `data/`에 넣는다. SHA256 `0710e7b2…6a54`.
- 첫 실행 때 변수 파일(`data/features_all.parquet`, 159열 + 조원 2 모듈)을 만드는 데 3–5분 걸린다.
- Mac: 필요하면 `brew install libomp`. 딥러닝은 Apple GPU(MPS)를 자동으로 쓴다.
- CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` 후 `python -c "import torch; print(torch.cuda.is_available())"`.
- **Colab:** 세션이 끊겨도 결과가 남도록 저장소를 Google Drive 안에 clone하고, 끊기면 같은 명령으로 다시 실행한다.

## 2. 실행 (명령 한 줄)

```bash
python scripts/jobs.py list      # 내 작업 확인
python scripts/jobs.py run B     # 자기 기기 문자
```

- 끝난 작업은 건너뛴다. 꺼지면 같은 명령으로 이어서 돌린다(진행 중이던 작업 하나만 처음부터).
- 스레드 수는 `SFR_N_THREADS`(기본 14). 한 기기에서 두 문자를 동시에 돌리지 않는다.
- 전원 연결, 덮개 열기, 절전 방지(Mac: `caffeinate -i python scripts/jobs.py run B`).

## 3. 결과 보내기

```bash
python scripts/jobs.py pack B    # results/pack_B.zip
```

zip을 A 담당에게 직접 보낸다. `oof` 파일에 학습 은하의 정답 SFR이 들어 있으므로 공개된 곳에 올리지 않는다.

## 4. 합치기 (A만)

```bash
python scripts/jobs.py unpack pack_B.zip pack_C.zip pack_D.zip
python scripts/a04_ensemble.py
```

## 지킬 것
- `src/`, `scripts/` 코드는 고치지 않는다(패키지 설치 문제만). 오류는 `results/logs/J??_<기기>.log`를 그대로 공유한다.
- **검증셋(공식 20%)은 어떤 선택에도 쓰지 않는다.** 최종 평가는 A가 한 번만 한다.
- 예전에 각자 쓰던 분할(예: 조원 1의 `split_indices.npz`)은 공식 분할과 다르므로 쓰지 않는다. 이 저장소 코드는 대회 양식 셀과 똑같은 분할을 만든다.
- 저장소에 push하지 않는다(A 담당만).
