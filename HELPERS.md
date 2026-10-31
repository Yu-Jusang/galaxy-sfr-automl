# 다른 컴퓨터에서 실험 나눠 돌리기

실험은 서로 독립이라 컴퓨터마다 다른 것을 돌리고 결과 파일만 모으면 된다.
모든 컴퓨터가 같은 분할·같은 5-fold를 쓰므로 결과를 그대로 합칠 수 있다.

## 1. 준비 (한 번, 약 20분)

```bash
git clone https://github.com/Yu-Jusang/galaxy-sfr-automl.git
cd galaxy-sfr-automl
conda create -n sfr python=3.11 -y
conda activate sfr
pip install "autogluon.tabular[all,tabarena]==1.4.0" xrfm optuna
```

- 원본 CSV(`Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv`)를 `data/` 폴더에 넣는다.
  CSV는 저장소에 없다. 팀원에게 직접 받는다.
- NVIDIA GPU가 있으면 PyTorch를 CUDA판으로 설치하면 딥러닝 모델이 자동으로 GPU를 쓴다:
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`

## 2. 맡은 실험 돌리기

맡은 줄만 실행한다. 처음 실행 때 변수 파일(`data/features_all.parquet`)을 만드는 데 1–2분 걸린다.

| 담당 | 명령 | CPU 기준 시간 |
|---|---|---|
| A | `python scripts/a03_autogluon.py AG2_F3 AG5_ALL AG1_F0 AG7_F5_log --force` | 약 6시간 |
| B | `python scripts/a03_autogluon.py AG4_F5A --force` | 약 1.5시간 |
| B | `python scripts/a06_deep.py` | GPU 1–2시간 / CPU 4–6시간 |

스레드 수는 `SFR_N_THREADS` 환경변수로 바꿀 수 있다(기본 14).
컴퓨터가 절전으로 들어가지 않게 전원을 연결하고 덮개를 열어 둔다.

## 3. 결과 보내기

끝나면 아래 세 폴더에서 맡은 실험 이름으로 시작하는 파일을 모두 압축해 보낸다.

```
results/oof/<실험이름>.parquet
results/val/<실험이름>.npy
results/meta/<실험이름>.json   (AutoGluon은 <실험이름>_leaderboard.csv 도)
```

예: `zip -r helper_A.zip results/oof/AG2_* results/val/AG2_* results/meta/AG2_* ...`

`oof` 파일에는 학습 은하의 정답 SFR(`y` 열)이 들어 있으므로 공개된 곳에 올리지 말고 직접 전달한다.
