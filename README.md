# 은하 SFR 예측 — AutoML 총동원 실험

SNU DataHub 천문 데이터 과제(시뮬레이션 은하 208,437개의 별 생성률 `sfr_linear` 회귀)에서,
원래 최종 모델(무튜닝 LightGBM + Tweedie p = 1.3, 변수 25개, CV RMSE 7.401 / 검증 7.619)을
다양한 단일 모델, AutoGluon(전처리 7가지), 앙상블로 이길 수 있는지 확인한 기록이다.

- 진행 상황 페이지: https://yu-jusang.github.io/galaxy-sfr-automl/
- 모든 선택은 학습 80% 안의 같은 5-fold CV로 하고, 검증셋은 마지막에 한 번만 평가한다.
- 원본 CSV는 올리지 않는다. DataHub(dataset 42, CC BY 4.0)에서 받아 `data/`에 둔다.

| 경로 | 내용 |
|---|---|
| `src/sfr_lib.py` | 원 프로젝트 공용 코드(고정 분할, 층화 CV, 지표, 물리·이웃 변수) |
| `src/core.py` | 이번 실험 공용 코드(추가 환경 변수, 변수 세트, 결과 저장) |
| `scripts/a02_models.py` | 단일 모델 CV |
| `scripts/a03_autogluon.py` | AutoGluon 변형 CV (bagging fold = 프로젝트 CV fold) |
| `scripts/a04_ensemble.py` | 앙상블, 최종 선택, 검증셋 1회 평가 |
| `results/meta/` | 실험별 fold 지표(JSON), AutoGluon 리더보드 |
| `docs/` | GitHub Pages |
