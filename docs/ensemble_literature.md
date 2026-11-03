# 앙상블 단계의 선행연구 근거

최종 앙상블(`scripts/a04_ensemble.py`)의 각 설계를 어떤 선행연구에 근거했는지 정리한다. 인용은 모두 서지 정보를 확인한 것이다.

## 1. 기존 설계의 근거

- **CV 밖에서 만든 예측으로만 메타모델을 학습한다.** Wolpert(1992, *Neural Networks* 5:241)의 stacked generalization 원칙이다. 우리 앙상블은 base 모델의 OOF 예측만 쓰고, 가중치도 바깥 fold 4개로 정해 나머지 1개에서 평가한다.
- **0 이상, 합 1인 가중치(`ENS_simplex`).** Breiman(1996, *Machine Learning* 24:49)은 stacking에서 비음수 제약이 있어야 안정적으로 이득이 난다고 보였다. Super Learner(van der Laan, Polley, Hubbard 2007, *SAGMB* 6(1))는 CV로 정한 볼록결합이 점근적으로 최고 후보만큼은 좋다는 oracle 부등식을 준다.
- **Greedy 선택(`ENS_greedy`).** Caruana et al.(2004, ICML)의 ensemble selection이다. Auto-sklearn(Feurer et al. 2015, NeurIPS)과 AutoGluon(Erickson et al. 2020, arXiv:2003.06505)의 마지막 층이 이 방식이다.

## 2. 이번에 추가한 것

- **Bagged greedy와 sorted initialization(`ENS_greedy_bag`).** Caruana, Munson, Niculescu-Mizil(2006, ICDM)은 모델 라이브러리가 크고 서로 비슷할수록 greedy 선택이 과적합한다고 보고, 두 가지 처방을 냈다. 하나는 상위 몇 개 모델에서 출발하는 것이고, 다른 하나는 라이브러리의 무작위 절반에서 여러 번 선택해 가중치를 평균하는 것이다. 우리 라이브러리는 약 110개이고 비슷한 AutoGluon 실행이 여럿 있어서 이 문제에 해당한다. 설정은 bag 20개, 각 bag에 모델 50%, 상위 3개에서 출발, 40단계다.
- **거의 같은 모델 제거(`*_pruned`).** CV 순서대로 내려가면서, 이미 남긴 모델과 OOF 상관이 0.99 이상인 모델은 뺀다. 근거는 Caruana et al.(2006)과 TabRepo(Salinas & Erickson 2024, AutoML Conf)로, 라이브러리의 크기보다 다양성이 중요하다는 결과다. 이 제거는 바깥 fold 4개의 행만 보고 하므로 평가 fold는 관여하지 않는다.
- **모델 계열별 가중치 보고(`family_weights`).** TabArena(Erickson et al. 2025, NeurIPS D&B)는 계열이 다른 모델을 섞은 앙상블이 성능을 끌어올리지만, 신경망은 검증셋 과적합 때문에 가중치가 부풀려질 수 있다고 경고한다. 그래서 각 앙상블에서 tree·AutoGluon·neural·TabPFN·linear/kernel이 차지하는 비중을 기록한다.
- **"CV 1위 고르기"의 낙관 편향 점검(`selection_check`).** 후보가 100개가 넘으면 CV 최솟값은 낮은 쪽으로 편향된다(Cawley & Talbot 2010, *JMLR* 11:2079; Varma & Simon 2006, *BMC Bioinformatics* 7:91).
  - 선택 절차 자체를 중첩 CV로 평가한다. 바깥 fold 4개에서 1위를 골라 남은 fold에서 잰다.
  - Tibshirani & Tibshirani(2009, *Ann. Appl. Stat.* 3:822)의 편향 보정값도 함께 낸다.
  - CV 1위와의 차이가 표준오차보다 작은 후보는 사실상 동률로 표시한다.
  - 최종 선택 규칙(CV 최저)은 바꾸지 않는다. 이 값들은 보고서에서 성능 추정이 얼마나 낙관적인지 밝히는 데 쓴다.

## 3. 도메인 선행연구와의 관계

헤일로 정보로 은하의 SFR을 예측한 연구들은 대부분 트리 앙상블을 쓰고 log 목표로 학습했다. SFR이 0인 은하는 빼거나, 하한값을 주거나, 인공적인 값으로 바꿨다.

- Kamdar, Turk, Brunner 2016, *MNRAS* 457:1162
- Agarwal, Davé, Bassett 2018, *MNRAS*, arXiv:1712.03255
- Jo & Kim 2019, *MNRAS* 489:3565
- Lovell et al. 2022, *MNRAS* 509:5046
- Chittenden & Tojeiro 2023, *MNRAS* 518:5670

Lovell et al.(2022)은 이렇게 했을 때 SFR이 0인 은하가 작은 양수로 퍼지고, 별을 만드는 은하는 낮게 예측된다고 보고했다. 우리 평가는 선형 RMSE라서 Tweedie 목적함수를 주력으로 썼다.

계열이 다른 모델을 stacking한 도메인 연구는 de Santi et al.(2022, *MNRAS*, arXiv:2201.06054) 하나를 찾았다. ERT, kNN, LightGBM, NN을 선형회귀로 쌓았고, 쌓은 모델이 가장 좋았다.

## 4. 적용하지 않은 것

- **CMA-ES 같은 제약 없는 가중치 최적화.** Purucker & Beel(2023, AutoML Conf)에서 검증 지표에 과적합하는 경우가 있었다. 그래서 비음수 또는 볼록 가중치만 쓴다.
- **2단계 메타모델.** Jo & Kim(2019)처럼 quenched 확률 같은 보조 예측을 메타 입력으로 넣는 방법이다. 지금은 log 별질량만 메타 입력으로 쓰는 `ENS_stack_lgbm`으로 대신한다.

## 5. 최종 선택 규칙 변경 (2026-11-03, test 평가 전)

A·B·C·D 결과를 모두 모은 뒤, test를 열기 전에 최종 선택 규칙을 "CV 최저"에서 1-SE 규칙으로 바꿨다. 이 변경은 아래 실행보다 먼저 커밋했다.

- **규칙:** CV 1위 후보가 앙상블이면, 그 앙상블과의 fold별 paired 차이가 1 SE(sd/√5) 안에 드는 비앙상블 후보 중 CV가 가장 낮은 것을 최종 모델로 한다. 그런 후보가 없으면 앙상블을 그대로 채택한다. CV 1위가 비앙상블이면 그대로 둔다.
- **근거 1.** 후보 100여 개 중 1위를 고르는 단계 자체가 과적합한다(Cawley & Talbot 2010).
- **근거 2.** 이전 프로젝트(Amendment 1)에서는 같은 fold로 구성을 고른 평균 앙상블이 중첩 CV에서 이득을 잃었다.
- **근거 3.** B 없이 미리 돌린 결과에서, 상위 3개 평균의 이득은 AG26 대비 0.005였고 이긴 fold는 2/5였다. 선택 단계까지 중첩하면 7.338이었다.
- **적용 범위:** 비앙상블 후보 사이에서는 더 단순한 모델로 내려가지 않는다. 즉 AutoGluon을 단일 LightGBM으로 바꾸지 않는다. 이 규칙은 앙상블 층을 하나 더 얹을지만 결정한다.
