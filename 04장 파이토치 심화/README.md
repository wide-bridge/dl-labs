### Ch04_1 정칙화 등.ipynb 

#### 🔎 정칙화 및 최적화 기법 실험 최종 정리

1️⃣ Loss와 정칙화의 역할 구분 
--
구분	   data loss	  regularization (L1, L2 등)
목적	  예측 정확도	  모델 복잡도 제어
학습   	  ✔ 포함	        ✔ 포함
검증/평가	✔ 포함	       ❌ 제외

- 학습 시 loss: data_loss + regularization
- 검증/평가 시 loss: data_loss only
- 👉 정칙화는 train/val loss가 아니라, test data loss로 성공 여부를 판단해야 함.

2️⃣ 실험 결과 요약

(동일 기준: test data_loss only)
2️⃣ 실험 결과 요약 (같은 기준: test data_loss)
정칙화 전: Final TEST loss (data_loss only): 0.096                                   baseline
L1 정칙화 후: Final TEST loss (data_loss only, with L1 regularization): 0.270707 -> ❌ 크게 악화
L2 정칙화 후: Final TEST loss (data_loss only, with L2 regularization): 0.199244 -> ❌ 악화
엘라스틱넷: Final TEST loss (data_loss only, with Elastic Net (λ1=0.001, λ2=0.001)): 0.215546 ❌ 악화
# 단 🔹 Elastic Net 계수
lambda_l1 = 1e-3
lambda_l2 = 1e-3 수치를 잘 조작하면 좋아짐. 
Final TEST loss (data_loss only, with Elastic Net (λ1=0.001, λ2=0.001)): 0.094662 ✔ 소폭 개선
가중치 감쇠: Final TEST loss (data_loss only, with weight decay=0.01): 0.095343 ->✔ 소폭 개선
모멘텀 적용: Final TEST loss (data_loss only, with SGD momentum=0.9): 0.092824 -> ✔ 가장 개선
그래드언트클리핑: Final TEST loss (data_loss only, with gradient clipping (max_norm=0.1)): 11839.280273 -> ❌ 학습 붕괴(가장 심각)
드롭아웃: Final TEST loss (data_loss only, with Dropout p=0.5): 3136.483154 ❌ 심각한 성능 붕괴

3️⃣ ❗ 소결 (핵심 판단)

이 데이터와 모델, 그리고 현재 설정에서는 L1·L2 정칙화 등은 일반화 성능을 개선하지 못했다.

이는 정칙화가 필요하지 않은 상황이었거나 정칙화 강도(λ)가 과도했기 때문이다.

4️⃣ 왜 이런 상황이 발생했는가? (왜 정칙화가 비효과적이었는가?)
(1) λ가 과도하게 큼
_lambda = 0.5 는 MSE 스케일 대비 매우 큰 값

L1
비미분점 + 강한 패널티
작은 모델에서 표현력 급격히 훼손

L2
과도한 shrinkage(수축) 발생
👉 정칙화 개념의 실패가 아니라, 하이퍼파라미터 튜닝 실패

(2) 데이터 자체가 이미 과적합 상태가 아님

정칙화 없이도 train / val / test loss 모두 안정

test loss ≈ 0.096
고칠 과적합이 존재하지 않음
👉 문제가 없는데 제약을 건 상황

(3) 이 문제에서는 feature를 줄이는 것이 오히려 손해

입력 feature: [x², x] (2개뿐)
L1의 본질: sparsity / feature selection

하지만: 두 feature 모두 의미 있음

하나라도 약화 → 표현력 손실 → 성능 저하
👉 L1의 장점이 단점으로 작동한 전형적인 사례

(4) Gradient Clipping & Dropout이 망가진 이유 (보완 설명)

🔹 Gradient Clipping
원래 목적: gradient 폭발 방지

하지만: 이 문제는 폭발 문제가 없음

max_norm=0.1 → 학습 신호 자체 제거 
👉 underfitting → 학습 붕괴
그래디언트 클리핑이란 gradient의 크기(norm)가 기준을 넘으면, 그 크기를 기준값까지 “줄여서(clipping해서)” 쓰라는 뜻 
결과적으로 잘 작동하던 그래디언트를 과도하게 제약해서 학습이 붕괴됨.

🔹 Dropout
회귀 문제 + 데이터 적음 + 출력 1차원
Dropout(p=0.5):
출력 분산을 인위적으로 흔듦

MSE 기반 회귀에서 치명적
👉 노이즈 주입 → 예측 자체 붕괴

5️⃣ 회고

정칙화는 항상 성능을 높여주는 마법이 아니라,
과적합이 존재하고, 적절한 강도로 적용될 때만 효과가 있다.
이번 실험에서는 그 조건이 성립하지 않았다.

L1 / L2 / Dropout / Gradient Clipping → 과도하거나 부적절

Weight Decay, Momentum → 약한 제어로 소폭 개선

가장 중요한 것은 문제·데이터·모델에 맞는 기법 선택

이 실험의 진짜 성과는 정칙화를 언제 쓰지 말아야 하는지를 확인했음.

### Ch04_2 데이터 증강 및 LLM기반 패러 플레이즈징

- LLM기반 패러플레이징이 가장 간편한 방법 확인

### Ch04_3 이미지 증강

#### 파이토치의 텐서를 다른 라이브러리로 전환할 경우 이용하는 방법 

- 텐서를 쓰지 않을 경우 - transforms.Compose ([...,transforms.ToPILImage()]) 에 추가
- 텐서를 쓰고 나서 .permute(1,2,0)으로 순서를 바꿈
