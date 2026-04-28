# Simulation AI

## Simulation Model

Simulation_LSTM.py는

```none
            ----------
            | input  |
            ----------
                | (input size)
                v
            ----------
            |  lstm  |
            ----------
                | (hidden size) (usually 128)
   (more lstm layer available)
                v
            ----------
            | linear |
            ----------
                | (output size)
                v
            ----------
            | output |
            ----------
```

다음과 같은 구조를 가진다.

## Dataset

position difference(3), attitude(4, quaternion), velocity(3), angular velocity(3), motor value(3, normalized) 총 17개의 벡터로 구성
sequence length 만큼의 이전 데이터를 input으로 다음 상태의 13개 벡터를 추론한다.(motor value 제외)

## Train

```python
outputs = model(inputs) # 추론
loss = criterion(outputs, targets) # loss 계산
loss.backward() # 역전파
optimizer.step() # 기울기 계산
```

## Log

position 없는 학습 : SimulLearn_LSTM_00-31-37.pth
position 있는 학습 : SimulLearn_LSTM_2026-02-27_21:19:20.pth
position 있고 lstm 2 layer 학습 : SimulLearn_LSTM_2026-03-03_13:25:20.pth

전부 정규화하고 position에 가중치 부여 : SimulLearn_LSTM_2026-03-17_15-45-43.pth
전부 정규화하고 가중치 균일 : SimulLearn_LSTM_2026-03-23_11-55-54.pth
--> 정규화 코드 적용 안됨

전부 정규화함 : SimulLearn_LSTM_Normalized_2026-03-24_14-55-16.pth -> 학습 오류
SimulLearn_LSTM_Normalized_2026-03-26_13-33-10.pth -> 학습 최적 ealry stopping

SimulLearn_Transformer_Normalized_2026-03-26_16-03-14.pth -> transformer 도입

SimulLearn_Transformer_Normalized_2026-03-31_11-09-42.pth -> 모델 크기 증가 및 early stopping 완화, learning rate 조절

SimulLearn_Transformer_Normalized_2026-04-01_15-29-09.pth -> mean pooling 대신 first token pooling

SimulLearn_Transformer_Normalized_2026-04-02_11-30-43.pth -> itransformer

SimulLearn_Transformer_Normalized_2026-04-23_15-58-58.pth -> iTransformer + long epoch

## Test

같은 조건의 데이터셋이되, 비행 수가 적은 데이터를 활용하여 가공한 데이터가 small_log_diffed이다.

model_tester_transform.py를 사용하면 small_log_diffed를 이용하여 현재 모델의 test를 진행할 수 있다.