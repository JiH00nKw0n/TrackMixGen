# MimicGen 섞기 방법 실험 안내 (2026-11-13)

이 문서는 MimicGen 7개 과제 전체를 우리 섞기 방법으로 다시 생성하고, 기준선과 같은 방식으로 학습하고 평가하는 방법을 적는다.
섞기 방법을 쓰는 실행은 메서드 설정 `mixup_only`로 돈다. 이 메서드는 서브태스크마다 원본 시연 각각의 성공 확률을 현재 장면에서
예측하고, 예측이 평균 이상인 원본만 그 확률에 비례해 섞는다. 섞기 전에 원본마다 대칭으로 같은 자세 가운데 선택된 원본에 가장
가까운 자세를 고른다(1절 4단계, 2026-11-14 추가). 기준선과 다른 것은 생성할 때의 이 섞기 하나뿐이다. 장면 범위,
원본 선택 규칙, 데이터 수, 정책, 학습, 평가는 기준선과 같다.

## 차례

- [1. 섞기 방법이 하는 일](#1-섞기-방법이-하는-일): 성공 확률 식과 섞는 규칙을 적는다.
- [1.1 대칭 맞추기를 넣은 이유와 영향받는 과제](#11-대칭-맞추기를-넣은-이유와-영향받는-과제): 2026-11-14에 추가한 단계를 왜 넣었고
  어느 과제가 바뀌는지 적는다.
- [2. 미리 만들어 둔 파일](#2-미리-만들어-둔-파일): 과제마다 커밋해 둔 계수 파일과 순위 파일, 그리고 이 파일들을 만든 생성 기록을 적는다.
- [3. 장면 범위](#3-장면-범위): 2026-11-04에 좁힌 범위가 계수 파일과 이번 실행에 모두 들어 있다는 것을 적는다.
- [4. 돌리는 실험](#4-돌리는-실험): 과제, 정책, 시드, 데이터 수와 평가 방식을 표로 적는다.
- [5. 실행 순서](#5-실행-순서): 코드 받기, 파일 확인, 실행 명령을 순서대로 적는다.
- [6. 결과 위치와 보고](#6-결과-위치와-보고): 결과 파일 위치와 기준선과 나란히 적는 방식을 적는다.
- [7. 주의할 점](#7-주의할-점): 기준선 결과를 옮기면 안 되는 이유와 계수 파일이 없을 때 일어나는 일을 적는다.

## 1. 섞기 방법이 하는 일

생성 시도 하나가 서브태스크 k에 도달했을 때, 원본 시연 j를 현재 장면에 맞춰 쓰면 서브태스크 k가 성공할 확률을 선형 로지스틱
회귀로 예측한다. 로지스틱 회귀는 입력의 가중합을 시그모이드 함수 σ(u) = 1 / (1 + e^(−u))에 넣어 0과 1 사이의 확률을 내는
모형이다.

```
P(서브태스크 k 성공 | 원본 j) = σ( α_kj + a_k·|Δx| + b_k·|Δy| + c_k·Δr + [k ≥ 2]·( d_k·Δh_위치 + e_k·Δh_회전 ) )
```

- `α_kj`는 원본 j가 서브태스크 k에서 가지는 고유한 성공 경향이다. 값이 클수록 장면과 상관없이 잘 성공하는 원본이다.
- `|Δx|`와 `|Δy|`는 서브태스크 k의 기준 물체가 지금 장면과 원본 j에서 테이블 x축과 y축 방향으로 떨어진 거리(m)다. 기준 물체는
  그 서브태스크에서 손이 다가가는 물체다. 예를 들어 Threading의 첫 서브태스크에서는 바늘이다.
- `Δr`은 지금 장면과 원본 j 사이에서 기준 물체 방향이 회전한 각도(rad)다.
- `Δh_위치`와 `Δh_회전`은 두 번째 서브태스크부터 들어가는 손안 자세 차이다. 손안 자세는 바로 앞 서브태스크의 기준 물체가 손(집게)
  좌표계에서 놓인 위치와 방향이다. 예를 들어 Threading의 두 번째 서브태스크에서는 집게에 잡힌 바늘의 위치와 방향이다. 두 항은
  지금 시도와 원본 j 사이에서 이 자세의 위치 거리(m)와 회전 각도(rad)다.
- 서브태스크 k의 계수는 원본이 J개일 때 첫 서브태스크에서 J + 3개, 두 번째부터 J + 5개다. 별도의 절편은 두지 않는다. α가
  절편 역할을 함께 하기 때문이다.

섞는 규칙은 서브태스크가 시작될 때마다 아래 순서로 적용한다.

1. 원본 J개 모두의 예측 확률 p_j를 위 식으로 계산한다.
2. 예측 확률이 J개의 평균 이상인 원본만 남긴다. 집게를 열고 닫는 순서가 예측이 가장 높은 원본과 다른 원본은 시간을 맞춰 섞을
   수 없으므로 뺀다. 예측이 가장 높은 원본은 항상 남는다.
3. 남은 원본의 가중치를 p_j / (남은 원본의 p 합)으로 정한다.
4. 대칭 맞추기. 같은 잡기를 서로 다르게 적은 원본이 있을 수 있다. 평행 집게는 다가가는 축(손 좌표계 z축)으로 180도 돌려도 같은
   집게다. 물체에 따라서는 물체를 돌려도 같은 물체다. 예를 들어 Hammer Cleanup의 망치는 손잡이 축으로 180도 돌려도 같다. 그래서
   MimicGen이 고른 원본(선택된 원본)을 뺀 남은 원본마다, 집게 180도 회전과 기준 물체의 대칭 회전을 조합한 후보 가운데 선택된
   원본에 가장 가까운 것을 골라 그 원본 구간 전체에 적용한다. 가까움은 기준 물체 좌표계에서 두 구간의 기준 시점 자세로 잰다.
   기준 시점은 구간에서 집게가 처음 열리거나 닫히는 단계이고, 그런 단계가 없으면 구간의 마지막 단계다. 점수는 CP-Gen이 대칭을 고를
   때 쓰는 식과 같은 위치 거리(m) × 5 + 회전 각도(rad)다. 물체 대칭은 과제 설정 `configs/experiment/generator/mimicgen/task/<과제>.yaml`의
   `mixing_symmetries`에 적고, 메서드 설정의 `mixing.symmetries`가 이 값을 읽는다. 집게 180도 회전은 설정 없이 늘 후보에 들어간다.
   원본마다 고른 후보와 맞춘 뒤의 거리는 기록에 남는다(6절).
5. 남은 원본의 구간을 각자의 기준 물체 좌표계에서 이 가중치로 섞는다. 집게가 열리고 닫히는 시점을 맞춘 뒤 위치는 가중 평균하고
   회전은 가중 회전 평균을 낸다. 그 다음 MimicGen이 원래 하던 대로 현재 장면으로 옮겨 실행한다.

코드는 `tmg/generation/method/logistic_mix.py`(섞기와 대칭 맞추기), `tmg/generation/method/success_model.py`(확률 식과 계수 파일),
`tmg/generation/fit_coefficients.py`(계수 맞추기)에 있다.

### 1.1 대칭 맞추기를 넣은 이유와 영향받는 과제

CP-Gen에 같은 섞기를 붙여 Hammer Cleanup을 20번 생성해 보았더니(2026-11-13) 20번 모두 실패했다. 섞기 없이 원본을 균등하게 고른
기준선의 생성 성공률은 40.9 %다. 실패 20번 가운데 16번은 망치를 잡으러 가는 동작 계획에서 실패했다. 원인은 원본들이 망치를
잡는 방식이 망치 좌표계에서 두 무리로 나뉘어 있다는 것이었다. 두 무리는 망치 손잡이 축으로 180도 돌린 관계였다. 두 무리를 그대로
섞으면 위치는 망치 가운데쯤, 방향은 양쪽 무리 모두와 약 180도 다른 자세가 나왔고, 로봇이 그 자세에 닿을 수 없었다. 대칭 맞추기를
넣고 같은 기록을 다시 계산하면 섞은 잡기 자세가 선택된 원본의 자세와 평균 1.7 cm, 3.0도 차이로 줄었다(이전 평균 5.8 cm,
113.6도).

MimicGen 원본 시연 10개에서 서브태스크마다 모든 (선택된 원본, 다른 원본) 쌍 90개에 4단계를 적용한 결과는 아래와 같다. 회전은
기준 시점에서 선택된 원본과의 회전 각도 평균이다. 표에 없는 서브태스크와 과제(Coffee, Threading, Three Piece Assembly, 그리고
Hammer Cleanup과 Mug Cleanup의 서랍 서브태스크)는 모든 쌍에서 대칭을 쓰지 않는 쪽이 골라져 섞기 결과가 이전과 같다.

| 과제 | 서브태스크(기준 물체) | 대칭을 고른 쌍 | 맞추기 전 회전 평균 | 맞춘 뒤 회전 평균 | 쓴 대칭 |
|---|---|---|---|---|---|
| Hammer Cleanup | 망치 잡기(`hammer`) | 48 / 90 | 96.4도 | 3.3도 | 망치 손잡이 축 180도 |
| Square | 너트 잡기(`square_nut`) | 50 / 90 | 101.1도 | 9.2도 | 집게 180도 |
| Square | 막대에 끼우기(`square_peg`) | 70 / 90 | 85.2도 | 14.4도 | 막대 z축 90, 180, 270도와 집게 180도 |
| Stack Three | 큐브 네 서브태스크(`cubeA`, `cubeB`, `cubeC`) | 66~76 / 90 | 87.1~99.8도 | 9.0~23.5도 | 큐브 z축 90, 180, 270도와 집게 180도 |
| Mug Cleanup | 머그 잡기(`object`) | 42 / 90 | 82.6도 | 47.6도 | 집게 180도 |

과제 설정에 물체 대칭을 적은 것은 Hammer Cleanup의 `hammer`(z축 180도, CP-Gen 주석과 같은 대칭), Square의 `square_peg`(z축 90,
180, 270도), Stack Three의 `cubeA`, `cubeB`, `cubeC`(z축 90, 180, 270도)다. 머그는 손잡이 때문에 정확히 대칭인 회전이 없어 적지
않았다. 그래서 Mug Cleanup의 머그 잡기는 집게 180도만 맞춰지고 평균 47.6도 차이가 남는다. Square 막대에 끼우기는 회전은 맞지만
위치 차이가 평균 13.9 cm에서 10.1 cm로만 줄어든다. 이 계산은
`/mnt/nvme0/jihoonkwon/tmg_tmp/hammer_diag/mg_align_offline.py`(A100 서버)로 했다.

## 2. 미리 만들어 둔 파일

과제마다 파일 두 개를 커밋해 두었다. 두 파일은 같은 생성 기록에서 만들었지만 서로 다른 파일이고, 서로 다른 메서드가 읽는다.

| 파일 | 담긴 내용 | 읽는 메서드 |
|---|---|---|
| `configs/experiment/generator/mimicgen/coefficients/<과제>.yaml` | 1절 식의 계수. 서브태스크마다 원본별 α와 공통 기울기 a, b, c, d, e가 들어 있다 | `mixup_only`와 `trackmix`(섞기 방식 `logistic`) |
| `configs/experiment/generator/mimicgen/ranking/<과제>.yaml` | 서브태스크마다 원본을 전체 과제 성공률이 높은 순서로 늘어놓은 순위 | `rank_top1`, `rank_top3`, `rank_top5`(순위 상위 K개 원본 가운데 하나를 고르는 비교 실험) |

두 파일을 만든 생성 기록은 과제마다 3,000번 시도한 풀이다. 풀은 생성 시도 기록을 모은 폴더를 말한다. 이 풀은 섞기 없이
(`METHOD=none`) 서브태스크마다 원본을 균등하게 무작위로 골라서(`SELECTION=ours`) 만들었다. 계수는 서브태스크 k에 도달한 시도만
써서 최대우도법으로 맞췄다. 정답 값은 그 서브태스크의 성공 여부다. 서브태스크 성공은 MimicGen이 매 단계 기록하는 서브태스크 종료
신호로 판정했다. 예를 들어 Threading 첫 서브태스크는 바늘을 잡으면 켜지는 `grasp` 신호를 쓴다. 마지막 서브태스크의 성공은
과제 전체의 성공이다.

1절 4단계의 물체 대칭은 과제 설정 파일 `configs/experiment/generator/mimicgen/task/<과제>.yaml`의 `mixing_symmetries`에 있다.
계수 파일과 순위 파일은 이 변경으로 바뀌지 않았다. 계수는 섞기 없이 만든 풀에서 맞췄기 때문이다.

계수 파일 첫머리의 `basis` 줄에 어느 풀에서 몇 번의 시도로 맞췄는지 적혀 있다. 과제별 서브태스크 종료 신호는
`configs/experiment/generator/mimicgen/task/<과제>.yaml`의 `subtask_term_signals`에 있다.

## 3. 장면 범위

2026-11-04에 네 과제(Hammer Cleanup, Mug Cleanup, Threading, Three Piece Assembly)의 장면 범위를 좁혔다. 장면 범위는 물체를
처음 놓는 위치와 회전의 범위다. 이 범위는 `configs/experiment/generator/mimicgen/scene/ours.yaml`에 들어 있고, 이번 실행도 이
파일을 그대로 쓴다.

계수 파일도 좁힌 범위로 맞췄다. 네 과제의 3,000번 풀은 2026-11-13에 좁힌 범위로 새로 생성했다. 나머지 세 과제(Coffee, Square,
Stack Three)는 범위가 바뀌지 않아 2026-10-29에 만든 풀을 그대로 썼다. Threading에는 범위를 좁히기 전인 2026-10-29에 만든 풀도
있었다. 이 풀은 계수에 쓰지 않았고 폴더 이름을 `rank_human_demos_20261029_oldrange`로 바꿔 남겨 두었다.

## 4. 돌리는 실험

| 항목 | 값 |
|---|---|
| 데이터 생성기 | MimicGen, 공식 원본 선택 방식(`SELECTION=official`, 드라이버 기본값으로 기준선과 같다) |
| 생성 메서드 | `mixup_only`. 1절의 섞기만 켜고 경로 추적은 끈다. 데이터셋에 저장하는 동작은 기준선과 같이 실제로 실행한 동작이다 |
| 과제 | MimicGen 7개 전부(Coffee, Square, Stack Three, Threading, Hammer Cleanup, Mug Cleanup, Three Piece Assembly) |
| 정책 | Diffusion Policy와 BC-RNN 둘 다 |
| 시드 | 202601, 202602, 202603 (정책마다 세 번 학습) |
| 학습 데이터 | 과제마다 성공 시연 1,000개, 영상 입력(agentview와 손목 카메라, 84×84) |
| 학습 | 10만 단계, 1만 단계마다 체크포인트를 저장해 모두 10개 |
| 평가 | 체크포인트마다 검증 장면 25개로 성공률을 재고, 가장 높은 체크포인트 하나를 시험 장면 300개로 시험한다 |

평가 장면 파일은 기준선이 쓴 파일과 같다. 그래서 두 결과를 같은 장면에서 비교할 수 있다.

## 5. 실행 순서

아래 명령은 `docs/MIMICGEN_BASELINE_RUNBOOK.md`의 예와 같이 머신 설정 이름을 `lab`, 저장소 위치를 `~/TrackMixGen`, 데이터
폴더를 `/data/tmg_data`로 적었다. 실제 값으로 바꿔서 쓴다. 데이터 폴더는 `make -s print-data-root CLUSTER=lab`으로 확인할 수 있다.

### 5.1 코드 받기

```bash
cd ~/TrackMixGen && git pull origin main && git log --oneline -3
```

출력에 `Logistic source mixing with a separate coefficient file` 커밋과, 그 위에 대칭 맞추기 커밋
`Logistic mixing: align each source to the selected one by symmetry before blending`이 있어야 한다. 대칭 맞추기 커밋보다 앞선 코드로
섞기 풀을 이미 만들기 시작했다면 7절을 본다. 이번 변경은 파이썬 코드, 설정 파일,
스크립트만 바꿨다. 컨테이너 이미지는 저장소 폴더를 `/opt/tmg`로 붙여 쓰므로 이미지를 다시 만들 필요는 없다.

### 5.2 계수 파일 확인

```bash
cd ~/TrackMixGen && ls configs/experiment/generator/mimicgen/coefficients/
```

`coffee.yaml`, `hammer_cleanup.yaml`, `mug_cleanup.yaml`, `square.yaml`, `stack_three.yaml`, `threading.yaml`,
`three_piece_assembly.yaml` 7개가 보여야 한다.

### 5.3 실행

```bash
cd ~/TrackMixGen && CLUSTER=lab GPU=0 METHOD=mixup_only \
  nohup scripts/run_mimicgen_baseline.sh > /data/tmg_data/logs/mimicgen_mixup_only_nohup.log 2>&1 &
```

드라이버는 기준선과 같은 `scripts/run_mimicgen_baseline.sh`이고, `METHOD=mixup_only`만 더 준다. `TASKS`를 주지 않으면 7개 과제를
모두 돈다. `POLICIES`(기본값 Diffusion Policy와 BC-RNN), `SEEDS`(기본값 세 시드), `SIZE`(기본값 1,000)는 기본값 그대로 두면 4절의
실험이 된다. GPU를 여러 장 쓸 때는 기준선 런북 4절처럼 `GPU`와 `POLICY_PARALLEL`을 조정한다. 진행은
`/data/tmg_data/logs/mimicgen_mixup_only/driver.log`에서 볼 수 있다.

## 6. 결과 위치와 보고

실행 하나의 시험 결과는 아래 파일이다. 기준선과 다른 곳은 경로의 메서드 자리(`mixup_only`)뿐이다.

```
<데이터 폴더>/runs/mimicgen/<과제>/ours/official/mixup_only/pool_robosuite_image/image/<정책>/seed<시드>/eval/test_h600/<체크포인트>/eval_summary.json
```

이 파일의 `all.success_rate`가 시험 장면 300개의 성공률이다. 표에는 세 시드 성공률의 평균 ± 표본 표준편차를 퍼센트로 적는다.
같은 과제와 정책의 기준선 결과는 경로의 메서드 자리가 `none`인 파일이고, 두 값을 나란히 적는다. 드라이버가 끝나면 정책별 묶음
파일이 `<데이터 폴더>/reports/mimicgen_mixup_only_<날짜>_<정책>` 아래에 생긴다.

생성 단계의 섞기 기록도 남는다. 풀의 `chunks/*/records/attempts.jsonl`에서 시도마다 `subtasks[i].mix`를 보면 서브태스크마다
원본별 예측 확률(`predicted`), 가중치(`weights`), 장면 특징(`scene_features`), 손안 특징(`inhand_features`)이 적혀 있다.
대칭 맞추기 결과는 `subtasks[i].mix.symmetry`에 원본 번호별로 적힌다. `symmetry`는 고른 후보 이름(`identity`는 그대로,
`eef:z180`은 집게 180도, `obj:z180`은 물체 z축 180도, 두 개를 함께 쓰면 `obj:z180+eef:z180`)이다. `pos_cm`와 `rot_deg`는 맞춘 뒤
선택된 원본과의 위치 거리(cm)와 회전 각도(도)이고, `identity_pos_cm`와 `identity_rot_deg`는 맞추기 전 값이다.

## 7. 주의할 점

- 기준선 결과 폴더(`.../official/none/...`)는 옮기거나 지우지 않는다. 섞기 실행은 `.../official/mixup_only/...`에 따로 쓰므로 기준선
  결과와 섞이지 않는다. 평가 장면 파일(`<데이터 폴더>/resets/mimicgen/<과제>/ours/val25_test300_seed202600.hdf5`)은 기준선과 함께
  쓰는 파일이다. 이미 있으면 그대로 쓰고, 없으면 드라이버가 기준선과 같은 방법으로 만든다.
- 2026-11-04 범위 변경 뒤에 기준선을 다시 돌리지 않은 과제가 있다면, 그 과제의 기준선 폴더에 이전 범위의 결과가 남아 있을 수 있다.
  비교하기 전에 기준선 결과가 새 범위로 만든 것인지 확인한다. 확인 방법은 `docs/MIMICGEN_RERUN_2026-11-04.md` 3.2절에 있다.
- 계수 파일이 없는 과제를 섞기 메서드로 돌리면 `make generate`가 먼저 `scripts/ensure_coefficients.sh`를 부른다. 이 스크립트는
  3,000번 풀을 생성하고(`METHOD=none`, `SELECTION=ours`, `GENERATION=rank_human_demos`), 순위 파일과 계수 파일을 쓴 뒤 본 생성을
  시작한다. 7개 과제 파일은 모두 커밋해 두었으므로 이번 실행에서는 이 단계가 돌지 않는다. 계수를 다시 맞추고 싶으면 그 과제의
  계수 파일을 지우고 `make coefficients GENERATOR=mimicgen TASK=<과제> SCENE=ours CLUSTER=lab`을 돌린다.
- 대칭 맞추기 커밋보다 앞선 코드로 시작한 섞기 풀(`.../official/mixup_only/...`)은 이어서 돌리지 말고 다시 만든다. 메서드 설정에
  `mixing.symmetries`가 새로 생겨서, 새 코드로 이어 돌리면 생성기가 끝난 청크의 설정이 다르다는 오류(`finished chunks differ from
  the requested configuration`)를 내고 멈춘다. 그 풀 폴더를 다른 이름으로 옮긴 뒤 5.3절 명령을 다시 돌린다.
- 섞기 메서드는 기준선보다 생성 성공률이 다를 수 있다. 드라이버는 성공 시연이 1,000개가 될 때까지 시도 수를 25 %씩 늘리므로,
  성공률이 낮으면 시간만 더 든다.
- 순위 기반 비교 실험(`METHOD=rank_top1`, `rank_top3`, `rank_top5`)도 같은 명령에서 `METHOD`만 바꿔 돌릴 수 있다. 이번 요청에는
  들어 있지 않다.
