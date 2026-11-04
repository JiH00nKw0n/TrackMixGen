# MimicGen 기준선 재실험 안내 (2026-11-04 범위 변경)

이 문서는 장면 범위를 좁힌 네 과제(Hammer Cleanup, Mug Cleanup, Threading, Three Piece Assembly)의 MimicGen 영상 기준선을
다시 돌리는 방법을 적는다. 학습과 평가 방법은 그대로이고, 물체를 처음 놓는 위치와 회전의 범위만 바뀌었다. 나머지 세 과제
(Coffee, Square, Stack Three)는 범위가 그대로라서 다시 돌리지 않는다.

## 차례

- [1. 바뀐 것](#1-바뀐-것): 네 과제의 범위를 이전 값과 새 값으로 나란히 적는다.
- [2. 다시 돌리는 실험](#2-다시-돌리는-실험): 과제, 정책, 시드, 데이터 수와 평가 방식을 적는다.
- [3. 실행 순서](#3-실행-순서): 코드 받기, 이전 결과 옮기기, 실행 명령을 순서대로 적는다.
- [4. 결과 확인과 보고](#4-결과-확인과-보고): 결과 파일 위치와 표에 넣는 방식을 적는다.
- [5. 주의할 점](#5-주의할-점): 이전 결과가 다시 쓰이는 문제와 확인한 사항을 적는다.

## 1. 바뀐 것

바뀐 범위는 main 브랜치 커밋 `a7326fa`에 들어 있다. 파일은 `configs/experiment/generator/mimicgen/scene/ours.yaml`이고,
CP-Gen 실험이 쓰는 `configs/experiment/generator/cpgen/scene/ours.yaml`도 같은 값으로 바뀌었다.

이전 범위(2026-10-26에 정한 범위)로 돌린 영상 기준선은 네 과제에서 성공률이 0–35 %에 그쳤다. MimicGen 논문의 D1 설정 성공률은
같은 과제에서 61–64 %다. 그래서 위치 범위의 양 끝을 MimicGen D1과 이전 범위의 중간쯤으로 옮겼다. 값은 0.05 m 단위로 맞추고,
딱 떨어지지 않는 곳만 0.025 m 단위를 썼다. 이전 범위와 D1의 차이가 0.05 m뿐인 쪽은 D1 값을 그대로 썼다. 회전 범위는 이전에도
D1과 같았으므로 그대로 두었다. Three Piece Assembly는 위치가 이미 D1과 같아서 회전만 D1(고정)과 D2의 중간으로 바꿨다.

아래 표에서 위치는 테이블 중심을 기준으로 한 미터 단위다. x는 로봇에서 멀어지는 방향이고 y는 좌우 방향이다. 괄호 안 숫자는
범위의 폭이다.

| 과제와 물체 | 축 | MimicGen D1 | 이전 범위 | 새 범위 |
|---|---|---|---|---|
| Hammer Cleanup 망치 | x | −0.20–0.20 (0.40) | −0.20–0.35 (0.55) | −0.20–0.25 (0.45) |
| Hammer Cleanup 망치 | y | −0.25 – −0.13 (0.12) | −0.31 – −0.07 (0.24) | −0.275 – −0.10 (0.175) |
| Hammer Cleanup 서랍 | x | 0–0.20 (0.20) | −0.10–0.30 (0.40) | −0.05–0.25 (0.30) |
| Hammer Cleanup 서랍 | y | 0.20–0.30 (0.10) | 0.20–0.35 (0.15) | 0.20–0.30 (0.10) |
| Mug Cleanup 서랍 | x | −0.15–0.05 (0.20) | −0.25–0.15 (0.40) | −0.20–0.10 (0.30) |
| Mug Cleanup 서랍 | y | 0.25–0.35 (0.10) | 0.20–0.35 (0.15) | 0.25–0.35 (0.10) |
| Mug Cleanup 머그 | x | −0.25–0.15 (0.40) | −0.35–0.35 (0.70) | −0.30–0.25 (0.55) |
| Mug Cleanup 머그 | y | −0.30 – −0.15 (0.15) | −0.35 – −0.075 (0.275) | −0.30 – −0.10 (0.20) |
| Threading 바늘 | x | −0.20–0.05 (0.25) | −0.325–0.175 (0.50) | −0.25–0.10 (0.35) |
| Threading 바늘 | y | 0.15–0.25 (0.10) | 0.10–0.30 (0.20) | 0.125–0.275 (0.15) |
| Threading 삼각대 | x | −0.10–0.15 (0.25) | −0.225–0.275 (0.50) | −0.15–0.20 (0.35) |
| Threading 삼각대 | y | −0.20 – −0.10 (0.10) | −0.25 – −0.10 (0.15) | −0.20 – −0.10 (0.10) |
| Three Piece Assembly 받침 | 회전 | 0°로 고정 | ±45° (D2와 같음) | ±20° |
| Three Piece Assembly 조각 1·2 | 회전 | 90°로 고정 | 0°–180° (D2와 같음) | 45°–135° |

Three Piece Assembly의 세 물체 위치는 이전과 같이 x와 y 각각 ±0.22 m다. 모든 과제에서 로봇 팔이 닿는 0.85 m 안쪽에만 물체를
놓는 조건도 그대로다.

## 2. 다시 돌리는 실험

| 항목 | 값 |
|---|---|
| 데이터 생성기 | MimicGen, 공식 원본 선택 방식(`SELECTION=official`, 드라이버 기본값) |
| 과제 | Hammer Cleanup, Mug Cleanup, Threading, Three Piece Assembly |
| 정책 | Diffusion Policy와 BC-RNN 둘 다 |
| 시드 | 202601, 202602, 202603 (정책마다 세 번 학습) |
| 학습 데이터 | 과제마다 성공 시연 1,000개, 영상 입력(agentview와 손목 카메라, 84×84) |
| 학습 | 10만 단계, 1만 단계마다 체크포인트를 저장해 모두 10개 |
| 평가 | 체크포인트마다 검증 장면 25개로 성공률을 재고, 가장 높은 체크포인트 하나를 시험 장면 300개로 시험한다 |

이전 실험과 다른 것은 장면 범위뿐이다. 학습 단계 수, 평가 장면 수, 체크포인트를 고르는 규칙은 바꾸지 않았다.

## 3. 실행 순서

아래 명령은 `/Users/jihoonkwon/Desktop/projects/research/TrackMixGen_claude/docs/MIMICGEN_BASELINE_RUNBOOK.md`의 예와 같이 머신
설정 이름을 `lab`, 저장소 위치를 `~/TrackMixGen`, 데이터 폴더를 `/data/tmg_data`로 적었다. 실제 값으로 바꿔서 쓴다. 데이터 폴더는
`make -s print-data-root CLUSTER=lab`으로 확인할 수 있다.

### 3.1 코드 받기

```bash
cd ~/TrackMixGen && git pull origin main && git log --oneline -1
```

마지막 줄이 `a7326fa` 또는 그 뒤의 커밋이면 된다. 이번 변경은 설정 파일과 스크립트만 바꿨으므로 컨테이너 이미지를 다시 만들
필요는 없다.

### 3.2 이전 범위로 만든 결과를 옆으로 옮기기

드라이버는 이미 있는 결과를 다시 쓴다. 과제의 데이터셋과 평가 장면 파일이 있으면 그 과제의 생성을 건너뛰고, 시험 결과가 있는
실행은 학습과 평가를 건너뛴다. 이전 결과가 같은 폴더(`<과제>/ours`)에 남아 있으면 새 범위로 다시 돌지 않고 이전 결과가 그대로
쓰인다. 그래서 네 과제의 폴더 이름을 먼저 바꿔 둔다. 지우지 않고 이름만 바꾸므로 이전 결과는 그대로 남는다.

```bash
D=$(make -s -C ~/TrackMixGen print-data-root CLUSTER=lab)
for t in hammer_cleanup mug_cleanup threading three_piece_assembly; do
  for kind in pools datasets resets runs; do
    [ -d "$D/$kind/mimicgen/$t/ours" ] && mv "$D/$kind/mimicgen/$t/ours" "$D/$kind/mimicgen/$t/ours_20261026"
  done
done
```

옮긴 뒤 `ls $D/datasets/mimicgen/hammer_cleanup`처럼 확인하면 `ours_20261026`만 남아 있어야 한다.

### 3.3 실행

```bash
cd ~/TrackMixGen && CLUSTER=lab GPU=0 TASKS="hammer_cleanup mug_cleanup threading three_piece_assembly" \
  nohup scripts/run_mimicgen_baseline.sh > /data/tmg_data/logs/mimicgen_rerun_20261104_nohup.log 2>&1 &
```

`POLICIES`(기본값 Diffusion Policy와 BC-RNN), `SEEDS`(기본값 세 시드), `SIZE`(기본값 1,000)는 기본값 그대로 두면 2절의 실험이
된다. GPU를 여러 장 쓸 때는 런북 4절처럼 `GPU`와 `POLICY_PARALLEL`을 조정한다. 진행은
`/data/tmg_data/logs/mimicgen_baseline/driver.log`에서 볼 수 있다.

## 4. 결과 확인과 보고

결과 위치는 이전과 같다. 실행 하나의 시험 결과는 아래 파일이다.

```
<데이터 폴더>/runs/mimicgen/<과제>/ours/official/none/pool_robosuite_image/image/<정책>/seed<시드>/eval/test_h600/<체크포인트>/eval_summary.json
```

이 파일의 `all.success_rate`가 시험 장면 300개의 성공률이다. 표에는 세 시드 성공률의 평균 ± 표본 표준편차를 퍼센트로 적는다.
드라이버가 끝나면 정책별 묶음 파일이 `<데이터 폴더>/reports/` 아래에 생긴다.

## 5. 주의할 점

- 3.2절의 폴더 옮기기를 빼먹으면 드라이버가 이전 범위의 결과를 그대로 다시 쓰고, 실행이 빠르게 끝난 것처럼 보인다. 로그에
  `dataset and resets exist, generation skipped`나 `test result exists, skipped`가 네 과제에 대해 나오면 폴더가 옮겨지지 않은
  것이다.
- 새 범위는 실행 전에 확인했다. MimicGen 환경을 과제마다 100번 리셋(물체를 새로 배치)해 보니 네 과제 모두 배치에 실패한 리셋이
  없었고, 물체가 설정한 범위 안에 놓였다. Hammer Cleanup은 100번 중 17번 서랍이 로봇 집게에 닿았다. 이전 범위에서는 200번 중
  22번, MimicGen D1에서는 200번 중 64번이었으므로 과제 자체의 서랍 위치에서 생기는 접촉으로 보고 범위를 더 줄이지 않았다.
- 단위 테스트는 276개가 통과했다. 실패한 2개(`tests/unit/test_ranking.py`의 Soft 순위 기본값 테스트와
  `tests/unit/test_autodata_runtime.py`의 Dex 추적 테스트)는 이번 변경 전의 main에서도 같은 방식으로 실패하며, MimicGen 기준선과
  관계가 없다.
