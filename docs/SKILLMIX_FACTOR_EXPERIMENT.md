# SkillMix 요인 실험: 조건과 실행 방법

SkillMix 섞기가 어떤 과제에서는 정책 성공률을 올리고 어떤 과제에서는 내리는 까닭을 가르기 위한 실험이다. SkillMix는 두 가지를 함께
한다. 하위작업마다 어느 원본 시연을 쓸지 고르고(선택), 고른 원본들을 가중 평균한다(섞기). 조건마다 이 가운데 하나만 바꿔 저차원 풀을
새로 만들고, 저차원 Diffusion Policy를 시드 3개로 학습해 시험 300장면에서 비교한다. 조건은 모두 설정 파일로 정의되어 있고, 코드에는
과제 이름, 원본 번호, 문턱값이 없다.

## 1. 조건

조건 하나는 메서드 설정 하나다(`configs/experiment/method/<메서드>.yaml`). 메서드 이름이 풀, 데이터셋, 학습 실행 폴더 이름에 들어가므로
조건끼리 폴더가 겹치지 않는다.

| 조건 | 메서드 (`METHOD`) | 원본 선택 | 섞기 | 설정에서 바뀌는 값 |
|---|---|---|---|---|
| C0 | `none` | 생성기가 뽑은 원본 그대로 | 없음 | (기존 설정) |
| C1 | `skillmix` | 적합된 점수 | 적합된 `d`로 `softmax(s / d)`, `d <= 0`인 하위작업은 점수가 가장 큰 원본 하나 | (기존 설정) |
| C2 | `skillmix_select` | 점수에 따른 확률 `softmax(s)`로 원본 하나를 뽑음 | 없음 | 파생 규칙 `select_only`(모든 하위작업 `d = -1`), `single_source: sample` |
| C3 | `skillmix_top1` | 점수가 가장 큰 원본 하나 | 없음 | 파생 규칙 `select_only`, `single_source: best` |
| C4 | `skillmix_uniform` | 없음(호환되는 원본 모두) | 거의 균등한 가중치 `softmax(s / 1000)` | 파생 규칙 `uniform`(모든 하위작업 `d = 1000`) |
| C7 | `skillmix_distcap` | C1과 같음 | C1에서 먼 원본을 뺌 | `mixing.max_source_distance: {rot_deg: 30, pos_cm: 3}` |
| C8 | `skillmix_neff3` | C1과 같음 | 유효 원본 수가 3 이상이 되도록 온도를 올림 | `mixing.min_effective_sources: 3` |
| G | `skillmix_mix_grasp` | 점수에 따른 확률로 뽑음 | 잡기 하위작업만 적합된 `d`로 섞음 | 파생 규칙 `mix_grasp`, `single_source: sample` |
| P | `skillmix_mix_place` | 점수에 따른 확률로 뽑음 | 놓기와 끼우기 하위작업만 적합된 `d`로 섞음 | 파생 규칙 `mix_place`, `single_source: sample` |

세 가지 옵션은 메서드 설정에 적혀 있을 때만 켜진다. 기존 메서드 설정(`none`, `skillmix`, `skillmix_sample`, `skillmix_sparsemax`)에는
키를 더하지 않았으므로 기존 풀의 청크 식별값과 결과는 그대로다.

- **파생 규칙 파일 (`mixing.derived: <조건>`).** 적합된 규칙 파일을 이름 붙은 변환으로 바꾼 사본을 읽는다. 변환은
  `configs/experiment/derived_coefficients.yaml`에 조건별로 적혀 있고, `scripts/derive_coefficients.py`가
  `configs/experiment/generator/<생성기>/coefficients/derived/<조건>/<과제>.yaml`로 쓴다. 파일 머리말과 `derived` 블록에 어느 파일에서
  어떤 변환으로 만들었는지와 원본 파일의 sha256이 남는다. 변환은 두 가지다. `set_d`는 고른 하위작업의 `d`를 바꾸고, `boost_a`는 고른
  하위작업에서 원본 하나의 `a`에 값을 더한다(원본 하나를 강제할 때). 하위작업은 `all`, 번호 목록, `{kind: grasp}`,
  `{except_kind: grasp}`로 고른다. 하위작업 종류는 `configs/experiment/generator/<생성기>/stages/<과제>.yaml`의 `subtask_kinds`에 있다.
- **거리 상한 (`mixing.max_source_distance`).** 대칭 정렬 뒤 선택된 원본(생성기가 뽑은 원본)의 자세와 회전 또는 위치가 상한보다 먼
  원본을 빼고 남은 가중치를 다시 정규화한다. 거리는 정렬 단계가 원본마다 기록하는 `rot_deg`, `pos_cm`이다. 선택된 원본 자신은
  빠지지 않는다. 선택된 원본에 가중치가 없고 남은 원본이 모두 상한 밖이면 가중치가 가장 큰 원본 하나만 남긴다. MimicGen과 CP-Gen이
  같은 함수(`SkillMixer.cap`)를 쓴다. 시도 기록의 `mix.distance_cap`에 상한, 빠진 원본(`dropped`, `n_dropped`), 빼기 전 가중치가 남는다.
- **유효 원본 수 하한 (`mixing.min_effective_sources: k`).** 유효 원본 수는 `1 / sum_j w_j^2`이다. 적합된 가중치가 `k`보다 작으면
  온도(`softmax(s / d)`의 `d`)를 `k`에 닿는 가장 작은 값까지 올린다. 호환되는 원본이 `k`개보다 적으면 그 원본들에 균등하게 나눈다.
  적합이 `d <= 0`을 준 하위작업도 섞인다. 기록의 `mix.temperature`에 올린 온도가 남는다.

과제에 따라 두 조건이 같은 풀을 만들 수 있다. 적합된 `d`가 모든 하위작업에서 0 이하인 과제(CP-Gen Threading)에서는 C1이 C3과 같고
G와 P가 C2와 같다. 3절의 확인 실행이 하위작업별 `d`와 섞인 비율을 보여 주므로 전체 실행 전에 알 수 있다.

## 2. 파생 규칙 파일 만들기

저장소에 들어 있는 파생 파일은 현재 적합 파일에서 만든 것이다. 규칙을 다시 적합했거나 조건을 더하면 다시 만든다. numpy와 PyYAML이
있는 곳(저장소를 설치한 환경이나 이미지 안)에서 돌린다.

```bash
python3 scripts/derive_coefficients.py --all        # 설정된 모든 조건 x 적합 파일이 있는 모든 과제
python3 scripts/derive_coefficients.py --check      # 파생 파일이 지금의 적합 파일과 맞는지 확인(단위 시험도 확인한다)
# 명령줄에서 정의하는 조건. 예: CP-Gen Threading의 끼우기(하위작업 1)에서 원본 0을 강제
python3 scripts/derive_coefficients.py --generator cpgen --task threading --condition force_source0_insert \
    --transform '{type: set_d, subtasks: all, value: -1}' --transform '{type: boost_a, subtasks: [1], source: 0, by: 100}'
```

명령줄에서 정의한 조건을 돌리려면 `mixing.derived: force_source0_insert`를 가진 메서드 설정을 하나 더하고(이름은 새로 짓는다),
CP-Gen이면 `configs/experiment/generator/cpgen.yaml`의 `method_overrides`에 그 이름으로 `{labels: pre_noise}`를 더한다.

## 3. 실행

드라이버는 `scripts/run_factor_experiment.sh`다. 조건마다 `scripts/run_mimicgen_baseline.sh`를 그 메서드, 생성 설정
`pool_factor_lowdim`, 관측 `low_dim`으로 한 번 부른다. 그래서 기준선 드라이버와 똑같이 이어 돌릴 수 있다. 끝난 풀, 데이터셋, 학습,
시험 결과는 건너뛴다. 생성 설정 `pool_factor_lowdim`은 `pool_robosuite_lowdim`과 내용이 같고 이름만 달라서, 예전 코드로 만든
`pool_robosuite_lowdim` 폴더를 이어 쓰는 일이 없다. 원본 선택은 모든 조건에서 `ours`(하위작업마다 원본을 무작위로 뽑음)다.

**먼저 50회 확인 실행.** 조건과 과제마다 50회 시도 풀(`test/factor_check50`)을 만들고 시도 기록에서 하위작업별 요약을 찍는다. 학습은
하지 않는다.

```bash
CLUSTER=a100 GPU=6 GENERATOR=mimicgen TASKS="threading coffee three_piece_assembly square" CHECK=1 scripts/run_factor_experiment.sh
CLUSTER=a100 GPU=6 GENERATOR=cpgen TASKS="threading" CHECK=1 scripts/run_factor_experiment.sh
```

요약(`scripts/mix_report.py`)은 하위작업마다 `d`, 둘 이상의 원본이 섞인 비율, 유효 원본 수(평균, 최소, 최대), 원본별 평균 가중치,
가중치가 가장 큰 원본의 분포, 거리 상한으로 원본이 빠진 비율과 평균 개수, 온도가 올라간 비율을 찍는다. 화면과
`<데이터 루트>/logs/factor_<생성기>/check_<메서드>_<과제>.report`에 남는다. 이미 만든 풀은 직접 요약할 수 있다.

```bash
python3 scripts/mix_report.py --pool /data/pools/mimicgen/threading/ours/ours/skillmix_distcap/factor_check50
```

조건별로 볼 것: C2와 C3은 섞인 비율 0 %, C4는 유효 원본 수가 호환 원본 수에 가깝고, C7은 `distance cap` 줄에 상한과 빠진 비율이
나오고, C8은 유효 원본 수의 최소가 3(호환 원본이 3개 미만인 경우는 그 수)이며, G와 P는 해당 종류의 하위작업에서만 섞인 비율이 0보다 크다.

**전체 실행.** 조건 하나씩 돌리려면 `CONDITIONS`에 메서드 이름을 준다. 주지 않으면 아홉 조건을 차례로 돌린다.

```bash
CLUSTER=a100 GPU=6 GENERATOR=mimicgen TASKS="threading coffee three_piece_assembly square" \
  nohup scripts/run_factor_experiment.sh > factor_mimicgen.out 2>&1 &
```

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `GENERATOR`, `TASKS` | `mimicgen`, (필수) | 생성기와 그 과제들 |
| `CONDITIONS` | 아홉 조건 | 돌릴 메서드 이름들 |
| `POLICY_PARALLEL` | 3 | 한 조건 안에서 GPU 한 장에 동시에 도는 학습과 평가 줄 수 |
| `CONDITION_PARALLEL` | 1 | 동시에 도는 조건 수 |
| `PORT_BASE` | 8800 | 첫 정책 서버 포트. `CONDITIONS`의 k번째 조건이 `PORT_BASE + k x (40 x 과제 수 x 정책 수 x 시드 수)`부터 한 블록을 쓴다. 같은 컴퓨터의 다른 드라이버는 이 범위 밖의 값을 써야 한다(드라이버가 시작할 때 범위를 찍는다) |
| `SEEDS`, `SIZE`, `POLICIES` | `202601 202602 202603`, 1000, `diffusion_policy` | 학습 시드, 풀과 데이터셋의 성공 시연 수, 정책 |

로그는 조건마다 `<데이터 루트>/logs/factor_<생성기>_<메서드>/`에, 드라이버 로그는 `<데이터 루트>/logs/factor_<생성기>/driver.log`에 쌓인다.
`MINI=1` 등 기준선 드라이버의 다른 변수도 그대로 전달된다.

## 4. 평가의 단계 기록

`tmg eval`이 에피소드마다 각 단계 신호가 처음 켜진 걸음을 `episodes.jsonl`의 `term_first`에 적는다(`{신호: 걸음 번호, 켜진 적 없으면 -1}`,
생성 기록의 `term_first`와 같은 키, 같은 형식, 같은 코드 `tmg.core.records.SignalLog`). `eval_summary.json`의 `all`과 `by_split` 블록에는
`stages`가 더해져 신호별 도달 비율(`n`, `n_reached`, `reach_rate`)을 준다. 실패한 에피소드가 잡기 전에 실패했는지, 잡은 뒤 끼우다
실패했는지를 이 값으로 가른다.

- MimicGen: 과제 설정의 `subtask_term_signals`가 있으면 MimicGen 환경 인터페이스의 종료 신호(`get_subtask_term_signals`, 생성 때 걸음마다
  기록하는 것과 같은 호출)를 읽는다.
- CP-Gen: 종료 신호가 없으므로 `configs/experiment/generator/cpgen/stages/<과제>.yaml`의 `stage_checks`에 적힌 검사를 쓴다.
  `object_lifted`는 물체가 에피소드 시작 높이보다 `height`(m) 넘게 올라가 있으면 켜진다. 물체 이름과 높이는 그 파일에 있다.

예전 `episodes.jsonl`에는 `term_first`가 없다. 그런 줄은 도달 비율 계산에서 빠지고 나머지 요약은 전과 같다. `stage_checks`와
`subtask_kinds`를 과제 설정이 아니라 `stages/` 파일에 둔 까닭은 과제 설정 전체가 끝난 청크의 식별값에 들어가기 때문이다. 과제 설정에 키를
더하면 그 과제의 기존 풀이 모두 "설정이 바뀜"으로 판정되어 이어 돌리거나 늘릴 수 없게 된다.

## 5. 알아 둘 것

- CP-Gen의 `object_lifted` 높이 0.04 m는 robosuite Lift 과제의 들기 판정값을 가져온 것이고, CP-Gen 장면에서 따로 재 보지 않았다.
- CP-Gen Coffee, Hammer Cleanup, Mug Cleanup은 규칙 파일의 하위작업(제약) 수가 과제 설정의 하위작업 수보다 많아 `subtask_kinds`를 적지
  않았다. 이 과제들에는 G와 P의 파생 파일이 없다. 제약 순서대로 종류를 적은 뒤 `scripts/derive_coefficients.py`를 다시 돌리면 생긴다.
- 새 메서드 이름은 CP-Gen의 `method_overrides`에 `labels: pre_noise`로 등록되어 있다. DexMimicGen은 메서드 이름별로 손 이벤트 설정을
  따로 두므로, 그 생성기에서 이 조건들을 돌리려면 `configs/experiment/generator/dexmimicgen.yaml`에 같은 이름을 더해야 한다.
