# 생성기 어댑터와 접목 지점

이 문서는 공식 생성기 코드에 TrackMix와 기록 기능을 끼워 넣는 방법을 적는다. 공식 코드는 복사하지도 수정하지도 않는다.
실행 시점에 공식 함수의 소스를 읽어 해시를 확인하고, 구문 트리(AST)의 정해진 자리(앵커)에 우리 함수 호출 한 줄씩을 끼운 뒤
다시 컴파일해 바꿔 끼운다. 끼우는 코드는 `tmg/generation/generators/<생성기>/runtime.py`에, 끼우는 도구는
`tmg/generation/graft.py`에 있다.

## 1. MimicGen (NVlabs/mimicgen@72bd767)

끼우는 함수는 둘이다. 소스 해시는 `configs/experiment/generator/mimicgen.yaml`의 `graft_hashes`에 있고, 설치된 패키지의 해시가 다르면
청크 프로세스가 시작하지 않는다.

### 1.1 `DataGenerator.generate`

| 앵커(대입되는 이름) | 끼우는 문장 | 하는 일 |
|---|---|---|
| `new_initial_state = env.get_state()` 다음 | `_tmg.begin_attempt(env)` | 시도 기록을 시작하고 장면 메타(도달 제약의 거부 횟수 등)를 읽는다 |
| `src_subtask_object_pose = ...` 다음 | `src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())` | 원본 혼합. 방법이 꺼져 있으면 입력을 그대로 돌려준다 |
| `transformed_seq = WaypointSequence.from_poses(...)` 다음 | `_tmg.record_subtask(locals())` | 하위작업 기록(원본 번호, 구간, 기준 물체 자세, 이음 시작과 목표 자세, 혼합 기록) |
| 마지막 `return results` 앞 | `_tmg.end_attempt(results)` | 시도 기록을 쓴다. 라벨이 `pre_noise`이면 `results["actions"]`를 잡음 전 라벨로 바꾼다 |

원본 혼합은 두 번째 앵커에서 일어난다. 이 자리의 자세들은 원본 시연의 세계 좌표계 값이고, 바로 뒤에서 공식 코드가
`transform_source_data_segment_using_object_pose(obj_pose=현재 물체, src_eef_poses, src_obj_pose=원본 A의 물체)`로 현재 장면에
옮긴다. 그래서 혼합기는 A와 B를 각자의 기준 물체 좌표계로 옮겨 섞은 뒤 A의 물체 자세로 되돌려 놓는다. 공식 변환은 A의 기준
자세를 그대로 쓰므로 바뀌지 않는다.

### 1.2 `WaypointTrajectory.execute`

| 앵커 | 바뀌는 내용 | 하는 일 |
|---|---|---|
| `for seq in self.waypoint_sequences:` 몸통 첫 줄 | `_tmg.begin_segment(seq, self)` | 구간 기록 시작. 경로 추종이 켜져 있으면 이 구간의 추종기를 만든다 |
| `for j in range(len(seq)):` | `for j in _tmg.step_indices(seq):` | 끄면 `range(len(seq))` 그대로. 켜면 추종기가 끝날 때까지 j를 낸다 |
| `waypoint = seq[j]` | `waypoint = _tmg.waypoint(seq, j, env_interface)` | 끄면 `seq[j]`. 켜면 측정한 말단 자세로 다음 목표를 정한다 |
| `action_pose = env_interface.target_pose_to_action(...)` 다음 | `_tmg.pre_noise(action_pose)` | 잡음 전 명령을 보관한다. 추가 스텝이면 비공개 난수로 같은 잡음을 더한다 |
| `datagen_info = env_interface.get_datagen_info(...)` 다음 | `_tmg.record_step(waypoint, datagen_info, play_action)` | 스텝 기록(목표 자세, 실제 자세, 잡음 전 명령, 실행 명령, 종료 신호) |
| `cur_success_metrics = env.is_success()` 다음 | `_tmg.record_success(cur_success_metrics, env_interface)` | 스텝 뒤 상태의 성공 판정과 하위작업 종료 신호를 기록한다 |

끼운 문장들은 공식 지역 변수를 공통 계층에 넘기는 일만 한다. 기록은 `tmg/core/records.py`, 혼합은 `tmg/generation/method/`,
실행 중의 경로 추종과 난수 회계와 라벨은 `tmg/generation/execution.py`의 `TrajectoryExecution`이 맡는다. 이 계층은
생성기와 무관하므로 Isaac Lab 어댑터도 같은 코드를 쓴다.

난수 흐름에 대한 약속은 다음과 같다. 공식 실행기는 경로점마다 `np.random.randn(6)`을 한 번 부른다(`waypoint.noise`는 항상 실수라
분기가 늘 실행된다). 방법이 꺼져 있으면 끼운 문장은 난수를 뽑지 않으므로 전역 난수 흐름이 공식 실행과 같다. 경로 추종이 켜지면
한 하위작업의 궤적(이음 구간과 기술 구간 전부)을 하나의 경로로 따라가며, 스텝마다 새로 지나친 원 경로점 하나당 전역 난수를
하나씩 순서대로 뽑아 그 경로점을 명령하는 스텝에 적용하고, 추가 스텝에는 청크 시드로 만든 비공개 난수를 쓴다. 끼운 경로점은
`noise=None`으로 넘겨 공식 분기가 난수를 뽑지 않게 하고, `pre_noise`가 공식 문장과 같은 식으로 잡음을 더하고 자른다. 그래서 한
하위작업에서 전역 난수를 뽑는 횟수가 공식 실행과 같고, 같은 시드의 기준선 실행과 방법 실행이 같은 장면 순서를 본다. 혼합이 일어난
구간은 길이가 바뀌므로 그 시도부터는 짝이 맞지 않는다.

### 1.3 양팔 과제에서 지켜야 할 것 (Isaac Lab 어댑터가 따르는 규칙)

한 하위작업의 양팔 경로는 추종기 하나가 같은 경로 변수로 전진시킨다. 손가락 명령의 손 구분은 환경 설정의 관절 이름
(`L_`, `R_` 접두사, `tmg/core/actions.py`의 `split_by_prefix`)으로 만들고, 공식 mimic 환경의 `actions[:, 14:25]` 분할은
쓰지 않는다. 집게 사건은 두 손 모두에서 찾아 합친다(`tmg/generation/method/grippers.py`의 `closure` 검출기).

추종기의 오차는 목표 자세와 같은 좌표계의 측정 자세로 재야 한다. GR1 공식 환경은 역기구학 목표를 `*_hand_pitch_link`에
걸지만 관측의 말단 자세 `*_eef_pos`, `*_eef_quat`는 `*_hand_roll_link`를 읽는다. 두 링크는 손목 피치 관절 하나로 이어져 있어
그 사이의 회전(0.3에서 0.7 rad)이 "추적 오차"로 잡히고, 기다려도 줄지 않는다. Isaac Lab 어댑터는 역기구학 목표 링크의 자세를
직접 읽어 오차를 계산한다(2026-10-26 Dex 분석).

### 1.4 원본 시연 선택 방식: 지금은 우리 값, 최종 평가는 MimicGen 공식 값 (2026-10-27 결정)

생성 프로토콜은 `configs/experiment/generation/pool_robosuite_image.yaml`의 `protocol` 블록에 있고, 다른 generation 설정(`pool_robosuite_lowdim`,
`rank_human_demos`, `pool_isaac`, `test/` 아래의 시험용 설정들)은 이 블록을 물려받는다. 지금 값은 하위작업마다 원본 시연을 독립적으로 고르고
(`select_src_per_subtask: true`) 고르는 방식은 무작위다(`selection_strategy: random`). MimicGen 공식 설정 템플릿
(`mimicgen/exps/templates/robosuite/*.json`)은 일곱 과제 모두 `select_src_per_subtask: false`(한 시도의 모든 하위작업이 같은 원본
시연을 쓴다)다. 그러나 MimicGen이 논문 데이터를 만들 때 쓴 설정은 템플릿이 아니라 템플릿 위에 과제별 값을 덮어쓰는
`mimicgen/scripts/generate_core_configs.py`의 결과이고, 이 스크립트는 Stack Three(그리고 Stack, Pick Place)의
`select_src_per_subtask`를 true로 바꾼다(논문 부록 N.6도 같다). 그래서 공식 선택은 다음과 같다. Square는 첫 하위작업(너트 잡기)에서
현재 물체 자세와 가장 가까운 원본 3개 중 하나를 골라(`nearest_neighbor_object`, `nn_k: 3`) 시도 끝까지 쓰고, Stack Three는 모든
하위작업에서 그렇게 다시 고르며, 나머지 다섯 과제는 시도마다 원본 하나를 무작위로 골라 끝까지 쓴다. 공식 스크립트는 Square의 두
하위작업 모두에 `nearest_neighbor_object`를 넣고 템플릿은 둘째 하위작업을 `random`으로 두지만, Square는 시도마다 첫 하위작업에서 한 번만
고르므로 둘째 값은 쓰이지 않아 결과가 같다. 2026-10-30까지 이 문서와
코드는 템플릿만 보고 Stack Three도 false라고 적었고, 코드는 과제 설정의 `official_select_src_per_subtask`(Stack Three: true)로
고쳤다. 잘못된 값으로 만든 풀은 없다. 2026-10-30에 A100 서버와 Isaac 서버를 확인했을 때 공식 선택(`selection=official`)으로 만든
MimicGen Stack Three 풀은 어느 서버에도 없었다.

결정은 두 단계로 나뉜다.

- 지금 진행 중인 재현 확인과 혼합 방법 탐색은 현재 값으로 계속한다. 이미 만든 기준 풀 일곱 개와 탐색 결과를 그대로 쓰기 위해서다.
- 논문에 들어갈 최종 평가에서 MimicGen은 공식 선택(위 문단, `selection=official`)을 그대로 써야 한다. 이를 위해
  필요한 작업은 두 가지다. 첫째, `tmg/generation/generators/mimicgen/config_builder.py`가 지금은 하나의 `selection_strategy`를 모든
  하위작업에 같이 넣으므로, 과제 템플릿의 하위작업별 값을 기본으로 쓰도록 바꿔야 한다(프로토콜에서 값을 주지 않으면 템플릿 값을
  남기는 방식). 둘째, 그 값으로 기준 풀 일곱 개를 다시 생성해야 한다. 최종 평가 전까지 이 두 가지가 끝나지 않으면 평가를 시작하지
  않는다.

2026-10-29에 첫째 작업을 했고, 선택 규칙을 풀 설정과 분리한 독립 축으로 만들었다(사용자 결정). 설정 묶음 `configs/experiment/selection/`의
`official`은 생성기의 공식 규칙(MimicGen은 `generate_core_configs.py`가 템플릿 위에 덮어쓴 값이다. `config_builder`가 null을 받으면 하위작업별 선택 방식은 템플릿 값을 그대로 두고, `select_src_per_subtask`는 과제 설정의 `official_select_src_per_subtask`가 있으면 그 값을 쓴다. CP-Gen은 원본 0번만),
`ours`는 우리 규칙(MimicGen은 하위작업별 무작위, CP-Gen은 원본 전부. CP-Gen 쪽은 시범별 제약 구간이 생기기 전까지 오류를 낸다)이다. 어느 풀
설정(영상, 저차원, 순위, 시험용)과도 조합할 수 있고 기본값은 `official`이다. 풀·데이터셋·실행 폴더는 장면 다음에 선택 규칙 이름이 들어간다
(`pools/<생성기>/<과제>/<장면>/<선택 규칙>/<방법>/<생성 설정>`). 2026-10에 만든 MimicGen 풀(순위 풀, 저차원 탐색 풀)은 모두 `ours` 규칙이고
그 폴더로 옮겼으며, Isaac Lab과 CP-Gen 풀은 `official` 폴더로 옮겼다. 둘째 작업(공식 값으로 기준 풀 생성)은 MimicGen 기준선 실행
(`scripts/run_mimicgen_baseline.sh`, 기본 `SELECTION=official`)이 한다.

선택 규칙을 `official`로 두어도 MimicGen 생성 설정 전체가 공식 값이 되지는 않는다. `config_builder.py`는 모든 풀에 생성
프로토콜(`configs/experiment/generation/pool_robosuite_image.yaml`의 `protocol` 블록)의 값을 함께 쓴다. 그중 행동 잡음 0.05, 보간 걸음 5,
고정 걸음 0은 공식 값과 같다. 원본 수(`source.n: null`, 원본 파일의 시연 전부)는 공개 원본 파일마다 시연이 10개라서 공식 값 10과 같은
원본을 쓴다. 성공 수 보장(우리는 끄고 정해진 수만큼 시도한 뒤 모자라면 시도를 늘린다. `generate_core_configs.py`는 성공 1,000개까지
돈다), 영상 풀의 실패 시연 파일(우리는 쓰지 않고 템플릿은 쓴다), 실패 시연 보관 상한(우리는 없고 `generate_core_configs.py`는 25),
시연 동영상 렌더링(우리는 끄고 `generate_core_configs.py`는 성공 10개와 실패 25개를 그린다)은 공식 값과 다르다.

현재 값과 공식 값의 차이가 결과에 미치는 영향은 다음과 같이 측정하고 추정했다.

- 측정한 것. Three Piece Assembly를 공식 D2와 같은 장면 범위로 두고 현재 프로토콜로 생성하면 11,000회 중 1,137회 성공(10.3%)이었고,
  이 풀에서 1,000개를 뽑아 학습한 저차원 BC-RNN(시드 1, 25장면 검증으로 고른 9만 번째 체크포인트)은 300장면 시험에서 14회
  성공(4.7%, 95% 구간 2.8–7.7%)이었다. 논문의 D2 생성 성공률은 31.3%, 저차원 BC-RNN 성공률은 38.7%다. 궤적 길이(평균 334단계,
  최대 406단계로 상한 600보다 짧다)와 관측 기록(물체 상태 42차원, 로봇 상태 9차원)은 정상이었다.
- 추정인 것. 생성과 정책 양쪽에서 같은 방향으로 차이가 나므로 원본 선택 방식의 차이가 유력한 원인이라고 추정하지만, 공식 값으로
  다시 생성해 비교한 적은 없어서 확인된 사실은 아니다. 정책 학습 갱신 횟수(현재 10만, 공식 저차원 설정 20만)와 평가 상한(현재 600,
  공식 500)도 다르다.

### 1.5 Three Piece Assembly D2 재현 조사 결과 (2026-10-27, A100)

논문의 D2 저차원 BC-RNN 성공률 38.7%가 우리 설정에서 나오지 않는 원인을 네 단계로 갈랐다. 결론은 "원본 선택 방식은 생성 성공률
차이만 설명하고, 정책 성공률은 robomimic 원본 구현으로 같은 데이터를 학습해도 우리와 같은 수준(10만 갱신에서 최대 4%)이므로 우리
학습·평가 파이프라인의 오류가 아니다"이다. 논문 수치와의 나머지 차이는 학습 예산과 보고 방식에서 나온다고 추정한다.

| 학습 데이터 (모두 D2 장면, 1,000개, 저차원 BC-RNN 시드 1, 10만 갱신) | 생성 성공률 | 25장면 검증 최고 | 300장면 시험(상한 600) |
|---|---|---|---|
| 우리 생성, 하위작업별 원본 독립 선택 (`pool_robosuite_lowdim`) | 10.3% (11,000회 중 1,137회) | 9만 체크포인트 2/25 | 14/300 = 4.7% (95% 구간 2.8–7.7%) |
| 우리 생성, 한 시도에 원본 하나 (`test/pool_robosuite_lowdim_onesrc`, 공식 템플릿 값) | 29.1% (4,000회 중 1,166회) | 10만 체크포인트 4/25 | 14/300 = 4.7% (95% 구간 2.8–7.7%) |
| 논문 공식 D2 데이터셋 (`scripts/import_robomimic_dataset.py`로 변환, `generation=imported_official`) | 해당 없음 | 7만 체크포인트 2/25 | 8/300 = 2.7% (95% 구간 1.4–5.2%) |
| 논문 보고값 | 31.3% | 해당 없음 | 38.7% (부록 O는 학습 중 체크포인트마다 50회 롤아웃하고, 각 에이전트가 시드 3개에 걸쳐 낸 최고 성공률을 보고한다고 적는다. 시드마다 최고 체크포인트를 고른 뒤 시드끼리 묶었는지, 시드 가운데 최대를 골랐는지는 그 문장으로 정해지지 않는다) |

우리 보고 방식은 MimicGen과 CP-Gen 어느 쪽과도 다르다. 우리는 시드마다 체크포인트를 검증 25장면으로 고르고 그 하나를 시험
300장면에서 평가하며, 학습 시드 3개의 평균과 표준편차를 적는다. MimicGen은 위 칸처럼 체크포인트별 최고값을 보고한다. CP-Gen(arXiv
2508.03944)은 시드 3개 각각의 최고 체크포인트를 50회씩 돌리고 그 평균과 표준오차를 보고한다.

측정으로 확인한 것은 다음과 같다.

- 원본 선택 방식을 공식 값으로 바꾸면 생성 성공률은 10.3%에서 29.1%로 올라 논문의 31.3% 근처가 된다. 정책 성공률은 4.7%로 변하지 않는다.
- 논문이 공개한 데이터셋을 우리 학습기와 평가기에 그대로 넣어도 2.7%다. 따라서 차이는 우리 데이터 생성이 아니라 학습 또는 평가에 있다.
- 평가 환경은 데이터의 환경과 일치한다. `scripts/check_replay_consistency.py`로 공식 시연 20개를 우리 평가 환경에 복원(`reset_to`와
  제어기 재고정)하고 기록된 행동을 그대로 다시 실행하면, 첫 관측은 기록과 오차 0으로 같고 20개 중 19개가 성공하며 말단 위치 어긋남은
  최대 2–3 mm다.
- 정책 호출 경로(관측 조립, 정규화, 은닉 상태 처리)에 결정적 결함은 보이지 않는다. `scripts/check_policy_teacher_forcing.py`로
  학습 시연의 관측을 서버의 `act` 경로에 한 단계씩 넣으면 팔 6채널 평균제곱오차 0.023(0을 내는 기준선 0.099), 그리퍼 부호 일치 90–96%다.
  정규화는 설정대로 모두 항등이고, 혼합 모형에서 성분 하나를 뽑는 방식은 robomimic과 같다.
- 학습 손실은 robomimic과 비슷한 속도로 내려간다(음의 로그가능도: 우리 2만 갱신 -12.9, robomimic 7천 갱신 -11.8).

남은 후보는 둘이었다. 학습기 구현의 세부 차이(robomimic BC-RNN-GMM과의 차이, 갱신 횟수 10만 대 공식 20만), 그리고 논문 보고 방식이
주는 상향 편향(시드마다 체크포인트 40개 가운데 최고값을 쓰는 방식. 시드 3개를 어떻게 묶었는지는 부록 O의 문장으로 정해지지 않는다). 이를 가르기 위해 같은 공식 데이터로 robomimic 원본 구현(robomimic 0.5.0,
robosuite 1.4.1, CPU torch 2.6)의 학습과 평가를 돌렸다. 설정은 공식 저차원 BC-RNN 템플릿과 같고, 1,000에폭(10만 갱신), 100에폭마다
50회 롤아웃, 상한 500, 시드 1개다(`/mnt/nvme0/jihoonkwon/tmg_data/robomimic_ref_three_piece_d2.py`, 로그 `logs/robomimic_ref/train.log`,
출력 `/mnt/nvme0/jihoonkwon/tmg_data/runs/robomimic_ref`). 2026-10-27 18:29 UTC에 정상 종료했다.

| robomimic 원본 구현, 공식 D2 데이터, 에폭(갱신 수) | 100 (1만) | 200 (2만) | 300 (3만) | 400 (4만) | 500 (5만) | 600 (6만) | 700 (7만) | 800 (8만) | 900 (9만) | 1000 (10만) |
|---|---|---|---|---|---|---|---|---|---|---|
| 50회 롤아웃 성공률 | 0% | 0% | 0% | 0% | 2% | 0% | 4% | 4% | 2% | 4% |

결론은 다음과 같다.

- 측정으로 확인한 것. robomimic 원본 구현도 같은 데이터와 같은 갱신 횟수에서 최대 4%(체크포인트 10개 가운데 최대값)다. 우리 학습기와
  평가기의 결과(공식 데이터 2.7%, 우리 생성 데이터 4.7%)와 같은 수준이므로, Three Piece Assembly D2의 낮은 정책 성공률은 우리
  BC-RNN 학습기나 평가기의 구현 오류 때문이 아니다. 우리 파이프라인은 이 과제에서 robomimic 원본과 같게 동작한다.
- 추정인 것. 논문의 38.7%와의 차이는 학습 예산(20만 갱신)과 보고 방식(체크포인트 가운데 최고값)에서 나온다고 보는 것이 지금까지의
  증거에 맞는다. 다만 10만 갱신까지의 추세(0%에서 4%로 느리게 오름)에서 추정한 것이고, 20만 갱신으로 늘렸을 때 38.7%에 이르는지는
  측정하지 않았다.
- 따라서 Three Piece Assembly D2는 재현 조사를 여기서 끝내고, 우리 학습기와 평가기는 그대로 둔다. 최종 평가에서 이 과제를 쓸 때는 논문
  수치와 직접 비교하지 말고, 같은 파이프라인 안에서 기준 방법과 제안 방법을 비교한다.

## 2. Isaac Lab 생성기: SoftMimicGen과 DexMimicGen

두 생성기는 Isaac Lab의 `generate_dataset.py` 스크립트로 돈다. 스크립트는 인자를 읽은 뒤 Isaac Sim을 띄우고 나서야 Isaac Lab
모듈을 들여오므로, 접목을 미리 설치할 수 없다. 청크 프로세스(`tmg/generation/generators/isaaclab/entrypoint.py`)는 스크립트
소스를 읽어 `if __name__ == "__main__":` 바로 앞에 `_tmg_install(globals(), _tmg_spec)` 한 줄을 끼운 사본을 실행한다. 그
함수는 스크립트의 들여오기가 끝난 뒤, `main()`이 돌기 전에 (1) 과제의 `register_modules`와 카메라 환경 모듈을 들여오고, (2)
`setup_env_config`를 감싸 장면 범위를 환경 설정에 써넣고, (3) 접목을 설치한다. 공식 파일은 디스크에서 바뀌지 않는다. 2026-10-29부터
이 스크립트 방식은 SoftMimicGen에 쓰이고, DexMimicGen은 같은 진입점으로 AutoData의 스크립트를 돈다(2.5절, 2.9절).

패키지는 Soft의 `softmimicgen.datagen`이다. DexMimicGen의 AutoData 런타임(2.9절)은 같은 런타임 기반(`IsaacLabRuntime`)을 상속해 자기 훅만 채운다. 표의 Dex 열(`generate_eef_subtask_trajectory`)은 Isaac Lab Mimic 경로로, 코드에는 남아 있지만 지금은 어떤 생성기 설정도 고르지 않는다. 끼우는 자리는 다음과 같다.

| 함수 | 앵커(대입되는 이름) | 끼우는 문장 | 하는 일 |
|------|---------------------|-------------|---------|
| `DataGenerator.generate` | `new_initial_state = ...` 뒤 | `_tmg.begin_attempt(self, env_id)` | 시도 기록을 열고, 잡음 전 라벨을 받기 위해 `env.target_eef_pose_to_action`을 감싼다 |
| `DataGenerator.generate` | 마지막 `return results` 앞 | `_tmg.end_attempt(self, env_id, results)` | 감싼 함수를 되돌리고 기록을 쓴다(`labels_%06d.npz`에 잡음 전·실행 행동) |
| `generate_eef_subtask_trajectory`(Dex) / `generate_trajectory`(Soft) | `src_subtask_object_pose = ...`(Dex) 또는 `src_subtask_nodal_positions = ...`(Soft) 뒤 | `src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())` | 안내 혼합(없으면 입력을 그대로 돌려준다) |
| 같은 함수 | `transformed_seq = WaypointSequence.from_poses(...)` 뒤 | `_tmg.record_subtask(locals())` | 하위작업의 원본 번호와 자세를 기록한다 |
| `MultiWaypoint.execute` | 첫 `play_action = env.target_eef_pose_to_action(...)` 뒤 | `_tmg.record_command(self, env, env_id, state, play_action)` | 목표 자세와 명령을 기록한다 |
| `MultiWaypoint.execute` | `success = ...` 뒤 | `_tmg.record_step(self, env, env_id, obs, success)` | 측정 자세와 성공 여부로 걸음 기록을 완성한다 |

잡음 전 라벨은 시도 동안 `env.target_eef_pose_to_action`을 감싸 얻는다. 감싼 함수는 먼저 토치 난수 상태를 저장한 채 잡음
없는 행동을 계산해 라벨로 두고, 난수 상태를 되돌린 뒤 공식 함수를 잡음과 함께 부른다. 그래서 공식 난수 열은 바뀌지 않는다.
공식 기록기는 실행된 행동만 hdf5에 쓰므로 라벨은 우리 기록(`labels_%06d.npz`)에 있고, `tmg dataset`은 이 생성기들에서
`actions_pre_noise`를 거기서 읽는다(`demo_layout()`의 `labels_from_records`).

추종(`method.tracking.enabled`)이 켜지면 공식 반복문을 그대로 쓸 수 없다. 공식 반복문은 말단마다 한 걸음에 경유점 하나씩
따로 전진시키는데, 추종기는 두 손을 경로 변수 하나로 움직여야 하기 때문이다. 그래서 `make_tracking_generate`가
시도 동안 `DataGenerator.generate`를 바꿔 끼운다. 반복문 본체는 `tmg/generation/generators/isaaclab/tracking_loop.py`의 `TrackingLoop` 하나이고, 패키지마다 다른 단계(초기화, 궤적 생성기 호출, 집게·통과 명령 읽기, 경유점 만들기, `MultiWaypoint.execute` 호출, 결과 수집)는 Soft의 `IsaacLabTrackingLoop`와 Dex의 `AutoDataTrackingLoop`가 훅으로 채운다. 바뀐 함수도 공식 궤적 생성기(`generate_*trajectory`,
`merge_eef_subtask_trajectory`), `MultiWaypoint.execute`, 공식 기록기를 그대로 부르고, 두 말단이 모두 궤적을 가진 원본
시간 구간마다 `ErrorAdaptiveFollower` 하나를 만든다(`tracking_loop.cut_interval`이 구간을 자른다). 오차는 설정된 링크
(`generator.measure_links`, Dex는 `*_hand_pitch_link`)의 자세로 재고, 집게 사건은 두 손의 명령 사건을 합쳐 쓰며, 공식
걸음별 잡음 뽑기는 `tracking_loop.NoiseLedger`가 보존한다(원본 경유점마다 공식 뽑기 한 묶음, 추가 걸음은 사설 난수). `method=none`은
항상 공식 `generate`를 돈다.

SoftMimicGen의 Rope 성공 판정(`rope_ends_close_tracked`)은 호출 시점에 민감하다. 이 함수는 에피소드마다 밧줄의 두 끝 마디를
"가장 멀리 떨어진 두 마디"로 처음 호출될 때 정하고(`env._rope_end_indices`), 그 뒤로는 그 마디 번호를 계속 쓴다. 리셋 사건
`reset_rope_end_tracking`이 리셋 때 이 캐시를 지운다. 공식 생성기와 공식 평가기는 모두 **첫 행동을 실행한 뒤** 처음으로 성공
판정을 부르는데, 리셋 직후에 부르면 밧줄이 아직 자리 잡기 전이라 다른 마디(예를 들어 0번·11번 대신 1번·10번)가 끝으로 잡히고
같은 공식 수식이 다른 성공 결과를 낸다(2026-10-27 Codex 감사). tmg가 지키는 규칙은 다음과 같다. 생성에서는 공식
`MultiWaypoint.execute` 안의 `success` 값을 읽기만 하고 성공 판정을 따로 부르지 않는다(추종 반복문도 공식 `execute`를 그대로 쓴다).
Isaac Lab 평가를 만들 때도 성공 판정은 첫 행동 뒤부터 매 걸음 부르고, 리셋 직후에는 부르지 않으며, 리셋 사건이 캐시를 지우도록
그대로 둔다. 이 규칙은 `configs/experiment/generator/softmimicgen/task/rope.yaml`의 `success_term`에 적혀 있다.

소스 해시는 `configs/experiment/generator/{softmimicgen,dexmimicgen}.yaml`의 `graft_hashes`에 있다. 아직 `null`인 항목은
첫 청크 실행이 설치된 해시를 로그에 찍으니, 그 값을 적어 고정한다. 고정 뒤에는 패키지가 바뀌면 청크가 시작하지 않는다.

### 2.1 공식 생성 설정값과 원본 시연 (2026-10-28 조사, 2026-10-30 갱신)

두 생성기에서 원본 시연 선택 방식, 행동 잡음, 보간 걸음 수는 공식 설정 파일의 값이 그대로 적용된다.
`configs/experiment/generation/*.yaml`의 `protocol` 블록에서 이 생성기들이 읽는 값은 `guarantee`와 `camera_names`(AutoData 어댑터만)
둘뿐이다. 그러나 어댑터가 공식 스크립트에 넘기거나 바꾸는 것은 시도 수, 시드, 입출력 파일보다 많다. 그 밖에 바꾸는 것은 다음과 같다
(`tmg/generation/generators/isaaclab/adapter.py`, `tmg/generation/generators/isaaclab/entrypoint.py`,
`tmg/generation/generators/autodata/adapter.py`, `tmg/generation/generators/autodata/runtime.py`).

- 성공 수 보장. 생성 설정의 `protocol.guarantee`로 덮어쓴다. SoftMimicGen은 환경 설정의 `generation_guarantee`를, AutoData는 과제 기술
  파일 사본의 `guarantee_success`를 바꾼다. 본 풀(`pool_isaac`)은 true라서 Soft 과제와 Pick Place에서는 공식 값과 같고, Microwave와
  Bottle Placement and Door Closing에서는 공식 기술 파일의 false와 달리 성공 수를 보장한다. 시험용 설정과 순위 풀은 false다.
- 장면 범위. 진입점이 공식 스크립트의 `setup_env_config`를 감싸 장면 파일의 범위를 환경 설정에 써 넣는다. `scene=official`이면 범위를
  바꾸지 않는다.
- 저차원 관측 항목. 진입점이 과제 설정의 `lowdim_terms`를 `policy` 관측 그룹에 더한다(2.10절). 공식 환경에는 없는 항목이다.
- 성공 판정 준비와 초기화. AutoData 런타임이 Arena의 클래스형 성공 판정을 생성 전에 한 번 인스턴스로 만들고, 시도마다 그 진행 상태를
  지운다(2.6절). 공식 스크립트 그대로는 생성자 인자 오류가 난다.
- 환경 선택. AutoData 어댑터는 `protocol.camera_names`가 카메라를 담으면 Pick Place를 우리 카메라 환경
  `TMG-PickPlace-GR1T2-Images-Mimic-v0`(2.9절)으로, 비어 있으면(순위 풀) 공식 환경 `Isaac-PickPlace-GR1T2-Abs-Mimic-v0`으로 돌린다.
  SoftMimicGen 과제는 공식 환경이 카메라를 이미 기록하므로 카메라 환경이 없다.
- 제어기 임시 폴더. 진입점이 역기구학 제어기의 URDF 출력 폴더를 청크마다 따로 둔다. 병렬 청크가 `/tmp/urdf`를 함께 쓰던 문제를 막기
  위해서다.

아래 값은 공식 소스(SoftMimicGen `source/softmimicgen/softmimicgen/envs/*_mimic_env_cfg.py` @ c9d146ba, AutoData
`autodata_examples/tasks/gr1_pick_place.yaml`, `gr1_open_microwave.yaml`, `gr1_put_and_close_door.yaml` @ 835f4aab)에서 읽었다. 2026-10-28
조사 때 이 표의 Dex 줄은 Isaac Lab 2.x(@3c6e67bb)의 환경 설정에서 읽은 Pick Place, Nut Pour, Exhaust Pipe였다. Nut Pour와 Exhaust Pipe는
원본 파일이 없어 계획에서 빠졌고(아래), 2026-10-29부터 Dex 과제는 AutoData로 도는 세 과제다(2.5절, 2.9절). Pick Place의 값은 두
소스에서 같다.

| 과제 | 한 시도에 원본 하나 사용 (select_src_per_subtask) | 원본 선택 방식 (selection_strategy, nn_k 3) | 행동 잡음 (action_noise) | 하위작업별 보간 걸음 수 | 실패 시도 기록 (keep_failed) |
|---|---|---|---|---|---|
| Soft Rope (밧줄을 U자로) | 그렇다 (false) | 정합 비용이 가장 낮은 3개 중 선택 (registration_cost) | 0.0 | 10, 5 | 기록한다 |
| Soft Towel (수건 접기) | 그렇다 (false) | 같음 | 0.01 | 5, 5 | 기록한다 |
| Soft Jenga (블록을 탑에 놓기) | 아니다 (true, 하위작업마다 고른다) | 잡기는 registration_cost, 놓기는 현재 물체 자세에 가장 가까운 3개 중 선택 (nearest_neighbor_object) | 0.01 | 5, 5 | 기록한다 |
| Dex GR1 Pick Place (AutoData) | 그렇다 (false) | nearest_neighbor_object | 0.003 | 오른손 0, 3 / 왼손 0 | 버린다 (false) |
| Dex GR1 Microwave Opening (AutoData) | 아니다 (true, 하위작업마다 고른다) | nearest_neighbor_object | 0.005 | 오른손 5, 5 / 왼손 0 | 버린다 (기술 파일에 없어 AutoData 기본값 false) |
| Dex GR1 Bottle Placement and Door Closing (AutoData) | 아니다 (true, 하위작업마다 고른다) | nearest_neighbor_object | 0.003 | 오른손 5, 5, 5, 5 / 왼손 0, 0 | 버린다 (기술 파일에 없어 AutoData 기본값 false) |

공식 성공 수 보장은 Soft 세 과제(`generation_guarantee`)와 Pick Place(`guarantee_success: true`)에서 켜져 있고, Microwave와 Bottle
Placement and Door Closing에서는 꺼져 있다(`guarantee_success: false`, 공식 `num_trials`는 50과 100). 우리 `generation=pool_isaac`(성공
250개 청크 4개, `protocol.guarantee: true`)은 모든 과제에서 `--generation_num_trials`가 성공 수를 뜻하게 하므로, 뒤의 두 과제에서는
공식과 다르다. 기록기는 실패 시도까지 모두 남기므로 생성 성공률은 기록에서 계산한다. 시험용 설정과 순위 풀(`rank_human_demos`)은
`guarantee: false`여서 어댑터가 보장을 꺼 적힌 수만큼만 시도한다(2026-10-29부터).

원본 시연 파일의 공개 여부는 다음과 같다. 측정한 사실이다.

- SoftMimicGen은 Hugging Face 저장소 `masoudmoghani/SoftMimicGen`의 `annotated_dataset/` 아래에 Rope, Towel, Jenga, Stack(soft cube),
  humanoid teddy/towel, surgical threading/tissue, yam bag/towel의 주석 달린 원본 파일을 공개한다. Rope는 원본 3개이고, 나머지 개수는
  내려받을 때 `tmg sources`가 매니페스트에 적는다.
- Dex GR1 Pick Place의 주석 달린 원본은 Isaac 자산 서버
  `Assets/Isaac/5.1/Isaac/IsaacLab/Mimic/pick_place_datasets/dataset_annotated_gr1.hdf5`(원본 5개)에 있다.
- Microwave Opening과 Bottle Placement and Door Closing의 주석 달린 원본은 Hugging Face의 Arena 데이터셋에 있다(2.5절 표).
- (2026-10-28 기록, 지금 계획에 없는 과제) Dex GR1 Nut Pour와 Exhaust Pipe의 주석 달린 원본 파일은 2026-10-28 기준으로 찾지 못했다. Isaac 자산 서버의 `Mimic/` 아래 전체
  목록에는 두 과제의 USD 자산 폴더(`nut_pour_task/`, `exhaust_pipe_task/`)만 있고 hdf5가 없으며, Isaac Lab 문서(고정 커밋과 main)도
  두 과제의 데이터 파일을 안내하지 않는다. 따라서 두 과제는 원본을 직접 녹화하고 주석을 달지 않으면 돌릴 수 없다. 이 저장소에는
  두 과제의 설정 파일을 두지 않았다.

이 조사로 고친 설정이 둘 있다. Rope 과제 파일의 카메라 이름과 로봇 상태 키를 생성 파일의 실제 관측 키(`agentview_image`,
`robot0_eye_in_hand_image`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`)로 바꿨고, Pick Place의 원본 주소를 위 경로로
바꿨다.

### 2.1.0 공식 정책 설정과 우리 적용 (2026-10-29)

과제별 공식 정책 설정 파일에서 읽은 값과 우리 적용은 다음과 같다. 정책 구조와 학습 예산은 생성기 전체가 공유하는 우리 설정을 쓰고,
영상 자르기와 평가 상한은 과제별로 둔다(`task.policy_overrides`, `task.horizon`). 2026-10-29 결정으로 Isaac Lab 세 과제는 모두
자르지 않고(Soft 공식 Diffusion Policy와 같다), MimicGen은 공식 76×76을 유지한다. 평가 상한은 공식 값이 아니라 2026-10-26 결정
(SoftMimicGen 400, DexMimicGen 계열 600)을 쓴다. 2026-10-29에 공식 값으로 바꾼 커밋이 있었으나 결정에 어긋나 같은 날 되돌렸다.

| 과제 | 공식 BC-RNN (영상) | 공식 Diffusion Policy (영상) | 공식 평가 상한 | 우리 적용 |
|---|---|---|---|---|
| Soft Rope, Towel (`softmimicgen_tasks/<task>/config/agents/robomimic/*.json`) | 128×128 영상을 76×76으로 무작위 자르기, ResNet18과 SpatialSoftmax 32, 1,200에폭, 배치 16, 순서 10 | 자르기 없음, 관측 1단계, 행동 8단계, 예측 16단계, DDPM 100, EMA, U-Net [256, 512, 1024], 1,200에폭, 배치 16 | Rope 250걸음, Towel 220걸음 (성공 시 종료) | 두 정책 모두 자르기 없음(2026-10-29 결정: Isaac Lab 과제는 한 규칙), 상한 400(결정값, 공식 250과 220이 아니다). Towel의 300×300 agentview는 128×128로 줄인다(공식과 다른 점, 2.6절). Diffusion Policy의 구조와 배치는 MimicGen 과제와 같은 우리 설정이라 공식과 다르다. 우리는 관측 2단계, U-Net [512, 1024, 2048], 확산 단계 임베딩(잡음 단계 번호를 바꾼 벡터의 차원) 128, 배치 64이고, 공식 SoftMimicGen은 관측 1단계, U-Net [256, 512, 1024], 확산 단계 임베딩 256, 배치 16이다(`franka_rope`, `franka_towel`의 robomimic json) |
| Dex GR1 Pick Place | 영상 설정 없음 (`bc_rnn_low_dim.json`만 있다) | 없음 | 환경 자체 길이 20초 = 400걸음 | 두 정책 모두 자르기 없음. 상한 600(결정값, 환경 자체 길이 400이 아니다). (참고로 Isaac Lab의 다른 GR1 영상 설정은 160×256을 144×236으로, 변의 90%로 자른다) |
| MimicGen 7과제 | 84×84를 76×76 | 같음 | 과제별 | 76×76 (기본값) |

### 2.1.1 컨테이너에서 Isaac Sim이 GPU를 잡기까지 (2026-10-28, Isaac 서버)

처음 빌드한 Isaac Lab 이미지에서 Isaac Sim은 `vkCreateInstance failed ... ERROR_INCOMPATIBLE_DRIVER`로 GPU 장치를 만들지 못했다.
공식 Isaac Lab 이미지(`nvcr.io/nvidia/isaac-lab:3.0.0-rc1`)는 같은 호스트에서 GPU를 잡았으므로 호스트 문제는 아니었다. 라이브러리
적재 추적(`LD_DEBUG=libs vulkaninfo`)으로 두 컨테이너를 비교하니, NVIDIA Vulkan 드라이버(`libGLX_nvidia.so.0`)는 초기화 과정에서
`libEGL.so.1`, `libEGL_nvidia.so.0`, `libgbm.so.1`, `libdrm.so.2`, `libnvidia-egl-gbm.so.1`을 차례로 열고, 우리 이미지에는 `libegl1`,
`libgbm1`, `libdrm2` 패키지가 없어서 그 자리에서 조용히 실패했다. 세 패키지를 넣자 `vulkaninfo`가 GPU를 보였다. 함께 확인한 사실은
다음과 같다.

- `docker run --gpus all`은 드라이버 라이브러리는 넣어 주지만 Vulkan 드라이버 파일 `/etc/vulkan/icd.d/nvidia_icd.json`을 넣지 않았다.
  `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`로 돌리면 들어온다. Makefile의 Isaac Lab 이미지
  블록이 이 방식으로 돈다.
- Kit은 자기 Vulkan 로더(`omni.gpu_foundation/bin/deps/libvulkan.so.1`)를 쓰므로 시스템 로더 버전은 원인이 아니었다(새 로더를 빌드해
  시험해도 같은 오류였다). CUDA 호환 라이브러리(`cuda-compat`)도 원인이 아니었다.
- pip으로 설치한 Isaac Sim은 패키지 폴더 아래 `kit/cache`, `kit/data`, `kit/logs`에 쓰므로, 호스트 사용자 권한으로 도는 컨테이너에서는
  그 폴더를 쓰기 가능하게 만들어야 한다(Dockerfile의 `chmod`).
- 공식 생성 스크립트는 `--seed`를 받지 않는다. 청크 시드는 `env_cfg.datagen_config.seed`에 써 넣고, 스크립트가 그 값으로 random,
  numpy, torch를 초기화한다.
- SoftMimicGen의 USD 자산(밧줄, 수건, 젠가, 로봇, 약 220 MB)은 Hugging Face 저장소의 `data/` 폴더에 있고 `source/softmimicgen_assets/data`에
  있어야 한다. 이미지 빌드 때 내려받아 넣는다.
- Pillow 버전 충돌. SoftMimicGen의 Isaac Lab 포크(`source/isaaclab/setup.py`)는 `pillow==11.2.1`을 고정하고, Isaac Sim 5.1
  (`isaacsim-kernel`)은 `Pillow==11.3.0`을 고정한다. 공식 Isaac Lab 3c6e67bb(Dex 이미지)도 11.3.0이므로 포크의 값은 옛 Isaac Lab 배포판의
  고정값일 뿐이다. 두 이미지 모두 시뮬레이터의 값 11.3.0을 쓴다. Isaac Sim이 함께 고정하는 packaging 23.0, click 8.1.7, idna 3.10, psutil 5.9.8은
  되돌리지 않는다. 그 옛 버전은 원본 내려받기에 쓰는 huggingface_hub와 httpx의 요구와 충돌하고, 새 버전으로 생성 시범 실행이 정상
  동작했기 때문이다. 그래서 `pip check`에는 Isaac Sim의 그 네 줄과 Soft 포크의 pillow 줄이 남지만 동작에는 영향이 없다.

### 2.2 Isaac Lab 과제의 정책 평가 (`tmg eval`, 2026-10-28)

평가기는 `tmg/evaluation/isaaclab_rollout.py`에 있고 생성기 이미지 안에서 GPU로 돈다. 환경은 Isaac Lab의 공식 평가 스크립트
`scripts/imitation_learning/robomimic/play.py`와 같은 방식으로 만든다. 과제 번호(`task.camera_env`가 있으면 그것, 없으면
`task.env_name`)로 환경 설정을 읽고, 관측 항목을 사전 형태로 두고, 시간 초과 종료와 기록기를 떼고, 성공 판정 항목을 설정에서 꺼내
직접 부른다. 성공 판정은 매 걸음 `env.step` 뒤에만 부르고 리셋 직후에는 부르지 않는다(Rope 규칙, 2절).

고정 장면은 저장된 상태가 아니라 시드다. `tmg resets`는 `<paths.resets>/<생성기>/<과제>/<장면>/val25_test300_seed202600.json`에
장면 번호마다 `seed = SeedSequence([evaluation.seed, index])`를 적고, 롤아웃은 `env.reset()` 직전에 그 값으로 `env.seed`, 토치, 넘파이,
`random`을 모두 초기화한다. 그래서 같은 `evaluation.seed`(202600)이면 어느 실행, 어느 정책, 어느 조건에서도 장면 번호 i의 배치는
같다. 물리 결과까지 장치에 무관하게 같다고는 보장하지 않는다. 에피소드 기록(`episodes.jsonl`)에는 성공 여부, 걸음 수, 장면 시드,
정책 시드와 함께 리셋 직후의 물체 자세(강체는 위치와 사원수, 변형체는 마디 평균 위치)를 남겨 뒤에 장면별 분석에 쓴다.

### 2.3 컨테이너 전체 경로 시범 실행 결과 (2026-10-28, Isaac 서버)

Rope를 `tmg/isaaclab-soft` 이미지로 전체 경로를 한 번 돌렸다. 측정한 사실이다.

- 생성(`generation=test/pilot20`, 성공 20개 보장): 26회 시도, 성공 20회, 834초(Isaac Sim 시작 포함, 시도당 평균 23.8초). 원본 3개가 모두
  선택되었다. 기록(`attempts.jsonl` 26줄, 라벨 npz)과 공식 `generated.hdf5`(성공 20개)가 맞았다.
- 고정 장면 시드 파일(검증 25, 시험 300)과 영상 데이터셋(시연 10개, 1,710걸음, 128×128 두 카메라)을 썼다.
- Diffusion Policy 200갱신 학습 뒤 `make validate`가 새 평가기로 검증 25장면을 모두 돌렸다. 오류 0, 모두 400걸음까지 실패(기대한 결과,
  200갱신 모형), 에피소드당 평균 89초, 리셋 직후 밧줄 마디 위치가 에피소드 기록에 남았다. `validation.jsonl`에 행이 추가되었다.

Dex Pick Place도 같은 경로를 `tmg/isaaclab-dex` 이미지로 끝까지 돌렸다(2026-10-28). 생성 50회 시도에 성공 20회(40%, 1,940초, 시도당
39초), 고정 장면과 데이터셋(시연 10개, 1,329걸음, 두 카메라 128×128), 200갱신 학습, 검증 25장면(오류 0, 에피소드당 약 2.9분,
초기 물체 위치가 기록됨)까지 정상이었다. 생성 성공률 40%는 이전 Codex 실행의 34%(2,979회 중 1,000회)와 같은 수준이고 Isaac Lab 문서의
65에서 80%보다 낮다. Towel과 Jenga는 생성 경로만 돌렸고 각각 24회와 약 20회 시도에서 성공 20개를 얻었다.

## 2.5 AutoData (isaac-sim/AutoData @ 835f4aab): DexMimicGen 세 과제의 공통 실행 환경 (2026-10-29)

논문 계획(`method_evaluation.tex`)의 DexMimicGen 과제는 Pick-and-Place, Microwave Opening, Bottle Placement and Door Closing이고,
뒤의 둘은 NVIDIA AutoData가 Isaac Lab-Arena 위에서 제공하는 GR1 과제다. AutoData는 Isaac Lab Mimic의 생성 알고리즘을 자기
패키지(`autodata_core`)로 다시 쓴 구현이라 `isaaclab_mimic`에 의존하지 않고, 스택도 다르다(Isaac Sim 6.1, Isaac Lab 3.0 @ 28a386f0,
Arena @ 481f5ae5, Python 3.12, torch 2.11). 이미지는 `tmg/isaaclab-dex`(`docker/isaaclab-dex.Dockerfile`, Arena의 uv 잠금 파일로 설치),
어댑터는 `tmg/generation/generators/autodata/`, 생성기 이름은 `dexmimicgen`, 과제는 `pick_place`, `microwave`, `bottle_door`다.
2026-10-29 결정으로 Pick Place도 이 환경에서 돈다(2.9절). 그 전에는 Pick Place가 Isaac Lab 2.x(3c6e67bb, Isaac Sim 5.1)의
`isaaclab_mimic`으로 별도 이미지에서 돌았고, 두 Arena 과제는 `autodata`라는 별도 생성기 이름을 썼다.

| 과제 | Arena 환경 이름 | 원본 시연 | 공식 생성 설정 (`autodata_examples/tasks/*.yaml`) | 에피소드 길이 |
|---|---|---|---|---|
| Microwave Opening (오른손으로 전자레인지 문 열기) | `gr1_open_microwave` | Hugging Face `nvidia/Arena-GR1-Manipulation-Task`, 파일 `arena_gr1_manipulation_dataset_annotated.hdf5`, 리비전 `arena_v0.2_lab_v3.0`, 주석 달린 시연 10개 | 하위작업마다 원본 선택(select_src_per_subtask true), nearest_neighbor_object(nn_k 3), 행동 잡음 0.005, 보간 오른팔 5 왼팔 0, 첫 하위작업 종료 오프셋 [10, 20] | 5초 = 250걸음 (50 Hz) |
| Bottle Placement and Door Closing (병을 냉장고에 넣고 문 닫기) | `put_item_in_fridge_and_close_door` | Hugging Face `nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task`, 파일 `ranch_bottle_into_fridge/ranch_bottle_into_fridge_annotated.hdf5`, 같은 리비전, 주석 달린 시연 5개(카드에는 10개로 적혀 있다) | 하위작업마다 원본 선택, nearest_neighbor_object(nn_k 3), 행동 잡음 0.003, 보간 오른팔 5 왼팔 0, 종료 오프셋 [0, 0] | 10초 = 500걸음 |

어댑터는 청크마다 공식 기술 파일의 사본에 세 값을 바꿔 쓴다. `generation_policy.seed`(청크 시드), `num_trials`(청크의 수),
`guarantee_success`(생성 설정의 `protocol.guarantee`. 본 풀은 true라 수가 성공 수를 뜻하고, 시험용 설정과 순위 풀은 false라 시도 수를
뜻한다)다. 두 Arena 과제의 공식 기술 파일은 `guarantee_success`가 false이고 `num_trials`가 50과 100이므로, 본 풀에서는 공식과 달리 성공
수를 보장한다. 어댑터는 기술 파일 밖에서도 장면 범위, 저차원 관측 항목, 성공 판정 준비와 초기화, 카메라에 따른 환경 선택을 바꾼다
(2.1절 목록).

실행 방식은 AutoData의 자체 Arena 시험과 같다. `scripts/generate_dataset.py --external_callback
isaaclab_arena.environments.isaaclab_interop.environment_registration_callback --env_name <환경> --alg dexmimicgen
--task_descriptor <청크 사본> --embodiment autodata_examples/embodiments/gr1_ik_abs.yaml --enable_cameras`이고, `--` 뒤의 `--seed`는
Arena에 전달된다. 기록 그라프트는 AutoData의 자리(`initial_state` 대입 뒤, 두 `return GenerationResult` 앞, `seq = WaypointSequence.from_poses`
뒤, `play_action = datastream.target_eef_pose_to_action` 뒤, `execute`의 마지막 반환값)에 들어가며, 잡음 전 라벨은
`datastream.target_eef_pose_to_action`을 감싸 얻는다. 혼합과 추종은 AutoData에 아직 연결하지 않았다(`method=none`만 된다).

관측은 Arena GR1 임바디먼트의 `left_eef_pos`, `left_eef_quat`, `right_eef_pos`, `right_eef_quat`, `hand_joint_state`(Isaac Lab Pick Place와
같은 이름)와 머리 카메라 `robot_pov_cam_rgb`(512×512, 관측 그룹 `camera_obs`)다. 데이터셋 단계와 평가기가 같은 함수로 128×128로
줄인다(`tmg/data/images.py`). 평가기는 Arena 콜백으로 환경을 등록한 뒤 AutoData처럼 종료 조건을 모두 떼고 성공 판정 항목만 꺼내
쓰며, Arena의 성공 판정은 클래스(`TaskSuccessTerm`)라 한 번 인스턴스로 만들어 매 걸음 부른다. 행동은 36차원(양손 자세 14 + 손가락 22)이다.

이 절의 내용은 코드와 설정을 기준으로 적었고, Isaac 서버의 시범 실행 결과는 2.6절에 덧붙인다.

## 2.6 AutoData 시범 실행 결과 (2026-10-28, Isaac 서버)

전자레인지 과제를 `tmg/isaaclab-autodata` 이미지로 돌려 세 가지 공식 쪽 문제를 찾아 어댑터에서 처리했다. 측정한 사실이다.

- AutoData 스크립트는 Arena의 클래스형 성공 판정(`TaskSuccessTerm`)을 함수처럼 바로 불러 생성자 인자 오류가 난다. Isaac Lab 3.0의
  `initialize_success_term`으로 생성 전에 인스턴스를 만들고, 시도마다 `reset_success_term`으로 환경의 진행 상태를 지운다.
- Isaac Lab 3.0의 앱 실행기는 `--headless`와 `--enable_cameras` 플래그가 없다(`--viz none`과 생성자 인자 `enable_cameras`). 어댑터와 평가기는
  실행기가 아는 쪽을 고른다.
- Isaac Lab 기록기는 `policy` 관측 그룹만 저장하므로 Arena의 머리 카메라(`camera_obs` 그룹, 512×512)는 공식 `generated.hdf5`에 없다.
  기록 그라프트가 매 걸음 행동 직전에 카메라를 읽어 128×128로 줄여 두고, 성공한 시도의 영상을 `records/images_%06d.npz`로 쓴다.

그 뒤 생성은 20회 시도에 성공 20회(100%), 1,020초(Isaac Sim 시작 포함, 시도당 약 50초)였고, 고정 장면 시드 파일, 데이터셋(시연 10개,
984걸음, 128×128 한 카메라), 200갱신 학습까지 정상이었다. 평가기 검증도 Arena 환경에서 25장면을 오류 없이 돌았다(에피소드당 평균
68초, 25장면 39분). 200갱신 모형이 25장면 중 2장면에서 성공으로 판정되었는데(한 번은 21걸음 만에), 거의 학습되지 않은 정책이므로
문 손잡이를 우연히 밀었거나 성공 기준(열림 0.8)이 느슨한 경우로 보인다. 전체 규모 실행에서 성공률과 함께 다시 본다.

병 놓기·문 닫기 과제(`bottle_door`, Arena의 `put_item_in_fridge_and_close_door`)도 같은 이미지로 전체 경로를 통과했다. 원본은
`nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task`의 주석 달린 시연 5개이고, 생성은 34회 시도에 성공 20회(58.8%), 3,401초(시도당
약 100초)였다. 데이터셋은 시연 10개에 3,244걸음(에피소드당 약 324걸음)이고, 200갱신 학습 뒤 검증 25장면이 오류 없이 돌았다(상한 500걸음,
모두 실패해 끝까지 돌았고 에피소드당 평균 158초, 25장면 66분). 에피소드 기록의 초기 배치에는 병(`ranch_dressing_hope_robolab`)의 자세와
냉장고(`refrigerator`)의 자세와 관절값이 들어 있다.

두 과제의 장면 무작위화는 공식 환경 정의를 그대로 따르며, 그 범위를 생성 기록과 평가 기록에서 측정했다.

- 전자레인지 과제(`gr1_open_microwave_environment.py`)는 로봇(x −0.4 m), 전자레인지(x 0.4 m, y −0.00586 m, z 0.22773 m), 문의 시작
  열림 정도(0.2)를 모두 고정값으로 두고 무작위화 항목이 없다. 생성 20회 시도의 초기 자세가 모두 같았고, 검증과 시험 에피소드 24개의
  초기 배치도 모두 같았다. 따라서 이 과제의 검증 25장면과 시험 300장면은 초기 상태가 같고 정책 시드(에피소드마다 다르게 고정)와
  물리 시뮬레이션의 비결정성만 다르다. 장면 시드 파일은 다른 과제와 같은 형식으로 만들어지지만 이 과제에서는 초기 상태를 바꾸지 않는다.
- 병 놓기·문 닫기 과제(`gr1_put_and_close_door_environment.py`)는 병만 `RandomAroundSolution(x_half_m=0.03, y_half_m=0.01)`로 흔들고
  냉장고와 로봇은 고정이다. 측정한 병 위치의 폭은 생성 38회 시도에서 x 8.2 mm, y 3.6 mm, 평가 12에피소드에서도 같은 크기였다. 코드가
  선언한 상자(x ±30 mm, y ±10 mm)보다 좁은 것은 Arena의 배치기(`ObjectPlacer`)가 범위를 적용하는 방식 때문으로 보이며, 생성과 평가가
  같은 공식 재설정 경로를 쓰므로 두 쪽의 분포는 일치한다. 원인은 더 파지 않았다.

Towel의 공식 agentview 카메라는 300×300이고 손목 카메라는 128×128이다. 정책은 카메라 영상을 쌓아 쓰므로 데이터셋 단계에서 과제의
`image_size`(128×128)로 줄이며(`tmg/data/images.py`), 평가기도 같은 함수로 실시간 영상을 줄인다. 2026-10-28 MINI 전체 실행에서 이
크기 차이로 Towel 학습이 실패해 찾은 것이다.
이 축소는 공식과 다른 점이다. 공식 Diffusion Policy(robomimic 구현)는 카메라마다 인코더를 따로 두어 300×300을 그대로 넣고, LeRobot
구현은 카메라별 인코더 옵션을 켜도 모든 카메라 영상을 (배치, 관측 단계, 카메라, 채널, 높이, 너비) 한 텐서로 쌓으므로 크기가 같아야 한다.
두 카메라를 300×300으로 키우는 대안은 픽셀 수가 약 5.5배라 택하지 않았고, agentview를 128×128로 줄이는 쪽으로 2026-10-29에 정했다.

## 2.7 드라이버 MINI 전체 실행 결과 (2026-10-28, Isaac 서버)

`MINI=1 scripts/run_isaaclab_baseline.sh`(생성 20개 성공, 데이터셋 시연 10개, 200갱신, 검증과 시험 각 6장면, 시드 202601과 202602)로
Rope, Towel, Pick Place, Microwave 네 과제가 학습, 검증, 선택, 시험을 거쳐 결과 묶음(`reports/isaaclab_mini_20261028.tar.gz`,
`reports/isaaclab_mini_towel.tar.gz`)까지 만들어졌다. 묶음에는 과제마다 생성 시도 기록(`generation/attempts.jsonl`), 장면 시드 파일,
데이터셋 기록, 시드마다 학습 기록(`train_log.jsonl`), 검증 줄(`validation.jsonl`), 선택 결과(`SELECTED.json`), 검증과 시험의 에피소드별
기록(`eval/<split>_h<상한>/<체크포인트>/episodes.jsonl`)과 요약(`eval_summary.json`)이 들어 있는 것을 확인했다.

| 과제 | 생성 성공률 | 시드당 정책 단계 시간 (학습 200갱신 + 검증 6장면 + 시험 6장면) | 시험 성공 (200갱신 모형) |
|---|---|---|---|
| Rope | 20/26 | 약 18분 | 0/6, 0/6 |
| Towel | 20/24 | 약 21분 | 0/6, 0/6 |
| Pick Place | 20/50 | 약 34분 | 0/6, 0/6 |
| Microwave | 20/20 | 약 33분 | 0/6, 1/6 |
| Microwave (AutoData만 따로, 처음부터) | 20/20 | 약 35분 | 1/6, 0/6 |
| Bottle Placement and Door Closing (처음부터) | 20/38 | 약 41분 | 0/6, 0/6 |

200갱신 모형의 성공률은 파이프라인이 끝까지 도는지 보려는 값이라 뜻이 없다. 첫 MINI 실행에서 실패한 네 건은 모두 원인을 찾아 고쳤다.

표의 마지막 두 줄은 2026-10-28 22:04 UTC에 AutoData 두 과제만으로 드라이버를 처음부터(원본 내려받기부터 결과 묶음까지) 돌린 것이고,
실패 없이 끝났다(`reports/isaaclab_mini_autodata.tar.gz`). 생성 레인은 전자레인지 풀 941초, 병 놓기·문 닫기 풀 3,637초였고, 정책 레인은
그 사이 전자레인지 시드 두 개를 돌렸다. 병 놓기·문 닫기는 같은 청크 시드(9001)로 시범 실행에서는 34회 시도에 20개 성공, 드라이버 실행에서는
38회 시도에 20개 성공이 나왔다. AutoData 생성은 GPU 물리와 비동기 실행 때문에 시드를 고정해도 시도 수가 조금씩 달라진다는 뜻이며, 성공
수(`num_trials`)는 그대로 보장된다.

- Towel 학습 실패는 300×300 agentview 영상과 128×128 손목 영상의 크기 차이였다(2.6절). 데이터셋 단계의 크기 조절을 넣고 다시 돌려 통과했다.
- 병 놓기·문 닫기의 원본 단계가 `docker: invalid reference format`으로 실패한 것은 드라이버가 `AUTODATA_TASKS` 기본값을 단어로
  나누지 않아 생성기 이름이 빈 문자열로 들어간 버그였다. 고쳤다.
- Microwave 풀 청크 실패는 같은 풀 폴더에 시범 실행 프로세스를 동시에 돌려 청크 파일을 서로 덮어쓴 운영 실수였다. 풀 폴더 하나에
  생성 프로세스는 하나만 돌려야 한다(안내서 9절).
- Rope 시드 202601의 검증 줄은 앞선 시범 실행이 상한 400걸음으로 남긴 것을 그대로 썼다. `validation.jsonl`에 줄이 있는 체크포인트는
  다시 검증하지 않으므로, 상한을 바꾸면 그 줄과 `eval/val_h*` 폴더를 지워야 한다(안내서 9절). 새 기계에서는 생기지 않는다.

## 2.8 넓힌 장면 `ours`의 생성 성공률 (2026-10-29, Isaac 서버)

Rope, Towel, Microwave의 공식 초기 상태 분포가 좁아 생성이 거의 실패하지 않으므로(2.7절), `ours` 장면을 공식과 다르게 넓혔다
(결정 2026-10-29). 규칙은 MimicGen의 `ours`와 같다. 공식 범위의 중심을 두고 폭을 두 배로 하며, 공식이 0으로 둔 축은 좌우 대칭으로
연다. 로봇 관절의 시작 잡음은 공식값(표준편차 0.02 rad)을 유지해 바뀌는 것이 물체 분포만이 되게 했다. 세 과제의 성공 판정은 모두
물체의 위치와 방향에 무관하다(Rope는 두 끝의 거리, Towel은 주성분 분석으로 잰 펼친 면적의 비율, Microwave는 문 관절의 열림 정도).

| 과제 | 공식 | ours | 구현 |
|---|---|---|---|
| Rope 전체 이동 | x [0, 0.1] m | x [−0.05, 0.15], y [−0.1, 0.1], yaw ±30° | 재설정 이벤트 `reset_object_position`의 `pose_range` |
| Rope 절반 변형 | x [0, 0.1], y [0, 0.1], yaw ±30° | x [−0.05, 0.15], y [−0.05, 0.15], yaw ±60° | 같은 이벤트의 `partial_pose_range` (`apply_scene`이 중첩 매개변수를 지원하도록 확장) |
| Towel | x [0, 0.1], y ±0.05, yaw ±30° | x [−0.05, 0.15], y ±0.1, yaw ±60° | `reset_object_position`의 `pose_range` |
| Microwave 자세 | 고정 | x ±0.05, y ±0.1, yaw ±15° | 공식 환경에 재설정 이벤트가 없어 `isaaclab.envs.mdp.reset_root_state_uniform`을 추가(`add: true`) |
| Microwave 문 시작 열림 | 0.2 고정 | 0.1에서 0.3 균등 | `tmg.generation.simulators.isaaclab_events.set_joint_percentage_uniform` 추가. Arena의 고정값 재설정 뒤에 돈다 |

추가한 이벤트는 환경 설정의 기존 이벤트 뒤에 붙어 매 재설정에서 공식 이벤트 다음에 실행되며, 평가기(`tmg/evaluation/isaaclab_rollout.py`)
도 같은 `apply_scene`을 거치므로 검증과 시험 장면이 같은 분포에서 나온다. 전자레인지는 생성 기록에서 시도마다 자세가 달라지는 것을
확인했다(x 0.38에서 0.42 m, y −0.07에서 −0.01 m).

성공 50개를 채울 때까지 한 프로세스씩 돌린 결과다(`generation=test/pilot50`, 청크 시드 9501, 세 프로세스 동시 실행).

| 과제 | 공식 (성공 20개 실행) | ours (성공 50개 실행) | ours 시간 |
|---|---|---|---|
| Rope | 20/26 = 76.9% | 50/94 = 53.2% | 3,758초 (시도당 40초) |
| Towel | 20/24 = 83.3% | 50/65 = 76.9% | 3,367초 (시도당 52초) |
| Microwave | 20/20 = 100% | 50/66 = 75.8% | 2,508초 (시도당 38초) |

Rope가 가장 크게 떨어졌고 Towel은 거의 변하지 않았다. 공식 쪽 표본이 20개라 두 비율의 차이에는 표본 오차가 크다. 기존 `ours` 이름으로
공식 범위에서 만들어 두었던 Isaac 서버의 시범 풀(`pilot20`), 데이터셋, 장면 시드, 실행 폴더는
`<data_root>/<종류>_stale/<생성기>/<과제>/ours_officialranges_20261029`로 옮겨 두었다.

### 2.8.1 실패가 나오는 영역 (2026-10-29)

실패는 넓힌 범위 전체에 고르게 퍼지지 않고 특정 영역에 몰린다. Rope와 Towel은 공식 스크립트가 남기는 실패 기록(`generated_failed.hdf5`)의
초기 노드 위치에서 물체 중심, 주축 방향(주성분 분석), 로프의 양 끝 거리를 계산했고, 공식 범위 실행(성공 20개, 26회와 24회 시도)의 같은
값으로 공식 분포의 발자국을 잡았다. Microwave는 시도 기록의 초기 자세를 썼다.

| 과제 | 공식 발자국 안의 시도 | 공식 발자국 밖의 시도 | 실패가 몰리는 영역 |
|---|---|---|---|
| Rope | 7/11 성공 (64%) | 43/83 (52%) | 로프가 많이 굽은 시작 모양. 양 끝 거리 0.78–0.81 m(거의 곧음)에서 20/24 = 83%, 0.69–0.74 m에서 7/23 = 30%, 0.60–0.69 m에서 10/24 = 42%. 공식 범위의 양 끝 거리는 0.69–0.80 m라 0.69 m 아래는 넓힌 절반 변형(회전 ±60°, y 최대 +15 cm)이 만든 모양이다. 중심 y가 +9에서 +16 cm인 쪽도 9/24 = 38%로 낮다 |
| Towel | 7/7 (100%) | 43/58 (74%) | 중심 x가 0.59–0.64 m(로봇에서 가장 먼 4분위)에서 7/17 = 41%, 그 안쪽 세 구간은 94%, 94%, 76%. 회전이 공식 한계(±30°)를 넘는 ±32–45°에서 11/17 = 65%, 공식 범위 안의 회전에서는 31/34 = 91% |
| Microwave | 모든 범위의 안쪽 절반(x ±2.5 cm, y ±5 cm, 회전 ±7.5°)에서 7/7 (100%) | 43/59 (73%) | 회전 방향 한쪽. 회전 +7.5°에서 +15°에서 5/16 = 31%, 0°에서 +7.5°에서 9/13 = 69%, 반대쪽 −15°에서 0°에서는 36/37 = 97%. 실패 16회 중 15회가 + 방향 회전이다. y −10에서 −5 cm도 10/17 = 59%로 낮다. 문 시작 열림의 영향은 실패 시도의 관절값이 기록에 없어 보지 못했다 |

정리하면 Rope는 절반 변형을 두 배로 넓혀 생긴 심하게 굽은 모양이, Towel은 로봇에서 먼 x 끝과 공식 한계를 넘는 회전이, Microwave는 한쪽 회전
(+ 방향 7.5° 이상)이 실패의 대부분을 만든다. 범위를 줄인다면 Rope 절반 변형의 회전을 공식 ±30°로 되돌리고, Towel 회전을 ±30°로 되돌리고
x 상한을 +0.12 m로 낮추고, Microwave 회전을 −15°에서 +5°로 자르는 것이 측정값에 맞는 조정이다. 2026-10-29에 현재 범위를 그대로
두기로 정했다. ours 장면의 생성 성공률은 위 표의 Rope 53.2%, Towel 76.9%, Microwave 75.8%가 기준값이고, 어려운 영역을 뺀 범위로
바꾸려면 두 ours.yaml 파일의 해당 범위를 고친 뒤 pilot50을 다시 돌려 이 표를 갱신해야 한다(`generation=test/pilot50`).

## 2.9 Dex 세 과제를 한 실행 환경으로 합친 일 (2026-10-29)

사용자 결정으로 Pick Place를 AutoData 환경으로 옮겨 DexMimicGen 세 과제가 이미지 하나(`isaaclab-dex`), 생성기 하나(`dexmimicgen`)로
돈다. 확인한 사실과 바꾼 것은 다음과 같다.

- Isaac Lab-Arena에 딸린 Isaac Lab 3.0은 Pick Place 환경 `Isaac-PickPlace-GR1T2-Abs-Mimic-v0`를 그대로 등록하고(`isaaclab_mimic.envs`),
  AutoData는 공식 과제 기술 파일 `autodata_examples/tasks/gr1_pick_place.yaml`을 배포한다. 그 값(한 시도에 원본 하나, nearest_neighbor_object
  nn_k 3, 행동 잡음 0.003, 보간 0과 3, 성공 보장)은 Isaac Lab 2.x의 환경 설정 `pickplace_gr1t2_mimic_env_cfg.py`와 같다. 원본 데이터
  `dataset_annotated_gr1.hdf5`도 같은 파일을 쓴다(주석 필드 eef_pose, object_pose, target_eef_pose, subtask_term_signals, 행동 36차원).
- Isaac Lab 3.0의 Pick Place 과제 설정은 `isaaclab_tasks.contrib.pick_place`로 옮겨졌고, 원격 조종용 머리 카메라 `robot_pov_cam`(450×720)이
  정책 관측 그룹에 들어 있다. 우리 카메라 환경(`tmg/generation/simulators/isaaclab_tasks/gr1_pickplace_cameras.py`)은 그 카메라를 빼고
  앞·옆 128×128 카메라 두 대를 `camera_obs` 관측 그룹에 넣는다. Isaac Lab 기록기는 `policy` 그룹만 저장하므로 생성 파일에 영상이
  들어가지 않고, Arena 과제처럼 런타임이 매 걸음 영상을 읽어 `records/images_%06d.npz`에 쓴다. 카메라 위치와 해상도는 2026-10 재현
  때의 값 그대로다.
- 생성기 설정 `dexmimicgen.yaml`은 AutoData 스택의 값(패키지, 커밋, 스크립트, 임바디먼트 파일)을 갖고, Arena 콜백은 과제 파일의
  `external_callback`으로 내려갔다(Pick Place는 null이고 대신 `register_modules: [isaaclab_mimic.envs]`와 카메라 모듈을 들여온다).
  평가기(`tmg/evaluation/isaaclab_rollout.py`)는 생성기의 패키지가 `autodata_core`이면 AutoData의 `setup_env_config`처럼 종료 조건을
  모두 떼고 Pink IK 행동 클래스를 공유 URDF 변환판으로 바꾼다. 이 처리가 이제 Pick Place에도 적용되므로, Isaac Lab 2.x에서 평가할 때
  살아 있던 "물체 낙하" 종료 조건은 더 이상 에피소드를 끊지 않는다(상한까지 돈다).
- 섞기(`mix_segment`)는 AutoData의 `generate_eef_subtask_trajectory`에서 `src_subtask_object_pose` 대입 뒤에 끼운다. 원본마다 구간
  시작의 말단 자세, 목표 자세, 그 손의 통과 명령(11개 관절)과 기준 물체 자세를 섞기 모듈에 넘기는 것은 Isaac Lab 런타임과 같다.
  추종은 `DataGenerator.generate`를 공유 경로 반복문으로 바꾼다(AutoData의 궤적 생성·병합 함수와 `MultiWaypoint.execute`, 기록기
  수명주기를 그대로 쓴다). 잡음 장부의 draw 모양은 AutoData 순서(왼손 위치 3, 왼손 사원수 4, 오른손 위치 3, 오른손 사원수 4)이고,
  잡음 크기가 0인 손은 draw를 하지 않으므로 장부의 묶음을 다 쓰지 않을 수 있다(공식 난수열과의 일치는 분포 수준이다).
- 추종이 돌 수 있는 조건은 한 시도의 모든 하위작업이 같은 원본을 쓰는 것이다. Pick Place는 공식 설정이 그렇다. Microwave와 Bottle은
  공식 기술 파일이 `select_src_per_subtask: true`라 추종을 켜면 첫 시도 전에 `NotImplementedError`로 멈춘다. 섞기는 세 과제 모두 된다.
- 손가락 열림·닫힘 사건의 열(`gripper_events.columns: [0, 1, 2, 3]`)은 AutoData 임바디먼트 파일 `gr1_ik_abs.yaml`의 손 명령 순서
  (근위 손가락 관절 index, middle, pinky, ring가 앞 네 개)에서 Isaac Lab 2.x 런타임과 같은 관절을 가리킨다.
- 추종 오차는 Pink IK가 목표로 삼는 `*_hand_pitch_link`에서 잰다. Arena의 GR1 Pink 임바디먼트는 Isaac Lab Pick Place의 `ActionsCfg`를
  그대로 쓰므로 세 과제의 목표 링크가 같다.
- 파드용 네이티브 실행(`scripts/native_runtime.py`)은 `isaaclab-dex`를 거부한다. 그 이름의 네이티브 환경은 옛 Isaac Lab 2.x 설치이고,
  AutoData 스택의 부트스트랩 단계는 아직 없다.
- Isaac 서버에서는 기존 `tmg/isaaclab-autodata:latest` 이미지에 `tmg/isaaclab-dex:latest` 태그를 붙였고(옛 Isaac Lab 2.x 이미지는
  `tmg/isaaclab-dex-lab2:latest`로 남겨 두었다), `sources`, `pools`, `datasets`, `runs`, `resets`의 `autodata/` 폴더를 `dexmimicgen/`으로
  옮겼다. Isaac Lab 2.x로 만든 Pick Place의 `pilot20` 풀과 실행 폴더는 그대로 있고, 새 환경의 확인 실행은 `pilot50`으로 했다.

확인 실행의 결과는 2.9.1절에 적는다.

### 2.9.1 통합 환경 확인 실행 결과와 Pick Place 생성 성공률 (2026-10-29, Isaac 서버)

통합 환경에서 Pick Place의 생성 성공률이 옛 Isaac Lab 2.x 환경보다 낮게 나왔고, 원인을 두 갈래로 조사했다. 결론은 "환경(물리 엔진과 에셋
버전) 차이로 보이며 생성 논리의 차이는 아니다"이고, 사용자 결정(2026-10-29)대로 통합 환경을 그대로 쓴다.

| 실행 | 환경 | 생성기 | 시도 | 성공 | 성공률 |
|---|---|---|---|---|---|
| 2026-10-28 pilot20 | Isaac Lab 2.x (Isaac Sim 5), 카메라 환경 | isaaclab_mimic 공식 datagen | 50 | 20 | 40.0% |
| 2026-10-29 pilot50 (서버 재부팅으로 49회에서 중단) | Isaac Lab 3.0 (Isaac Sim 6.1), 카메라 환경 | AutoData (dexmimicgen) | 49 | 11 | 22.4% |
| 2026-10-29 pilot50 다시 실행 | 같음 | 같음 | 50 | 12 | 24.0% |
| 2026-10-29 대조 실행 | Isaac Lab 3.0, 공식 환경(카메라 없음) | isaaclab_mimic 공식 datagen (3.0 이미지의 `generate_dataset.py`, 성공 20개 보장) | 64 | 20 | 31.2% |

세 성공률(40%, 31%, 23%)은 표본이 50에서 100회여서 서로 통계적으로 구분되지 않는다(이항 검정 p가 모두 0.2 이상). 확인한 사실은 다음과 같다.

- 로봇 설정(`GR1T2_HIGH_PD_CFG`의 구동기 강성 4400, 감쇠 40, 다리와 머리는 두 버전 모두 구동기 없음), 성공 판정(`task_done_pick_place`), 초기 배치
  이벤트, 과제 환경 설정의 물리 시간 간격(1/120 s, decimation 6)은 2.x(커밋 3c6e67bb)와 3.0(커밋 28a386f0)에서 같다. 3.0이 새로 찍는 경고
  "Actuator groups cover 39 joints; expected 54"는 두 버전에 공통인 상태를 3.0만 알리는 것이다. 3.0은 손가락 링크에 접촉 센서를 더했다.
- 동작 변환(목표 자세를 36차원 행동으로 바꾸고 잡음을 더하는 코드)은 isaaclab_mimic의 환경 메서드와 AutoData의 임바디먼트 어댑터가 같은 계산을 한다.
  AutoData는 사원수의 부호를 w가 양수가 되게 맞추는데(3.0의 isaaclab_mimic도 같다), 같은 회전이므로 결과가 바뀌지 않는다.
- 생성 알고리즘(원본 선택, 구간 변환, 보간, 양팔 동기, 실행과 성공 판정)을 두 구현에서 비교한 결과 이 과제의 설정에서 동등했다. 차이는 난수열의
  순서(같은 시드에서 다른 궤적), 손 관절 명령의 시간축(2.x는 행동의 14:25 열을 왼손, 25:36 열을 오른손으로 잘라 실제 손 배치(왼손 5, 오른손 5,
  왼손 5, 오른손 5, 왼손 1, 오른손 1)와 어긋나므로 오른손 근위 관절 다섯 개가 왼손 시간축을 따라갔고, AutoData는 올바른 배치를 쓴다), 2.x의
  잡음이 저장된 경로점에 누적되는 점이다. 셋 모두 AutoData 쪽에 불리하지 않다.
- 따라서 남는 설명은 Isaac Sim 5에서 6.1로 바뀐 물리 엔진과 에셋(GR1T2와 조향 핸들 USD는 Nucleus 자산 서버의 버전별 경로에서 받는다)이다.
  대조 실행(3.0 공식 생성기 31%)이 2.x(40%)보다 낮은 것이 이 방향과 맞지만, 표본 크기 때문에 확인된 사실이 아니라 추정이다.

통합 환경의 MINI 전체 실행(2026-10-29 07:46–10:22 UTC, 다른 작업 다섯 개와 GPU를 나눠 쓰는 상태)은 끝까지 오류 없이 돌았다. 생성 50회 시도에 성공 12개
(24.0%, 영상 포함), 시연 10개의 영상 데이터셋, Diffusion Policy 200갱신 학습, 검증 6장면(성공 0)으로 체크포인트 선택, 시험 6장면(성공 0/6, 상한 600)이다.
성공 0은 200갱신짜리 모형이라 예상된 값이고, 확인 대상은 경로가 끊기지 않는 것이었다. 결과 묶음은 Isaac 서버의 `reports/isaaclab_mini_dex_unified`에 있다.

해결하지 못한 점이 하나 남아 있다. AutoData의 자체 성능 시험(`autodata_tests/datagen_perf/test_dexmimicgen_gr1_data_generation_success_rate.py`
16행, `SUCCESS_RATE_THRESHOLD = 0.70`)은 Pick Place 생성 성공률이 70% 이상이어야 통과한다고 정하는데, 우리 통합 환경의 생성 성공률은
50회 중 12회(24%)였다. 그 시험의 조건은 환경 `IsaacContrib-PickPlace-GR1T2-Abs`, 병렬 환경 100개, 시도 500회, 장치 CPU로 우리 실행과
다르다. 위의 조사는 옛 Isaac Lab 2.x 환경과 비교했을 뿐 이 시험의 조건과 비교하지 않았고, 이 시험을 우리 이미지에서 돌리지도 않았다.
차이의 원인은 모른다(2026-10-30 기준).

## 2.10 Isaac Lab 과제의 저차원 관측 (2026-10-29 결정과 확인)

MimicGen과 CP-Gen의 저차원 정책은 robosuite의 `object` 관측(물체마다 위치 3, 회전 사원수 4, 관절이 있는 물체는 관절 각도, 말단에서 물체까지의
상대 벡터)으로 학습한다. Isaac Lab 다섯 과제에는 Pick Place를 빼고 그런 항목이 없어서 같은 수준의 항목을 넣었다(사용자 결정: 다섯 과제 모두 저차원 실험을
하고, Rope는 마디 16개, Towel은 6×6 격자). 구현은 `tmg/generation/simulators/isaaclab_lowdim.py`이고, 과제 설정의 `lowdim_terms`가 항목을 정의하며
`object_state`가 데이터셋에 넣을 항목 이름을 든다. 생성 진입점과 평가기가 환경을 만들기 전에 같은 항목을 `policy` 관측 그룹에 넣으므로 Isaac Lab
기록기가 생성 파일에 함께 저장하고, 평가는 실행 중인 환경에서 같은 값을 계산한다. 2026-10-29 전에 만든 풀에는 이 항목이 없다.

| 과제 | 항목 | 차원 | 내용 |
|---|---|---|---|
| Pick Place | `object` (공식 항목) | 13 | 물체 위치 3, 사원수 4, 왼손·오른손(`*_hand_roll_link`)에서 물체까지 3+3 |
| Rope | `rope_state` | 54 | 밧줄 휴지 모양의 주축을 따라 고른 마디 16개의 위치(48)와 말단에서 첫 마디·끝 마디까지의 벡터(6) |
| Towel | `towel_state` | 111 | 수건 휴지 평면의 두 주축에 6×6으로 고른 마디 36개의 위치(108)와 말단에서 수건 무게중심까지의 벡터(3) |
| Microwave | `microwave_state` | 14 | 전자레인지 자세 7, 문 관절 `microjoint` 1, 양손에서 전자레인지까지 6 |
| Bottle Placement and Door Closing | `bottle_door_state` | 27 | 병 자세 7과 양손 상대 벡터 6, 냉장고 자세 7과 문 관절 `fridge_door_joint` 1과 양손 상대 벡터 6 |

마디 선택은 물체의 휴지 모양(`default_nodal_state_w`, 생성 시점의 메시)에서 한 번 하고 환경 안에서 고정한다. 주축은 특잇값 분해로 구하고, 축의 가장 큰
성분이 양수가 되게 부호를 맞춰 어느 프로세스에서나 같은 순서가 나오게 한다. 마디는 축 위에 고르게 놓은 목표 좌표(양 끝 포함)에 가장 가까운 것을 고른다.
위치는 모두 환경 좌표계(세계 좌표에서 `env_origins`를 뺀 값)다.

확인(Isaac 서버, 과제마다 3회 시도 풀 `lowdim_check`): 네 항목이 모두 위 차원으로 기록되었고 값이 유한했다. Rope의 마디 16개는 전체 마디 구름의 범위를
덮고 첫 마디와 끝 마디 사이 거리가 0.80 m(밧줄 길이)이며 말단 상대 벡터가 좌표 차와 일치했다. Towel의 격자 범위는 전체 마디 구름의 범위와 같았다.
Microwave의 자세와 관절값(−0.164)은 시뮬레이터 상태의 전자레인지 자세와 첫 관절값과 같았고, Bottle의 병 자세와 냉장고 문 관절(0.785, 상태의 둘째 관절)도
상태값과 같았다. 생성 성공(3회 중 2회)은 항목 추가에 영향을 받지 않았다. Rope의 저차원 전체 경로(데이터셋, 200갱신 학습, 6장면 평가) 확인은 실행 중이다.

AutoData 어댑터는 `generation.protocol.camera_names`를 따른다(2026-10-29). 비어 있으면(순위 풀) 영상을 저장하지 않고 Pick Place도 카메라 환경 대신
공식 환경을 쓴다. MINI 실행용 `test/pilot50`은 영상 정책을 학습하므로 과제 카메라를 저장한다.

## 3. CP-Gen (kevin-thankyou-lin/cpgen@1e7c1fc4)

### 3.0 장면: CP-Gen은 자기 Wide 환경에서 돈다 (2026-10-29 결정)

CP-Gen 논문의 주 실험은 물체 크기가 리셋마다 바뀌는 "Wide" 환경이다. 이 환경은 CP-Gen이 다시 구현한 MimicGen 환경의 D1 배치 범위에
물체별 크기 배율을 더한 것이다. 예를 들어 Coffee는 포드와 머신의 크기를 0.6에서 1.1배로 바꾼다. Threading은 바늘과 삼각대 모두 x와 z
축을 0.8에서 1.2배로 바꾸고, y 축은 바늘을 1배로 고정하고 삼각대를 0.9에서 1.1배로 바꾼다(`cpgen_envs/environments/manipulation/threading.py`의
`ThreadingWide`). CP-Gen 논문은 Threading의 크기 범위를 0.9–1.1로 적어 코드의 범위와 다르고, 우리 생성은 코드의 범위로 돈다. 결정은 "CP-Gen은 CP-Gen 환경에서,
MimicGen은 MimicGen 환경에서 따로"다. CP-Gen의 공식 장면 `official_wide`는 `<과제>Wide` 클래스 그대로이고, 우리 장면 `ours`는 그 클래스를 부모로
우리 배치 범위(MimicGen의 `ours`와 같은 값)를 얹으므로 크기 변화가 함께 들어간다. MimicGen 쪽 환경(robosuite 1.4.1)에는 크기 조절 기능이 없고 두
생성기를 한 환경에 맞추지 않는다. 장면 파일은 `env_class`와 `base_env_class`에 `"{env_prefix}Wide"`처럼 밑줄 없는 클래스 이름 패턴을 적을 수 있다.
평가의 고정 장면(`tmg resets`)은 리셋마다 모델 xml을 저장하므로 뽑힌 크기도 그대로 복원된다. 2026-10-27의 Coffee 재현(3.y절)은 크기 변화가 없는
`<과제>_D0` 기반 장면에서 한 것이라 이 결정 이후의 수치와 직접 비교되지 않는다.

### 3.1 원본 열 개를 모두 쓰기: 시범별 제약 구간과 시도별 원본 선택 (2026-10-29)

공식 CP-Gen은 원본 시범 0번만 쓴다. 배포된 제약 주석(`constraints.json`)이 과제마다 제약 2–4개를 담는데, 그 걸음 구간(`timesteps`)이 시범 0번의
것이고 CP-Gen은 불러온 모든 시범에 이 파일을 그대로 적용하기 때문이다. 핵심점(물체 원점 기준 ±1 cm 네 점), 손가락 핵심점, 대칭, 부착 관계, 제약
뒤 동작은 과제의 성질이라 시범이 달라도 같고, 시범에 따라 다른 것은 걸음 구간 하나다. 원본 열 개를 모두 쓰기 위해(`selection=ours`) 두 가지를 넣었다.

- **시범별 걸음 구간.** `tmg sources`가 CP-Gen 과제의 원본 사본에 시범마다 `constraint_data` 그룹(JSON 문자열, CP-Gen의 `load_demos`가 가장 먼저
  읽는 형식)과 상호작용 순서 속성을 쓴다(`tmg/generation/generators/cpgen/windows.py`). 구간은 CP-Gen의 분절 규칙(두 주체의 최소 거리가 문턱값 아래로
  내려가면 시작, CP-Gen의 접촉 규칙으로 끝)으로 구하되, 문턱값은 상호작용마다 시범 0번의 공식 구간에 가장 가깝게 되도록 0.1 cm에서 8 cm 사이에서
  고르고 같은 값을 1–9번에 적용한다. 0번은 공식 주석을 그대로 둔다. 과제별 보정 결과(0번에서 공식 구간과의 걸음 차이): Coffee 1, 1, 12 / Mug Cleanup 1,
  1, 5, 1 / Hammer Cleanup 1, 2, 9, 2 / Three Piece Assembly 1, 1, 1, 1 / Threading 11, 44. Threading의 둘째 구간(바늘을 삼각대에 끼우기)은 접촉 수준
  문턱값에서도 공식보다 약 40걸음 일찍 시작한다(공식 주석은 "가까이 있음"이 아니라 "끼우는 중"을 구간으로 잡았다). 50개 시범의 구간을 영상으로
  검토하는 대시보드를 만들어 사용자가 확인했고(2026-10-29, "일단 그대로 쓰고 이상하면 다시 본다"), Hammer Cleanup 시범 1·2·7의 12걸음 미만 구간
  세 개가 검토 표시로 남아 있다. 구간을 고치면 그 과제의 원본 사본과 순위 풀을 다시 만든다.
  분절 때 시뮬레이터는 CP-Gen 자체의 `reset_to`(`demo_aug/generate.py`가 로보미믹의 `EnvRobosuite.reset_to`를 실행 중에 바꿔 끼우는 함수)와
  같은 순서로 다룬다. 시범의 저장 모델 xml을 먼저 시도하고, 설치된 robosuite가 그 xml을 읽지 못하면 새로 초기화한 모델을 두고 상태를 복원한다.
  배포 원본은 robosuite 1.4.1로 기록되어 CP-Gen 이미지의 1.5 포크가 그 xml을 읽지 못하므로(자산 폴더 이름 `mounts`가 `bases`로 바뀌었고,
  1.5가 요구하는 로봇 사이트가 없다) 모든 시범이 이 대체 경로를 탄다. 여기서 한 가지가 더 필요했다. 물체의 위치는 상태 벡터에 있지만 물체의
  치수는 모델에 있고, MimicGen은 일부 물체(Hammer Cleanup의 망치)를 초기화마다 무작위 치수로 만든다. 그래서 새 모델 위에 기록된 상태를 올리면
  망치 크기가 매번 달라져 구간을 정하는 몸체 간 거리가 초기화마다 바뀌었다(같은 과정에서 시범 0번의 둘째·셋째 구간 시작이 1걸음씩 흔들리고,
  보정된 문턱값이 0.04 m와 0.03 m 사이에서 바뀌었다). `windows.py`의 `reset_model`은 xml 읽기가 실패하면 저장 xml의 몸체·기하 배치와 치수를
  이름이 같은 항목에 복사해 넣는다(로봇·그리퍼·받침대 몸체는 제외, 메시 기하는 치수를 그대로 둔다). 이렇게 하면 분절이 결정적이고 장면이
  기록된 그대로가 된다. CP-Gen 본체는 원본을 불러올 때 이 복사를 하지 않지만, 상태와 제약 구간만 읽으므로 필요하지 않다. 내려받은 원본 파일은
  Hugging Face 캐시가 읽기 전용으로 두므로 내려받기 함수가 쓰기 권한을 준다. Mug Cleanup은 환경을 만들 때 MimicGen의 컵 메시 폴더에 크기를
  조정한 xml을 쓰므로, Makefile이 cuRobo 작업 설정과 같은 방식으로 그 폴더의 쓰기 가능한 호스트 사본을 이미지 위에 올린다(세 가지 모두
  2026-10-29에 서버에서 발견해 고쳤다).
- **시도별 원본 선택.** CP-Gen은 특정 모드(`use_reset_near_constraint`)에서만 제약 열(원본)을 고르고 그 밖에는 늘 0번을 쓰므로, 런타임이 시도 시작에서
  묶음 시드로 고정된 난수열로 원본을 무작위로 고른다(`CPGenRuntime.begin_attempt`가 부르는 `SourcePicker`, 난수열은 `tmg.generation.seeds.source_pick_rng`). 어댑터는 `selection=ours`일 때 원본을 전부 불러오고 공식 주석 파일을
  넘기지 않는다(시범 안의 제약 데이터가 쓰인다). 고른 원본 번호는 시도 기록(`src_demo_inds`)에 남아 `tmg rank_sources`가 다른 생성기와 같은 방식으로
  순위를 매긴다. 섞기는 CP-Gen에 대응물이 없으므로 여전히 끄고, 방법은 추종만 쓴다.

CP-Gen은 원본 구간을 다시 재생하지 않는다. 시도마다 주석된 키포인트 제약을 새 물체 자세에 맞춰 다시 풀고(제약 걸음),
그 시작점까지 cuRobo로 운동 계획을 세운다(계획 걸음). `CPPolicy.get_action`은 대기열이 비면 다음 제약을 풀어 계획 걸음과
제약 걸음을 `action_queue`에 채우고, 한 걸음에 하나씩 꺼낸다. `DemoGenerator.generate_demo`는 꺼낸 행동에 균등분포
잡음을 더해 환경에 넣고, 꺼낸 행동을 저장한다. 잡음의 크기는 `SystemNoiseConfig`의 `motion_segment_noise_magnitude`(계획 걸음)와
`constraint_segment_noise_magnitude`(제약 걸음)인데, 두 값의 기본값은 7차원 모두 0이고(`demo_aug/generate.py` 3397–3407행) 우리 설정도
저자 실행 스크립트도 이 값을 바꾸지 않는다. 그래서 CP-Gen 생성에서 실제로 더해지는 잡음은 0이다. 공식 코드는 이때도 걸음마다 균등
난수를 뽑아 0을 곱한다. 저장 행동과 실행 행동이 같으므로 CP-Gen의 저장 행동을 잡음 전 라벨로 써도 된다.

끼우는 함수는 `DemoGenerator.generate_demo` 하나다(`tmg/generation/generators/cpgen/runtime.py`).

| 앵커(대입되는 이름) | 끼우는 문장 | 하는 일 |
|---------------------|-------------|---------|
| 첫 `state = self.env.get_state()` 뒤 | `_tmg.begin_attempt(self)` | 시도 기록을 열고 장면 메타데이터를 읽는다 |
| `action_dct = self.policy.get_action(obs)` | `action_dct = _tmg.get_action(self, obs)`로 바꾼다 | 공식 `get_action`을 부르고, 새 구간이 계획되면 구간(계획=연결, 제약=기술)과 원본 번호를 기록한다. 추종이 켜지면 계획된 행동들을 경로 하나로 삼아 추종기가 다음 목표를 낸다 |
| 반복문 안 `state = self.env.get_state()` 뒤 | `_tmg.record_step(action, action_to_take, reward, in_constraint_segment, system_noise_cfg)` | 목표 자세, 측정 자세, 라벨, 실행 행동, 잡음 크기를 기록한다 |
| 마지막 `return episode_data` 앞 | `_tmg.end_attempt(self, episode_data)` | 기록을 쓴다 |

추종기가 움직일 때는 정책의 `action_queue`를 추종기의 기준 걸음 번호에 맞춰 잘라 두어, 공식 잡음 규칙
(`in_constraint_segment`)이 올바른 구간을 읽게 한다. 경로가 끝나면 대기열을 비우고 `check_constraint`를 켠다. 공식 코드가
마지막 행동을 꺼낸 뒤 남기는 상태와 같다.

생성기 설정(`configs/experiment/generator/cpgen.yaml`)은 모든 방법에서 혼합을 끄고 라벨을 `pre_noise`로 둔다. 혼합은 CP-Gen에
대응물이 없고, 라벨은 공식 출력이 이미 잡음 전이기 때문이다. 소스 해시는 고정되어 있고, 같은 본문이
`tests/unit/fixtures/cpgen_generate_demo.py`에 있어 시험이 가짜 환경과 가짜 계획기로 접목된 함수를 실제로 돌린다.

청크 출력은 CP-Gen의 `main`이 만든다. `generated_obs.hdf5`가 성공 시도들의 상태로부터 두 카메라(84x84)와 로봇·물체 상태를
다시 그린 파일이고, `tmg dataset`은 이것을 읽는다. k번째 데모는 k번째 성공 시도다. 성공이 하나도 없으면 공식 `main`이 그릴
파일이 없어 예외를 내는데, 모든 시도가 기록된 뒤라면 청크 프로세스는 이를 정상 종료로 처리한다.

### 3.2 저자 실행 스크립트와 맞춘 값과 남은 차이 (2026-10-30)

로봇 시작 자세 잡음은 저자 실행 스크립트의 값으로 맞췄다. 이 잡음은 리셋 때 로봇 관절의 시작 자세에 더하는 값이다. CP-Gen 생성 설정
클래스(`demo_aug.generate.Config`)의 기본값은 정규분포 0.02이지만, 저자 실행 스크립트(`scripts/pipeline/run_datagen_and_training.sh`
202–203행)는 정규분포 0.06을 준다. 2026-10-30 결정으로 우리 설정(`configs/experiment/generator/cpgen.yaml`의 `config.initialization`)도
정규분포 0.06을 쓴다.

CP-Gen 이미지에 설치된 robomimic은 0.4.0이다. CP-Gen의 `pyproject.toml`은 robomimic 커밋 9273f9c를 적고 주석에서 v0.3.0이라고 부른다.
이 차이가 생성이나 평가에 영향을 주는지는 확인하지 않았다.

평가 상한은 우리 결정값 600걸음(2026-10-27 결정)이다. CP-Gen 저자 실행 스크립트의 평가 상한(`task.env_runner.max_steps`, 368–395행)은
Coffee, Mug Cleanup, Threading이 450걸음이고 Hammer Cleanup과 Three Piece Assembly가 600걸음이다. 이 값들은 MimicGen의 평가
상한(400 또는 500)과도 다르다.

### 3.3 잡음 재생과 다시 그리기 (2026-10-30 결정)

학습 데이터셋은 저자 실행 스크립트의 `add-noise`와 `render-rgb` 두 단계를 거친 파일로 만든다. 이전 판은 `add-noise`를 빼고
생성 직후의 성공 시연으로 학습했는데, 이것은 잘못이었다. 저자가 공개한 학습 데이터셋 16개는 모두 이 두 단계의 출력이다. 파일
이름이 모두 `...-original-action-noisy-state-action-std-0.03-rgb-84-84.hdf5`이다. 이전 판이 근거로 삼은 스크립트 주석("뒤의
다른 프로젝트용")은 공개 데이터셋의 실제 이름과 맞지 않는다.

두 단계는 다음과 같다.

- `add-noise`는 공식 `scripts/playback_dataset_w_noise_mp.py`를 `--num_retries 6 --use-actions --num_processes 12
  --action_noise_std 0.03`으로 실행한다. 각 성공 시연을 처음 상태에서 행동 그대로 다시 재생하되, 행동마다 정규분포 잡음을 더한다.
  재생이 성공하면 그 재생에서 지나간 상태들을 저장하고, 행동은 잡음 없는 원래 행동을 저장한다.
- 잡음의 표준편차는 시도 k(0부터 셈)마다 0.03 × (6 − k) / 6이다. 제약 시점(물체를 잡거나 끼우는 구간의 걸음)에서 10걸음 안쪽이면
  0.05–0.6배로 더 줄인다.
- 성공은 재생의 마지막 걸음 뒤 과제 판정으로 정한다. 여섯 번 모두 실패한 시연은 원래 상태를 그대로 저장하고 데이터셋에 남는다.
  그래서 시연 수는 줄지 않는다.
- `render-rgb`는 그 상태들에서 두 카메라(`agentview`, `robot0_eye_in_hand`)를 84x84로 다시 그린다. 플래그는 `--done_mode 2
  --compress --exclude-next-obs`이다.

우리 구현(`tmg/generation/generators/cpgen/noisy_replay.py`)은 `tmg dataset`이 고른 시연(데이터셋 크기만큼)에만 이 두 단계를
적용한다. 공식 함수는 고치지 않고 부른다. 우리가 바꾼 점은 세 가지다.

- 공식 스크립트가 시연을 끝난 순서대로 합친다. 그래서 우리는 시연마다 저장된 공식 출력을 선택 순서대로 다시 묶는다.
- 다시 그리기는 robomimic 0.4.0의 `dataset_states_to_obs`로 한다. CP-Gen 생성 단계가 쓰는 함수와 같다. 저자 스크립트가 부르는
  adaflow 사본(`adaflow-cpgen`의 `scripts/dataset_states_to_obs.py`)은 이미지에 없다. 두 코드를 맞대 읽은 결과(2026-10-30), 관측을
  바꾸는 차이는 없었다. 둘 다 `env.reset()` 뒤 `reset_to`로 시작 상태를 놓고, t번째 관측을 `reset_to(states[t])`로 얻으며, 영상을 직접
  뒤집지 않는다. adaflow에만 있는 와인잔 받침대 회전, 자르기, 도메인 무작위화는 기본값에서 우리 과제에 아무 일도 하지 않는다.
- 다시 그리기를 12개 프로세스로 나눈다. 출력 내용은 바뀌지 않는다.

공식 스크립트는 시작 상태를 두 번 기록한다. 그래서 저장된 상태가 행동보다 한 걸음 늦다. 행동을 0번부터 세면, t번째 이미지(t ≥ 1)는
t − 1번째 행동을 적용하기 직전의 상태를 보여 준다. 예를 들어 0번과 1번 이미지는 모두 시작 상태다. Coffee 20개 확인 실행에서 모든
시연의 0번과 1번 상태가 같았다. 영상과 로봇 상태가 같은 파일에서 오므로 둘이 같은 만큼 늦고, 우리 학습과 저자 학습 모두 t번째 관측에
t번째 행동을 짝짓는다. 공개 데이터셋도 같으므로 고치지 않는다. 행동 라벨과 우리 기록(실행 행동, 하위작업 구간)은 청크의 것을 그대로 쓴다. 설정은
`configs/experiment/generator/cpgen.yaml`의 `noisy_state_replay`이다.

### 3.4 Three Piece Assembly의 성공 기준 차이

CP-Gen의 Three Piece Assembly 넓은 장면(`ThreePieceAssemblyWide`)은 두 번째 조각의 높이 허용오차를 0.035 m로 쓴다. MimicGen의 같은
과제는 0.02 m를 쓴다. 두 값 모두 각 생성기의 공식 코드 값이라 우리는 바꾸지 않는다. 다만 두 생성기의 성공률을 한 표에 놓으면 CP-Gen
쪽 기준이 더 느슨하다는 점을 함께 적어야 한다.

### 3.5 행동의 회전 표현과 정규화 (2026-10-30 결정)

CP-Gen의 행동은 절대 자세 7개 값이다. 위치 3개, 축-각 회전 3개, 그리퍼 1개다. 축-각 회전은 회전축 방향 벡터에 회전각을 곱한 값인데,
회전각이 π 근처이면 같은 자세가 방향이 반대인 두 벡터로 표현된다. 실제 데이터가 π를 넘나든다. A100의 Three Piece Assembly 풀 483개
시연에서 벡터 크기가 2.39–3.57 rad이었고, 한 걸음 사이에 벡터가 1 넘게 튀는 곳이 69번 있었다.

저자 학습(adaflow-cpgen, `abs_action: True`)은 회전을 6차원 표현으로 바꿔 학습한다. 6차원 표현은 회전 행렬의 앞 두 행을 이어 붙인 값으로,
같은 자세가 늘 같은 값이 되고 자세가 조금 바뀌면 값도 조금 바뀐다. 정규화는 위치만 최솟값-최댓값으로 [−1, 1]에 맞추고, 6차원 회전과
그리퍼는 그대로 둔다. 이전 판의 우리 학습은 축-각 7개 값을 모두 최솟값-최댓값 정규화했다.

2026-10-30 결정으로 우리도 저자와 같이 한다(`tmg/policy/rotation.py`, 설정은 `configs/experiment/generator/cpgen.yaml`의
`action_rotation: rotation_6d`).

- 학습은 데이터셋의 행동을 10개 값(위치 3, 6차원 회전 6, 그리퍼 1)으로 바꾼 뒤 통계를 낸다. 회전과 그리퍼 차원은 통계를 최솟값 −1,
  최댓값 1로 두어 정규화가 값을 바꾸지 않게 한다.
- 체크포인트에 `action_rotation.json`을 남긴다. 정책 서버는 예측한 10개 값을 그람-슈미트 정규직교화로 회전 행렬로 만든 뒤 축-각으로
  바꿔 환경에 7개 값을 보낸다. 데이터셋은 다시 만들 필요가 없다.
- 이 설정이 켜지면 행동 정규화를 모든 정책에서 최솟값-최댓값으로 둔다. BC-RNN은 원래 행동을 정규화하지 않는다. MimicGen의 변화량
  행동은 이미 [−1, 1] 안에 있어서 그래도 됐다. 그런데 BC-RNN은 행동 평균을 tanh(출력을 −1과 1 사이로 누르는 함수)로 내므로, CP-Gen의
  절대 위치(높이 0.83–1.17 m)와 축-각 값(최대 3.6)을 낼 수 없었다. 이전 판의 CP-Gen BC-RNN은 이 이유만으로도 제대로 학습될 수 없었다.

### 3.6 저자 학습과 아직 다른 점

2026-10-30 검토에서 찾았고, 아래는 아직 맞추지 않았다.

- 학습 갱신 수. 우리는 10만 갱신이다(2.1.0절). 저자 실행 스크립트는 250에폭을 돈다. Coffee 시연 평균 길이 약 228걸음으로 어림하면
  에폭당 약 3,400갱신, 250에폭이면 약 85만 갱신이다. 이것은 어림이고, 우리 갱신 수의 약 8배다.
- 정규화의 기준. 저자는 영상을 [−1, 1]로 바꾸고 인코더 안에서 ImageNet 평균과 표준편차로 정규화하며, 로봇 자세 사원수는 정규화하지
  않는다. 우리는 LeRobot 방식으로 영상을 채널별 평균과 표준편차로, 로봇 상태를 최솟값-최댓값으로 정규화한다.
  `configs/experiment/policy/diffusion_policy.yaml`이 적은 차이는 robomimic 기준이고, CP-Gen의 기준인 adaflow와의 차이는 이 항목이다.
- 검증용 시연. 저자는 시연의 2%를 학습에서 빼서(`val_ratio: 0.02`) 약 980개로 학습한다. 우리는 1,000개 모두로 학습한다.
- 논문 표와의 대응. 검토는 논문 표 1의 기본 열이 자세만 바꾸는 D1이고, Wide는 크기 변화를 더한 별도 열이라고 읽었다. 이 경우 Wide
  장면의 결과는 그 열과 비교해야 한다. 3.0절의 "논문의 주 실험은 Wide"라는 문장과 어긋날 수 있다. 논문 원문으로는 아직 확인하지
  못했다.

### 3.x 실행 환경에서 확인한 두 가지 (2026-10-27, A100 Singularity)

- 환경 클래스 등록. CP-Gen 이미지의 robosuite 1.5 포크에는 `single_arm_env` 모듈이 없어서 MimicGen의 환경 모듈(`mimicgen.envs.robosuite.*`)이
  불러와지지 않는다. Coffee_D0 같은 기본 환경 클래스는 CP-Gen의 `cpgen_envs` 패키지가 robosuite 1.5용으로 다시 구현해 등록한다.
  `tmg.generation.simulators.robosuite.register_env_class`는 `mimicgen` 다음에 `cpgen_envs`도 불러오며(없으면 건너뛴다), 그래서
  장면 클래스(Coffee_OURS)는 cpgen_envs의 Coffee_D0를 부모로 만들어진다.
- 쓰기 위치. CP-Gen은 cuRobo 체크아웃의 과제 설정 폴더(`/opt/src/curobo/src/curobo/content/configs/task`)에 직선 비용을 더한 설정
  파일을 새로 쓰고, 현재 작업 폴더에 임시 xml 파일(`eef_and_env_model.xml` 등)을 쓴다. 이미지 안은 읽기 전용이므로 Makefile은
  CP-Gen 이미지에서 그 설정 폴더 위에 호스트의 쓰기 가능한 복사본(`<data_root>/cpgen_writable/curobo_task`, 처음 쓸 때 이미지에서
  채운다)을 붙이고, 조각 프로세스는 자기 조각 폴더의 `cwd/`(체크아웃 최상위를 심볼릭 링크로 비춘 폴더)를 작업 폴더로 쓴다. 그래서
  조각 프로세스 여러 개가 같은 임시 파일을 두고 겹쳐 쓰는 일도 없다.
- 화면 밖 렌더링 장치. robosuite 1.5의 EGL 문맥은 `MUJOCO_EGL_DEVICE_ID`가 없으면 `CUDA_VISIBLE_DEVICES`의 값을 EGL 장치
  번호로 쓴다. A100 서버의 Singularity는 NVIDIA EGL 라이브러리를 넣어 주지 않아(nvidia-container-cli 방식, graphics 기능 없음)
  Mesa 소프트웨어 렌더링만 가능하고, Mesa는 사용자가 열 수 없는 GPU 8대의 DRM 장치 다음에 소프트웨어 장치를 9번째(번호 8)로
  나열한다. 그래서 `configs/machine/cluster/a100.yaml`의 `egl_device_id: 8`을 Makefile이 CP-Gen 이미지 실행에
  `MUJOCO_EGL_DEVICE_ID`로 넘기고, `tmg.core.egl.apply`가 그 번호를 `CUDA_VISIBLE_DEVICES` 뒤에 덧붙인다(robosuite 1.5가 두 변수의 일치를 확인하는데, CUDA는 존재하지 않는 번호부터는 무시하므로 보이는 GPU는 바뀌지 않는다). 도커(Isaac)에서는 이미지의 `NVIDIA_DRIVER_CAPABILITIES`에 graphics를 넣어 NVIDIA EGL을 쓴다.

### 3.y Coffee 재현 결과 (2026-10-27, A100 GPU 7, 영상 기반 Diffusion Policy)

이 실행은 크기 변화가 없는 `Coffee_D0` 기반의 옛 `ours` 장면에서 한 것이다(3.0절). A100의 풀·데이터셋·실행·고정 장면 폴더는
`.../cpgen/coffee/ours_d0_20261027/`로 이름을 바꿔 보존했고, 지금의 `ours`(Wide 기반)로 다시 돌린 결과가 이 표를 대신하게 된다.

| 항목 | 우리 측정값 | 논문 |
|---|---|---|
| 생성 성공률 (2,500회, 장면 ours, 10조각 병렬) | 96.4% (2,411회) | 자세 변동 과제 평균 89% (과제별 미보고) |
| 정책 성공률 (1,000개 학습, 9만 체크포인트, 300장면, 상한 600) | 187/300 = 62.3% (95% 구간 56.7–67.6%) | Coffee 자세 변동 RGB 1.00 ± 0.00, MimicGen 베이스라인 0.58 ± 0.04 |

검증은 체크포인트 1만부터 10만까지 25장면 성공이 0, 5, 7, 9, 10, 12, 12, 15, 16, 16회로 학습이 진행될수록 올랐다. 시도 1회에
약 45초, 성공 궤적 하나의 영상 렌더링(Mesa 소프트웨어)에 약 37초가 걸려 풀 전체는 조각당 약 6시간 50분이었다. 논문과의 정책
성공률 차이(62% 대 100%)는 장면 범위(우리 ours, 논문은 범위 수치 미보고), 체크포인트 선택 방식(우리는 25장면 검증, 논문은 시드
3개의 최고 체크포인트), Diffusion Policy 구현(LeRobot 대 DDPM 변형)이 다르고, 1.5절의 파이프라인 문제가 해결되기 전까지는
해석을 보류한다.

## 4. 기록 형식


청크 폴더 `records/`에 시도마다 `attempt_%06d.npz`와 `attempts.jsonl` 한 줄을 남긴다. 열 정의는 `tmg/core/records.py`의
모듈 문서에 있다. 핵심은 스텝마다 목표 자세와 실제 말단 자세(추적 오차 계산용), 잡음 전 명령과 실행 명령, 구간 종류(이음 또는
기술), 하위작업 종료 신호, 성공 판정, 경로 추종의 경로 변수와 전진량과 영역이고, 시도마다 장면 조건(물체 초기 자세), 하위작업마다
고른 원본과 혼합 기록이다.

## 5. 공식 실행과의 동일성 시험

`method=none`으로 만든 청크가 공식 스크립트 실행과 같은지는 2단계에서 과제마다 20회 생성으로 비교한다. 비교 대상은
`demo.hdf5`와 `demo_failed.hdf5`의 상태, 행동, 관측 배열과 `important_stats.json`의 성공 수다.

이 시험(`make verify-baseline`, `tmg/generation/verify.py`)은 MimicGen 청크 하나를 같은 `mg_config.json`으로 두 번, 접목을 끄고 한 번,
접목을 켜고 `method=none`으로 한 번 돌려 출력을 바이트 단위로 비교한다. 그래서 방법이 꺼져 있을 때 우리 접목 코드가 결과를 바꾸지
않는다는 것만 보인다. 두 실행이 같은 설정을 쓰므로 우리 생성 설정이 저자 설정과 같다는 것은 보이지 않는다. 예를 들어 1.4절의 Stack
Three 원본 선택 오류는 이 시험으로 잡을 수 없었다.

CP-Gen과 Isaac Lab 생성기는 GPU에서 실행마다 결과가 같지 않으므로 이 시험을 MimicGen에만 쓴다. 2026-10-30에 같은 커밋과 같은 시드로
두 번 돌린 측정에서, CP-Gen은 첫 시도 안에서 계획된 행동이 1.4e-7만큼 달라지기 시작했고, SoftMimicGen은 첫 행동부터 1e-5에서 1e-3만큼
달랐다. 20회 시도의 성공 수는 CP-Gen이 2회와 4회, SoftMimicGen이 12회와 12회였다. DexMimicGen(AutoData)도 같은 청크 시드에서 성공 20개를
채우는 데 34회와 38회 시도가 걸렸다(2.7절). 따라서 이 생성기들은 설정이 같으면 결과의 분포가 같을 뿐 결과 자체가 같지는 않고, 20회
시도 풀의 생성 성공률은 실행마다 몇 회씩 달라질 수 있다.
