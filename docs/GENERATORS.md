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
| `new_initial_state = env.get_state()` 다음 | `_tmg.begin_attempt(env, env_interface, new_initial_state)` | 시도 기록을 시작하고 장면 메타(도달 제약의 거부 횟수 등)를 읽는다 |
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
| `waypoint = seq[j]` | `waypoint = _tmg.waypoint(seq, j, env, env_interface)` | 끄면 `seq[j]`. 켜면 측정한 말단 자세로 다음 목표를 정한다 |
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

### 1.4 원본 시연 선택 방식: 지금은 우리 값, 최종 평가는 공식 템플릿 값 (2026-10-27 결정)

생성 프로토콜은 `configs/experiment/generation/pilot20.yaml`의 `protocol` 블록에 있고, 다른 generation 설정(`pool`, `pool_lowdim`,
`rank3000`, `check200`, `pilot20_images`)은 이 블록을 물려받는다. 지금 값은 하위작업마다 원본 시연을 독립적으로 고르고
(`select_src_per_subtask: true`) 고르는 방식은 무작위다(`selection_strategy: random`). MimicGen 공식 설정 템플릿
(`mimicgen/exps/templates/robosuite/*.json`)은 일곱 과제 모두 `select_src_per_subtask: false`(한 시도의 모든 하위작업이 같은 원본
시연을 쓴다)이고, Square의 두 번째 하위작업과 Stack Three의 모든 하위작업은 `nearest_neighbor_object`(현재 물체 자세와 가장
가까운 원본을 고른다)를 쓴다. 나머지 과제와 하위작업은 공식도 random이다.

결정은 두 단계로 나뉜다.

- 지금 진행 중인 재현 확인과 혼합 방법 탐색은 현재 값으로 계속한다. 이미 만든 기준 풀 일곱 개와 탐색 결과를 그대로 쓰기 위해서다.
- 논문에 들어갈 최종 평가에서 MimicGen은 공식 템플릿 값을 그대로 써야 한다. 즉 모든 과제에서 한 시도의 하위작업들이 같은 원본
  시연을 쓰고, Square의 두 번째 하위작업과 Stack Three의 모든 하위작업은 현재 물체 자세에 가장 가까운 원본을 고른다. 이를 위해
  필요한 작업은 두 가지다. 첫째, `tmg/generation/generators/mimicgen/config_builder.py`가 지금은 하나의 `selection_strategy`를 모든
  하위작업에 같이 넣으므로, 과제 템플릿의 하위작업별 값을 기본으로 쓰도록 바꿔야 한다(프로토콜에서 값을 주지 않으면 템플릿 값을
  남기는 방식). 둘째, 그 값으로 기준 풀 일곱 개를 다시 생성해야 한다. 최종 평가 전까지 이 두 가지가 끝나지 않으면 평가를 시작하지
  않는다.

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
| 우리 생성, 하위작업별 원본 독립 선택 (`pool_lowdim`) | 10.3% (11,000회 중 1,137회) | 9만 체크포인트 2/25 | 14/300 = 4.7% (95% 구간 2.8–7.7%) |
| 우리 생성, 한 시도에 원본 하나 (`pool_lowdim_onesrc`, 공식 템플릿 값) | 29.1% (4,000회 중 1,166회) | 10만 체크포인트 4/25 | 14/300 = 4.7% (95% 구간 2.8–7.7%) |
| 논문 공식 D2 데이터셋 (`scripts/import_robomimic_dataset.py`로 변환, `generation=official`) | 해당 없음 | 7만 체크포인트 2/25 | 8/300 = 2.7% (95% 구간 1.4–5.2%) |
| 논문 보고값 | 31.3% | 해당 없음 | 38.7% (체크포인트별 50회 평가의 최대값, 시드 3개 중 최대) |

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
주는 상향 편향(체크포인트 40개와 시드 3개 가운데 최대값). 이를 가르기 위해 같은 공식 데이터로 robomimic 원본 구현(robomimic 0.5.0,
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
- 추정인 것. 논문의 38.7%와의 차이는 학습 예산(20만 갱신)과 보고 방식(시드 3개, 체크포인트별 최대값)에서 나온다고 보는 것이 지금까지의
  증거에 맞는다. 다만 10만 갱신까지의 추세(0%에서 4%로 느리게 오름)에서 추정한 것이고, 20만 갱신으로 늘렸을 때 38.7%에 이르는지는
  측정하지 않았다.
- 따라서 Three Piece Assembly D2는 재현 조사를 여기서 끝내고, 우리 학습기와 평가기는 그대로 둔다. 최종 평가에서 이 과제를 쓸 때는 논문
  수치와 직접 비교하지 말고, 같은 파이프라인 안에서 기준 방법과 제안 방법을 비교한다.

## 2. Isaac Lab 생성기: SoftMimicGen과 DexMimicGen

두 생성기는 Isaac Lab의 `generate_dataset.py` 스크립트로 돈다. 스크립트는 인자를 읽은 뒤 Isaac Sim을 띄우고 나서야 Isaac Lab
모듈을 들여오므로, 접목을 미리 설치할 수 없다. 청크 프로세스(`tmg/generation/generators/isaaclab/entrypoint.py`)는 스크립트
소스를 읽어 `if __name__ == "__main__":` 바로 앞에 `_tmg_install(globals(), _tmg_spec)` 한 줄을 끼운 사본을 실행한다. 그
함수는 스크립트의 들여오기가 끝난 뒤, `main()`이 돌기 전에 (1) 카메라 환경 모듈을 들여오고, (2) `setup_env_config`를 감싸
장면 범위를 환경 설정에 써넣고, (3) 접목을 설치한다. 공식 파일은 디스크에서 바뀌지 않는다.

패키지는 Soft가 `softmimicgen.datagen`, Dex가 `isaaclab_mimic.datagen`이다. 끼우는 자리는 다음과 같다.

| 함수 | 앵커(대입되는 이름) | 끼우는 문장 | 하는 일 |
|------|---------------------|-------------|---------|
| `DataGenerator.generate` | `new_initial_state = ...` 뒤 | `_tmg.begin_attempt(self, env_id, new_initial_state)` | 시도 기록을 열고, 잡음 전 라벨을 받기 위해 `env.target_eef_pose_to_action`을 감싼다 |
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
따로 전진시키는데, 추종기는 두 손을 경로 변수 하나로 움직여야 하기 때문이다. 그래서 `runtime.make_tracking_generate`가
시도 동안 `DataGenerator.generate`를 바꿔 끼운다. 바뀐 함수도 공식 궤적 생성기(`generate_*trajectory`,
`merge_eef_subtask_trajectory`), `MultiWaypoint.execute`, 공식 기록기를 그대로 부르고, 두 말단이 모두 궤적을 가진 원본
시간 구간마다 `ErrorAdaptiveFollower` 하나를 만든다(`_cut_interval`이 구간을 자른다). 오차는 설정된 링크
(`generator.measure_links`, Dex는 `*_hand_pitch_link`)의 자세로 재고, 집게 사건은 두 손의 명령 사건을 합쳐 쓰며, 공식
걸음별 잡음 뽑기는 `NoiseLedger`가 보존한다(원본 경유점마다 공식 뽑기 한 묶음, 추가 걸음은 사설 난수). `method=none`은
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

## 3. CP-Gen (kevin-thankyou-lin/cpgen@1e7c1fc4)

CP-Gen은 원본 구간을 다시 재생하지 않는다. 시도마다 주석된 키포인트 제약을 새 물체 자세에 맞춰 다시 풀고(제약 걸음),
그 시작점까지 cuRobo로 운동 계획을 세운다(계획 걸음). `CPPolicy.get_action`은 대기열이 비면 다음 제약을 풀어 계획 걸음과
제약 걸음을 `action_queue`에 채우고, 한 걸음에 하나씩 꺼낸다. `DemoGenerator.generate_demo`는 꺼낸 행동에 공식 균등
잡음을 더해 환경에 넣고, 꺼낸(잡음 없는) 행동을 저장한다. 그래서 CP-Gen의 저장 행동은 이미 잡음 전 라벨이다.

끼우는 함수는 `DemoGenerator.generate_demo` 하나다(`tmg/generation/generators/cpgen/runtime.py`).

| 앵커(대입되는 이름) | 끼우는 문장 | 하는 일 |
|---------------------|-------------|---------|
| 첫 `state = self.env.get_state()` 뒤 | `_tmg.begin_attempt(self, obs, state)` | 시도 기록을 열고 장면 메타데이터를 읽는다 |
| `action_dct = self.policy.get_action(obs)` | `action_dct = _tmg.get_action(self, obs)`로 바꾼다 | 공식 `get_action`을 부르고, 새 구간이 계획되면 구간(계획=연결, 제약=기술)과 원본 번호를 기록한다. 추종이 켜지면 계획된 행동들을 경로 하나로 삼아 추종기가 다음 목표를 낸다 |
| 반복문 안 `state = self.env.get_state()` 뒤 | `_tmg.record_step(self, obs, action, action_to_take, reward, in_constraint_segment, system_noise_cfg)` | 목표 자세, 측정 자세, 라벨, 실행 행동, 잡음 크기를 기록한다 |
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
