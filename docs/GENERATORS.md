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

## 4. 기록 형식


청크 폴더 `records/`에 시도마다 `attempt_%06d.npz`와 `attempts.jsonl` 한 줄을 남긴다. 열 정의는 `tmg/core/records.py`의
모듈 문서에 있다. 핵심은 스텝마다 목표 자세와 실제 말단 자세(추적 오차 계산용), 잡음 전 명령과 실행 명령, 구간 종류(이음 또는
기술), 하위작업 종료 신호, 성공 판정, 경로 추종의 경로 변수와 전진량과 영역이고, 시도마다 장면 조건(물체 초기 자세), 하위작업마다
고른 원본과 혼합 기록이다.

## 5. 공식 실행과의 동일성 시험

`method=none`으로 만든 청크가 공식 스크립트 실행과 같은지는 2단계에서 과제마다 20회 생성으로 비교한다. 비교 대상은
`demo.hdf5`와 `demo_failed.hdf5`의 상태, 행동, 관측 배열과 `important_stats.json`의 성공 수다.
