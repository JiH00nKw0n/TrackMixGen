# Isaac Lab 혼합 코드의 출처: 보관 브랜치와 Isaac 서버 사본 (2026-11-27)

이 문서는 Isaac Lab 생성기(SoftMimicGen의 rope·towel, DexMimicGen의 pick_place·microwave·bottle_door)에서 혼합을 돌렸던
이전 코드가 어디에 남아 있는지, 그리고 그 가운데 어느 파일이 git 어디에도 없고 Isaac 서버에만 있는지를 정리한다.
브랜치 `isaaclab-skillmix`의 새 구현(SkillMix 소프트맥스 규칙)은 이 코드를 참고만 했고 옮겨 오지 않았다.

## 1. 비교 방법

- Isaac 서버 `/home/gpuadmin/workspace/jihoonkwon/` 아래 네 사본 `TrackMixGen_q5`, `TrackMixGen_q5signed`,
  `TrackMixGen_q5_rope_split`, `TrackMixGen_q5rope_signed`에서 `tmg configs scripts tests docker` 아래의 코드·설정 파일
  (`*.py *.yaml *.md *.sh`, 백업 파일 `*.bak* *.orig *~ *.backup*`)과 `Makefile pyproject.toml README.md`의 md5와 수정 시각을
  읽기 전용 호출 한 번(2026-11-27)으로 받았다. 파일 내용도 같은 호출의 tar로 받아 md5가 서버 값과 모두 같음을 확인했다.
- 각 파일을 로컬 git의 세 커밋과 비교했다(`git show <커밋>:<경로>`의 md5).
  - `0f64078` (2026-11-07 01:00, 질문 4, `origin/archive/mixval-convex-mixing`의 끝): 네 사본 가운데 `q5`와 `q5signed`의 HEAD.
  - `489a20d` (`origin/archive/mixval-q5-isaaclab`, 2026-11-07 09:20, 질문 5): `0f64078` 위의 커밋 하나.
  - `6e4943b` (`origin/archive/mixval-q6-smooth-resp`, 2026-11-17, 질문 6): `489a20d` 위의 커밋 하나.
- 시각은 서버 파일 시스템의 UTC 값이다(한국 시각은 9시간 뒤).

## 2. 사본별 상태

| 사본 | git HEAD | 비교한 파일 | 보관 브랜치(q5, q6)와 내용이 같지 않은 파일 |
|---|---|---|---|
| `TrackMixGen_q5` | `0f64078` | 314 | 11 (코드 0, 설정 9, 백업 2) |
| `TrackMixGen_q5signed` | `0f64078` | 323 | 24 (코드 3, 설정 19, 백업 2) |
| `TrackMixGen_q5_rope_split` | `52a33d4` | 322 | 26 (코드 5, 설정 4, 스크립트 9, 테스트 8) |
| `TrackMixGen_q5rope_signed` | `52a33d4` | 323 | 26 (rope_split과 같은 목록, 그중 두 파일의 내용이 다르다) |

`52a33d4` (2026-11-07 22:47, "Load measured source rankings from explicit YAML paths")는 로컬 저장소에도 원격 브랜치에도 없는
커밋이다(`git cat-file`이 찾지 못한다). 두 rope 사본은 git 저장소이지만 이 커밋을 푸시한 적이 없다.

## 3. Isaac 서버에만 있는 코드

### 3.1 `TrackMixGen_q5`

코드 파일은 모두 보관 브랜치 어딘가와 같다. 작업 트리에서 수정된 파일(`git status`의 M)은 이렇다.

| 파일 | 같은 커밋 |
|---|---|
| `tmg/generation/generators/isaaclab/runtime.py`, `autodata/runtime.py`, `Makefile` | q5, q6 |
| `tmg/generation/method/transform_model.py` | q5 (q6의 것과는 다르다) |
| `scripts/evaluate_checkpoint.sh`, `scripts/task_policy_pipeline.sh` | q6 |

즉 `489a20d`는 이 사본의 작업 트리를 커밋한 것이다. 서버에만 있는 것은 설정과 백업뿐이다.

- `configs/experiment/generation/pool_robosuite_lowdim_persub.yaml` (11-10 14:26)
- `configs/experiment/generator/dexmimicgen/ranking/microwave.yaml` (11-07 05:30, q5·q6에 있는 같은 이름의 파일과 내용이 다르다)
- `configs/experiment/generator/dexmimicgen/transform_model_{decay,decay_inhand,location,location_inhand}/microwave.yaml` (11-07 05:30)
- `configs/experiment/generator/dexmimicgen/transform_model_signed_inhand/microwave.yaml` (11-09 15:21)
- `configs/experiment/method/mx_convex_adaptive_post.yaml` (11-10 14:18), `mx_random5.yaml` (11-09 19:14)
- 백업 `scripts/evaluate_checkpoint.sh.bak-q5`, `scripts/task_policy_pipeline.sh.bak-q5` (둘 다 11-07 07:35)

알려진 예로 들었던 "q5의 어댑터를 11/11에 고치고 백업을 남김"은 이 목록에서 찾지 못했다. `TrackMixGen_q5`의
`autodata/adapter.py`는 `0f64078`과 같고, 수정된 어댑터는 `q5signed`의 것(아래, 11-10 14:25 UTC = 11-10 23:25 KST)이다.
읽은 디렉터리(`tmg configs scripts tests docker`) 밖에 있는 백업이라면 이 비교에는 들어 있지 않다.

### 3.2 `TrackMixGen_q5signed`

q5와 같은 HEAD에 다음 수정이 더 있다(모두 서버에만 있다).

| 파일 | 수정 시각 | 내용 |
|---|---|---|
| `tmg/generation/method/transform_model.py` | 11-10 14:24 | 계수의 Laplace 사후 표본(`--posterior-samples`, `predict_posterior`). q6의 것보다 128줄 다르다 |
| `tmg/generation/method/convex_mix.py` | 11-10 14:18 | 규칙 `posterior_best`: 가중치 = 사후 표본에서 그 소스의 예측 성공이 가장 높을 확률 |
| `tmg/generation/generators/autodata/adapter.py` | 11-10 14:25 | `generation.protocol.autodata_policy`로 AutoData `generation_policy` 필드를 덮어쓴다(`pool_robosuite_lowdim_persub`: 팔마다, 부분과제마다 새 소스) |
| `configs/experiment/generator/dexmimicgen/ranking/pick_place.yaml` | 11-10 02:35 | 보관 브랜치의 같은 파일과 내용이 다르다 |

설정 파일 가운데 서버에만 있는 것(11-07부터 11-11 사이):
`transform_model/{microwave,pick_place}.yaml`, `transform_model_location/{pick_place,pick_place.onesrc}.yaml`,
`transform_model_location_inhand/pick_place.yaml`, `transform_model_location_post/{pick_place,pick_place.onesrc}.yaml`,
`transform_model_signed_inhand/{microwave,pick_place}.yaml`, `transform_model_{decay,decay_inhand,location,location_inhand}/microwave.yaml`
(이상 `configs/experiment/generator/dexmimicgen/` 아래), `configs/experiment/generator/mimicgen/transform_model_location_post/threading.yaml`
(11-11 08:21), `ranking/microwave.yaml`, `generation/pool_robosuite_lowdim_persub.yaml`, `method/mx_convex_adaptive_post.yaml`,
`method/mx_random5.yaml`, 그리고 q5와 같은 백업 두 개.

### 3.3 `TrackMixGen_q5_rope_split` (커밋 `52a33d4`, 원격에 없음)

| 파일 | 내용 |
|---|---|
| `tmg/generation/method/soft_pose.py` | 로프의 프레임 `nodal_yaw_v1`: 노드 중심, 그리고 소스 0의 첫 노드 구름에 대한 대응점 최소제곱 yaw |
| `tmg/generation/generators/isaaclab/runtime.py` | SoftMimicGen 런타임이 혼합 전에 로프 프레임·손 안 거리를 계산하고 기록(`soft_pose`, `soft_reference_sha256`, `inhand_gap`); 다른 소스의 시작 eef 자세도 노드 정합으로 옮김 |
| `tmg/generation/method/transform_model.py` | `soft_rows`: 로프 기록의 프레임을 소스 파일과 다시 맞춰 보고 행을 만듦 |
| `tmg/generation/method/convex_mix.py` | 섞는 좌표계는 단위 행렬로 두고 예측에는 노드에서 얻은 원래 자세(`feature_source_poses`)를 씀 |
| `tmg/generation/method/ranking.py` | 순위 파일을 이름 대신 명시한 YAML 경로로도 읽음(커밋 `52a33d4`의 제목) |
| `configs/experiment/generator/softmimicgen/task/rope.yaml` | `soft_pose: nodal_yaw_v1` 한 줄 |
| 설정 | `generation/q5_rope_500.yaml`, `generation/test/q5_rope_smoke.yaml`, `method/mx_random3.yaml` |
| 스크립트 | `scripts/q5_rope_{archive_auth,chunk_exchange,generate,generation_workflow,hub_archive,partition,policy_workflow,prepare_models,split_sites}.py` |
| 테스트 | `tests/unit/test_{soft_pose,convex_soft_frames,q5_archive,q5_partition,q5_rope_generate,q5_split_sites}.py`, 바뀐 `test_isaaclab_runtime.py`, `test_ranking.py` |

### 3.4 `TrackMixGen_q5rope_signed`

`q5_rope_split`과 같은 커밋에서 두 파일이 더 바뀌었다(11-09 16:59). `convex_mix.py`가 부호 있는 변위
(`signed_displacement`)를 모델에 넘기고, `transform_model.py`에 부호 있는 변위와 `--link gauss`("responsibility" 혼합)가
더해졌다. 메서드 설정 `mx_convex_signed_inhand.yaml`은 q6 보관 브랜치에 있는 것과 같다.

## 4. 새 구현이 참고한 것과 다르게 한 것

- 로프·수건의 물체 프레임: `soft_pose.py`의 정의(`nodal_yaw_v1`)를 그대로 따랐다(`tmg/generation/method/soft_frame.py`).
  기록에는 `object_frame: {version, reference_sha256}`이 남고, 적합은 정의나 기준 구름이 다른 기록을 거부한다.
- 손 안 특징: 보관 코드는 물체 좌표계에서 본 손의 위치 거리 하나를 썼다. 새 구현은 main의 SkillMix 정의(물체를 eef 좌표계에서
  본 자세의 위치 거리와 회전각, MimicGen·CP-Gen과 같음)를 쓴다.
- 두 팔: 보관 코드(`transform_model.isaaclab_rows`)는 부분과제가 가장 많은 팔 하나만 적합했다. 새 구현은 (팔, 부분과제)마다
  블록을 만들고 규칙 파일에 `<eef>/<k>`로 쓴다.
- 혼합기 자체(볼록 혼합, `posterior_best`, gauss 응답)는 옮기지 않았다. SkillMix 소프트맥스 규칙이 대신한다.
- AutoData `generation_policy` 덮어쓰기(`autodata_policy`, q5signed 어댑터)와 top-k 소스 선택(`pick_source`)은 옮기지 않았다.
