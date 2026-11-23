# SkillMix 데이터셋으로 정책 학습·평가하기 (2026-11-23)

이 문서는 허깅페이스에 올려 둔 MimicGen SkillMix 학습 데이터셋으로 정책(Diffusion Policy)을 학습하고 평가하는 방법을 적는다.
데이터를 새로 생성하지 않는다. 섞기 규칙(계수) 파일도, 원본 시연도, 생성 풀도 필요 없다. 데이터셋을 받아 학습하고, 평가
장면은 각자 만들어 이 문서의 체크섬과 같은지 확인한다.

이 문서는 2026-11-13의 [MIMICGEN_MIXUP_RUN_2026-11-13.md](MIMICGEN_MIXUP_RUN_2026-11-13.md)를 대신한다. 그 문서의 섞기
방법(`mixup_only`)은 main에서 빠졌다.

## 차례

- [1. 데이터셋](#1-데이터셋)
- [2. 준비](#2-준비)
- [3. 평가 장면 만들고 확인하기](#3-평가-장면-만들고-확인하기)
- [4. 실행](#4-실행)
- [5. 렌더러](#5-렌더러)
- [6. 결과 위치](#6-결과-위치)
- [7. 기준선과 비교할 때](#7-기준선과-비교할-때)

## 1. 데이터셋

허깅페이스 공개 데이터셋 저장소 [`Mayfull/Mimicgen-skillmix`](https://huggingface.co/datasets/Mayfull/Mimicgen-skillmix)에
과제마다 파일이 하나 있다. 로그인하지 않아도 받을 수 있다.

| 파일 | 과제 |
|---|---|
| `coffee.hdf5` | Coffee |
| `stack_three.hdf5` | Stack Three |
| `hammer_cleanup.hdf5` | Hammer Cleanup |
| `square.hdf5` | Square |
| `mug_cleanup.hdf5` | Mug Cleanup |
| `threading.hdf5` | Threading |
| `three_piece_assembly.hdf5` | Three Piece Assembly |

- 파일마다 성공 시연 1,000개가 들어 있다(`data/demo_0` … `data/demo_999`). robomimic 형식이다.
- 카메라는 `agentview`와 `robot0_eye_in_hand`, 84 × 84 RGB이고, GPU(NVIDIA EGL)로 그렸다(5절).
- MimicGen(robosuite 1.4)으로, 장면 범위 `ours`(`configs/experiment/generator/mimicgen/scene/ours.yaml`)와 서브태스크마다
  원본을 고르는 방식 `ours`에서 SkillMix로 섞어 만들었다.

## 2. 준비

1. main을 받는다(2026-11-23 이후, PR #2가 들어간 판). `GENERATION=hub` 설정
   (`configs/experiment/generation/hub.yaml`)이 이 저장소와 파일 이름을 알고 있다.
2. 이미지를 만든다. robosuite 이미지(평가)와 학습 이미지가 필요하다(README 3.2).
3. 머신 설정(`configs/machine/cluster/<이름>.yaml`)을 확인한다.
   - `render: egl`(기본값)이어야 한다(5절).
   - EGL 장치 번호가 GPU 번호와 다른 머신이면 `egl_device_of_gpu`에 GPU 0, 1, …의 EGL 번호를 적는다. 모르면
     `docs/MIMICGEN_BASELINE_RUNBOOK.md`의 렌더링 절에 확인 방법이 있다.
   - Singularity 머신에서 `--nv`가 NVIDIA EGL 라이브러리를 가져오지 못하면 `nvidia_egl_dir`를 적는다(같은 절).

## 3. 평가 장면 만들고 확인하기

평가 장면(검증 25개, 시험 300개)은 데이터셋과 함께 올리지 않았다. 장면 범위 설정과 고정된 시드(202600)로 뽑기 때문에,
같은 코드로 만들면 같은 파일이 나온다. 과제마다 한 번 만든다.

```bash
make resets GENERATOR=mimicgen TASK=<task> SCENE=ours CLUSTER=<머신>
md5sum <데이터 폴더>/resets/mimicgen/<task>/ours/val25_test300_seed202600.hdf5
```

결과가 아래 체크섬과 같아야 한다. 이 값은 우리 실험이 평가에 쓴 파일의 것이다. main 코드로 새로 만들어도 7과제 모두
같게 나오는 것을 확인했다(2026-11-23, A100). 장면 상태만 저장하므로 렌더러와 상관없다.

| 과제 | md5 (앞 8자리) |
|---|---|
| coffee | `cd45881a` |
| square | `fba34f1a` |
| stack_three | `f91f983c` |
| threading | `972e53ed` |
| mug_cleanup | `62ddd513` |
| hammer_cleanup | `4be63966` |
| three_piece_assembly | `57d5242d` |

- 이미 같은 경로에 평가 장면 파일이 있으면 `make resets`는 그 파일을 그대로 둔다(`resets exist, keeping them`). 그 파일의
  체크섬이 위와 다르면, 그 파일은 옮겨 두고(지우지 말고) 다시 만든다.
- 특히 Hammer Cleanup, Mug Cleanup, Three Piece Assembly는 2026-11-04에 장면 범위를 좁히기 전의 파일이 남아 있을 수 있다.
  그 파일로 평가하면 학습 데이터보다 넓은 범위에서 시험하게 된다(우리 실험에서 TPA 성공률이 2 %대에서 16 %로 바뀐 원인이다).

## 4. 실행

과제 하나씩 돌릴 때:

```bash
C="GENERATOR=mimicgen TASK=<task> SCENE=ours SELECTION=ours METHOD=skillmix GENERATION=hub OBSERVATION=image CLUSTER=<머신>"
make dataset $C                       # 허깅페이스에서 <task>.hdf5를 받아 데이터셋 경로에 둔다(이미 있으면 건너뜀)
make resets  GENERATOR=mimicgen TASK=<task> SCENE=ours CLUSTER=<머신>
make train   $C POLICY=diffusion_policy SEED=202601
```

7과제 × 시드 3개를 한 번에 돌릴 때(학습, 체크포인트마다 검증, 체크포인트 고르기, 시험까지):

```bash
CLUSTER=<머신> GPU=<번호> GENERATION=hub SELECTION=ours METHOD=skillmix POLICIES=diffusion_policy \
  scripts/run_mimicgen_baseline.sh
```

- `SCENE`, `SELECTION`, `METHOD`, `OBSERVATION`은 위 값이어야 한다. 이 이름이 데이터셋과 실행 폴더 이름이 되므로, 다른 값을
  주면 `make dataset`이 받기 전에 거부한다.
- 생성 단계(원본 받기, 풀 만들기)는 건너뛴다. `make generate GENERATION=hub`는 거부된다.
- 시드는 기본 `202601 202602 202603`이다. 학습 횟수, 배치, 검증·시험 장면 수는 기준선과 같은 기본값이다
  (`configs/experiment/training/default.yaml`, `configs/experiment/evaluation/default.yaml`).
- 같은 머신에서 드라이버를 둘 돌리면 `PORT_BASE`를 서로 40 × 실행 수 이상 떨어뜨린다.

## 5. 렌더러

데이터셋의 이미지는 GPU 렌더러(NVIDIA EGL)로 그렸다. **평가도 EGL로 해야 한다.**

- 같은 장면을 CPU 렌더러(Mesa)와 EGL로 그리면 화소의 50–61 %가 다르다(평균 0.7–1.4단계, 최대 76단계, 2026-11-22 측정).
- 학습 이미지와 다른 렌더러로 평가하면, 정책은 학습 때와 조금 다른 이미지를 보고 평가받는다. 우리 BC-RNN 실험에서는
  렌더러를 맞추자 과제에 따라 성공률이 −2 %p에서 +5 %p 바뀌었다.
- main의 기본이 `render: egl`이다. GPU가 없는 머신에서 `RENDER=mesa`로 돌리지 않는다.
- 렌더러는 풀 조각 설정에는 기록되지만, 평가 결과에는 따로 남지 않는다. 평가를 돌린 머신 설정을 결과와 함께 적어 둔다.

## 6. 결과 위치

`<데이터 폴더>/runs/mimicgen/<task>/ours/ours/skillmix/hub/image/diffusion_policy/seed<시드>/` 아래에 있다.

- `eval/test_h600/<체크포인트>/eval_summary.json`: 시험 300개의 결과(`by_split.test.success_rate`, `n_success`, `n`).
- `SELECTED.json`: 검증으로 고른 체크포인트.
- 드라이버로 돌리면 정책마다 결과를 묶은 파일이 `<데이터 폴더>/reports/<BUNDLE_NAME>_diffusion_policy.tar.gz`에 생긴다.

## 7. 기준선과 비교할 때

- 기준선(공식 원본 선택, 섞지 않음: `SELECTION=official METHOD=none`)과 평가 장면 파일이 같아야 한다. 평가 장면 파일은 장면
  (`ours`)마다 하나라 기준선과 SkillMix가 같은 파일을 쓴다. 3절의 체크섬으로 확인한다.
- 기준선도 같은 렌더러(EGL)로 평가한다. 기준선을 Mesa로 평가한 결과와 SkillMix를 EGL로 평가한 결과는 바로 비교하지 않는다.
- 학습 설정(학습 횟수, 배치, 체크포인트 고르기)은 기준선과 같은 기본값을 쓴다.
