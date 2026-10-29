# MimicGen 기준선 실행 안내 (일곱 과제, Diffusion Policy와 BC-RNN)

이 문서 하나로 GPU 한 장짜리 컴퓨터에서 MimicGen 기준선 실험을 처음부터 끝까지 돌리고, 분석에 필요한 결과 파일을 한 묶음으로
받을 수 있다. 기준선이란 공식 MimicGen 생성기를 공식 설정 그대로 돌려 시연을 만들고, 그 시연으로 우리 설정(넓힌 장면, 영상 관측,
평가 상한 600걸음, 고정 평가 장면)에서 정책을 학습하고 평가하는 것이다. 실험은 과제마다 "공식 원본 시연 10개 내려받기, 공식 설정으로
성공 시연 1,000개 이상 생성, 고정 평가 장면 만들기, 학습 데이터셋 1,000개 쓰기, 영상 Diffusion Policy와 영상 BC-RNN을 각각 시드
3개(202601, 202602, 202603)로 학습, 시드마다 검증 25장면으로 체크포인트 선택, 시험 300장면 평가" 순서로 진행한다. 명령
하나(`scripts/run_mimicgen_baseline.sh`)가 전부를 돌리고, 중단되면 같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 이어서 돈다.

## 차례

1. 돌리는 과제와 실험 설정. 일곱 과제, 생성 설정, 장면, 관측, 정책, 평가의 값을 적는다.
2. 요구 사항. 운영체제, GPU, CPU, 도커, 디스크, 네트워크 조건을 적는다.
3. 준비. 도커 설치, 저장소 받기, 데이터 폴더와 머신 설정, 이미지 두 개 빌드, 동작 확인을 순서대로 적는다.
4. 실행. 전체 실행 명령, 진행 상황 확인, 중단 뒤 재시작 방법을 적는다.
5. 결과 파일. 결과 묶음의 폴더 구조와 각 파일의 내용, 수치를 읽는 방법을 적는다.
6. 난수와 재현성. 어떤 시드가 어디에 쓰이는지 적는다.
7. 단계별 수동 실행. 문제를 좁힐 때 쓰는 단계별 명령을 적는다.
8. 측정한 시간과 예상 시간. 측정값과 추정값을 구분해 적는다.
9. 알려진 제한과 문제 해결. 자주 나는 오류와 대처를 적는다.

## 1. 돌리는 과제와 실험 설정

과제는 MimicGen 논문의 robosuite 과제 일곱 개다. 원본 시연은 모두 MimicGen이 공개한 사람 시연 10개(Hugging Face 저장소
`amandlek/mimicgen_datasets`의 `source/<과제>.hdf5`)이고, 스크립트가 처음 실행할 때 내려받는다.

| 과제 (설정 이름) | 하는 일 | 하위작업 수 |
|---|---|---|
| Coffee (`coffee`) | 커피 포드를 집어 커피 머신에 넣고 뚜껑을 닫는다 | 2 |
| Hammer Cleanup (`hammer_cleanup`) | 서랍을 열고 망치를 넣고 서랍을 닫는다 | 3 |
| Mug Cleanup (`mug_cleanup`) | 서랍을 열고 머그잔을 넣고 서랍을 닫는다 | 3 |
| Square (`square`) | 사각 너트를 집어 사각 기둥에 끼운다 | 2 |
| Stack Three (`stack_three`) | 정육면체 세 개를 쌓는다 | 4 |
| Threading (`threading`) | 바늘을 집어 삼각대의 구멍에 끼운다 | 2 |
| Three Piece Assembly (`three_piece_assembly`) | 조각 두 개를 바닥판 위에 차례로 끼운다 | 4 |

생성 설정은 MimicGen 공식 설정 템플릿(`mimicgen/exps/templates/robosuite/<과제>.json`)의 값을 그대로 쓴다. 우리 코드는 템플릿에
시도 수, 시드, 입출력 경로, 저장할 카메라만 써 넣는다. 아래 표의 값은 그 템플릿에서 읽은 것이고, 우리 생성 설정 파일
`configs/experiment/generation/pool_robosuite_image.yaml`은 원본 선택 항목을 비워 두어(null) 템플릿 값이 남게 한다.

| 항목 | 값 |
|---|---|
| 한 시도의 모든 하위작업이 같은 원본 시연을 쓰는가 | 그렇다 (`select_src_per_subtask: false`, 일곱 과제 공통) |
| 원본 시연을 고르는 방식 | 무작위. 단, Square의 첫 하위작업(너트 잡기)과 Stack Three의 모든 하위작업은 현재 물체 자세와 가장 가까운 원본 3개 중에서 고른다(`nearest_neighbor_object`, `nn_k: 3`) |
| 행동 잡음 | 0.05 (모든 하위작업) |
| 하위작업 사이 보간 걸음 수 | 5, 고정 걸음 0 |
| 시도 수 | 과제마다 다르다(아래). 성공 수를 보장하지 않고 정해진 수만큼 시도한 뒤 성공만 데이터셋에 쓴다 |
| 실패 시도의 저장 | MimicGen의 실패 시연 파일(`demo_failed.hdf5`)은 쓰지 않는다. 모든 시도의 걸음별 기록과 성공 여부는 우리 기록(`records/`)에 남는다 |

시도 수는 저차원 풀(2026-10, 하위작업별 무작위 선택)에서 성공 1,000개가 나오기까지 걸린 시도 수에 10%를 더한 값으로 시작하고, 풀의
성공 수가 1,000개에 못 미치면 스크립트가 시도 수를 25%씩 늘려 이어서 생성한다(이미 끝난 묶음은 다시 만들지 않는다).

| 과제 | 처음 시도 수 | 저차원 풀에서 본 생성 성공률 (참고값, 선택 규칙이 다르다) |
|---|---|---|
| Coffee | 2,500 | 48.0% |
| Threading | 6,250 | 18.4% |
| Hammer Cleanup | 7,000 | 17.7% |
| Mug Cleanup | 8,750 | 12.4% |
| Stack Three | 9,500 | 11.4% |
| Square | 15,000 | 7.0% |
| Three Piece Assembly | 17,000 | 10.3% (원본 하나로 돌린 풀에서는 29.1%) |

장면(`scene=ours`, `configs/experiment/generator/mimicgen/scene/ours.yaml`)은 물체의 초기 배치 범위를 넓힌 것이다. 규칙은 MimicGen
D1 범위의 중심을 두고 x와 y 폭을 두 배로 하되 탁자 중심에서 ±0.35 m를 넘지 않게 자르는 것이고, 회전 범위는 D1 값이다. Three Piece
Assembly만 MimicGen D2 범위를 그대로 쓴다. 그 위에 모든 물체가 로봇 밑판에서 0.85 m 안에 오도록 배치를 다시 뽑는 조건이 있다. 몇 과제는
물체가 로봇 받침대나 손가락 위치에 겹치던 쪽 경계를 D1 값으로 되돌렸다(파일의 주석에 과제별로 적혀 있다). 같은 범위가 생성과 평가에 함께
쓰이므로 검증·시험 장면도 이 분포에서 나온다. 환경 클래스 이름은 `<과제 접두어>_OURS`(예: `Square_OURS`)다.

관측은 영상이다. 카메라는 `agentview`와 `robot0_eye_in_hand` 두 대, 84×84 색상 영상이고 생성 중에 그려 풀에 저장한다. 로봇 상태는
말단 위치(3), 말단 회전 사원수(4), 집게 관절(2)이다. 두 정책 모두 학습 때 영상을 76×76으로 무작위로 자른다(MimicGen 공식 영상 정책의
자르기와 같다).

정책 둘의 설정은 다음과 같다. 학습 갱신 수와 배치 크기는 `configs/experiment/training/default.yaml`, 나머지는
`configs/experiment/policy/<정책>.yaml`에 있다.

| 항목 | Diffusion Policy (`diffusion_policy`) | BC-RNN (`bc_rnn`) |
|---|---|---|
| 구현 | LeRobot 0.6.1의 DiffusionPolicy. 공식 영상 U-Net 값(채널 512/1024/2048, DDPM 100걸음, 엡실론 예측, GroupNorm ResNet18을 처음부터 학습, 공간 소프트맥스 32점) | robomimic의 BC_RNN_GMM을 MimicGen 공식 영상 설정 값으로 옮긴 것(`tmg/policy/bc_rnn.py`) |
| 관측 걸음 수 / 예측 길이 / 실행 걸음 수 | 2 / 16 / 8 | 순차 길이 10 (학습과 실행에서 같은 길이의 은닉 상태) |
| 망 크기 | 위와 같음 | LSTM 은닉 1000, 2층, 혼합 정규분포 5개, 공간 소프트맥스 32점, 시각 특징 64 |
| 최적화 | Adam 1e-4, 코사인 일정과 예열 500걸음, 지수이동평균 가중치로 평가 | Adam 1e-4 |
| 갱신 수 / 배치 | 100,000 / 64 | 100,000 / 16 |
| 체크포인트 | 10,000갱신마다 (10개) | 같음 |
| 기울기 노름 자르기 | 1000 | 1000 |

평가는 고정 장면으로 한다. 과제마다 검증 25장면과 시험 300장면의 초기 배치를 시드 202600으로 미리 뽑아 파일로 두고(`resets/`), 모든
정책, 시드가 같은 장면을 쓴다. 체크포인트 10개를 모두 검증 25장면에서 평가해 성공률이 가장 높은 것(동률이면 이른 것)을 고르고, 그 하나를
시험 300장면에서 평가한다. 한 에피소드의 상한은 600걸음이다(2026-10-27 결정, MimicGen 공식값 400 또는 500과 다르다). 학습 시드는
202601, 202602, 202603이고 표에는 세 시드의 평균과 표준편차를 적는다. 한 시드의 300장면은 성공률 70% 근처에서 95% 구간 약 ±5.2
포인트다.

## 2. 요구 사항

- 운영체제. Ubuntu 22.04 또는 24.04를 기준으로 적었다.
- NVIDIA GPU 한 장. 학습과 정책 서버가 GPU를 쓴다. 영상 Diffusion Policy 학습 하나가 약 10 GB 안쪽을 쓰고, 기본 설정은 학습·평가
  파이프라인을 두 개 동시에 돌린다(`POLICY_PARALLEL=2`). 메모리 24 GB 이상을 권한다.
- NVIDIA 드라이버 570 이상(학습 이미지가 CUDA 12.8 기반이다). `nvidia-smi` 출력의 "Driver Version"으로 확인한다.
- CPU 코어. 생성과 평가 롤아웃은 CPU에서 돈다(robosuite, MuJoCo, 소프트웨어 렌더링). 묶음 하나가 코어 하나를 쓰므로 코어가 많을수록
  풀이 빨리 끝난다. 머신 설정의 `max_workers`를 코어 수에 맞춘다(3.5절). 16코어면 돌지만 풀 생성에 며칠이 걸린다(8절).
- Docker 24 이상과 NVIDIA Container Toolkit. 설치 방법은 3절에 있다.
- 디스크 약 400 GB. 이미지 두 개가 약 16 GB, 영상 풀이 성공 시도 하나에 약 13 MB(일곱 과제 합계 약 150 GB), 데이터셋이 과제당 약 13 GB,
  체크포인트가 실행(과제×정책×시드) 하나에 약 3 GB로 42개 실행에 약 130 GB다. 이 수치는 추정이고 3.7절의 확인 실행에서 실제 크기를 볼 수 있다.
- 인터넷. 이미지 빌드에 GitHub와 PyPI, 원본 시연 내려받기에 Hugging Face가 필요하다.
- 호스트에 git, make, Python 3이 있어야 한다(결과 묶음 스크립트는 표준 라이브러리만 쓴다). 그 밖의 모든 것은 이미지 안에서 돈다.
- 저장소 `https://github.com/JiH00nKw0n/TrackMixGen`은 공개 저장소라서 GitHub 계정 없이 받을 수 있다.

## 3. 준비

처음 받는 컴퓨터에서 아래 순서를 그대로 따른다. 이미 갖춘 단계는 건너뛴다. `sudo` 권한이 필요한 단계는 3.1부터 3.3까지다.

### 3.1 드라이버와 GPU 확인

```bash
nvidia-smi
```

GPU 이름과 "Driver Version"이 나와야 한다. 570보다 낮으면 컴퓨터 관리자에게 드라이버 갱신을 맡긴다.

### 3.2 git, make, Docker 설치와 도커 그룹

```bash
sudo apt-get update && sudo apt-get install -y git make python3 ca-certificates curl
```

Docker는 공식 저장소에서 설치한다. 다음 명령은 Docker 문서의 Ubuntu 설치 절차를 그대로 옮긴 것이다.

```bash
sudo install -m 0755 -d /etc/apt/keyrings && sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && sudo chmod a+r /etc/apt/keyrings/docker.asc
```

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

```bash
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
```

현재 사용자가 `sudo` 없이 도커를 쓰도록 그룹에 넣는다. 이 뒤에 한 번 로그아웃하고 다시 로그인해야 적용된다.

```bash
sudo usermod -aG docker $USER
```

다시 로그인한 뒤 다음이 오류 없이 돌아야 한다.

```bash
docker run --rm hello-world
```

### 3.3 NVIDIA Container Toolkit 설치

컨테이너 안에서 GPU를 쓰게 하는 구성 요소다. 다음 명령은 NVIDIA 문서의 설치 절차를 그대로 옮긴 것이다.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
```

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

컨테이너에서 GPU가 보이는지 확인한다. 호스트의 `nvidia-smi`와 같은 표가 나와야 한다.

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### 3.4 저장소 받기

```bash
git clone https://github.com/JiH00nKw0n/TrackMixGen.git ~/TrackMixGen
```

이 문서의 명령은 모두 저장소 폴더 안에서 실행한다.

```bash
cd ~/TrackMixGen
```

### 3.5 데이터 폴더 만들기와 머신 설정

데이터(원본 시연, 풀, 데이터셋, 체크포인트, 결과)는 모두 데이터 폴더 하나 아래에 쓰인다. 디스크 여유가 가장 많은 곳에 만든다.

```bash
mkdir -p /data/tmg_data
```

머신 설정 파일을 하나 만든다. 이름은 자유이고(여기서는 `lab`), 아래 명령의 `CLUSTER=lab`이 이 파일을 가리킨다. `data_root`는 위에서
만든 폴더, `max_workers`는 동시에 돌릴 생성 묶음 수다(CPU 코어 수에서 4 정도를 뺀 값을 권한다. 평가 롤아웃도 CPU를 쓴다).

```bash
cat > configs/machine/cluster/lab.yaml <<'EOF'
name: lab
runtime: docker
image: tmg/robosuite:latest
sif: null
data_root: /data/tmg_data
workers: auto
max_workers: 28
threads_per_worker: 1
min_free_disk_gib: 20
EOF
```

### 3.6 이미지 두 개 빌드

실험은 도커 이미지 두 개로 돈다. 파이썬과 torch 판이 달라 하나로 합치지 않는다.

| 이미지 | 도커 파일 | 내용 | 쓰는 단계 |
|---|---|---|---|
| `tmg/robosuite` | `docker/robosuite.Dockerfile` | Python 3.10, MuJoCo 2.3.2, robosuite 1.4.1, MimicGen 72bd767(공식 생성기, 고정 커밋), robosuite_task_zoo(Hammer Cleanup의 서랍 자산), Mesa 소프트웨어 렌더링. GPU를 쓰지 않는다 | 원본 시연, 생성, 고정 장면, 데이터셋, 평가 롤아웃 |
| `tmg/train` | `docker/train.Dockerfile` | Python 3.12, torch 2.8(CUDA 12.8), LeRobot 0.6.1 | 학습, 체크포인트 선택, 정책 서버 |

```bash
make build IMAGE=robosuite
```

```bash
make build IMAGE=train
```

각각 10에서 30분이 걸린다(네트워크 속도에 따라 다르다). 끝나면 `docker images`에 `tmg/robosuite`와 `tmg/train`이 보인다. 두 이미지는
저장소의 코드(`tmg/`, `configs/`)를 실행 때 마운트해서 쓰므로, 코드만 바뀌었을 때는 다시 빌드할 필요가 없다. 의존 패키지
목록(`requirements/`)이나 도커 파일이 바뀌었을 때만 다시 빌드한다.

### 3.7 동작 확인

단위 시험을 이미지 안에서 돌린다. 모두 통과해야 한다.

```bash
make test IMAGE=robosuite CLUSTER=lab
```

그 다음 축소판 실험(`MINI=1`)을 과제 하나로 끝까지 돌린다. 200회 시도 풀, 시연 10개 데이터셋, 두 정책을 200갱신만 학습, 검증과 시험을
6장면씩 돌린다. 실제 실험과 같은 코드 경로와 파일 배치를 쓰며 Coffee 기준 약 한 시간이 걸린다(8절).

```bash
CLUSTER=lab GPU=0 MINI=1 TASKS=coffee SEEDS=202601 scripts/run_mimicgen_baseline.sh
```

마지막에 `done`이 찍히고 `/data/tmg_data/logs/mimicgen_baseline/driver.log`에 `RESULT` 줄이 정책마다 하나씩 나오면 준비가 끝났다.
`FAILURES` 파일이 생겼으면 9절을 본다. 축소판의 풀과 실행 폴더 이름에는 `pilot50`이 들어가므로 뒤의 전체 실행(`pool_robosuite_image`)과
파일이 섞이지 않는다.

## 4. 실행

전체 실행은 명령 하나다. 터미널이 닫혀도 돌도록 `nohup`으로 띄운다.

```bash
cd ~/TrackMixGen && CLUSTER=lab GPU=0 nohup scripts/run_mimicgen_baseline.sh > /data/tmg_data/logs/mimicgen_baseline_nohup.log 2>&1 &
```

스크립트는 두 줄기를 동시에 돈다. 생성 줄기는 과제마다 원본 시연, 풀, 고정 장면, 데이터셋을 만들고(`POOL_LANES=2`, 두 과제씩), 정책
줄기는 데이터셋이 생긴 과제부터 과제×정책×시드 42개의 학습·평가 파이프라인을 `POLICY_PARALLEL=2`개씩 돈다. 자주 바꾸는 변수는
다음과 같다.

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `CLUSTER` | `local` | 3.5절의 머신 설정 이름 |
| `GPU` | `0` | 학습과 정책 서버가 쓰는 CUDA 장치 번호 |
| `TASKS` | 일곱 과제 | 돌릴 과제 목록(공백으로 구분). 긴 풀부터 적혀 있다 |
| `POLICIES` | `diffusion_policy bc_rnn` | 돌릴 정책 |
| `SEEDS` | `202601 202602 202603` | 학습 시드 |
| `POOL_LANES` | `2` | 동시에 생성하는 과제 수. 각 풀은 `max_workers`개의 묶음 프로세스를 쓴다 |
| `POLICY_PARALLEL` | `2` | GPU에서 동시에 돌리는 학습·평가 파이프라인 수 |
| `VAL_PARALLEL`, `TEST_SHARDS` | `4`, `6` | 한 실행 안에서 동시에 검증하는 체크포인트 수, 시험 300장면의 병렬 분할 수 |
| `ONLY_GENERATION` | 없음 | `1`이면 데이터셋까지만 만들고 끝낸다 |

진행 상황은 드라이버 로그로 본다.

```bash
tail -f /data/tmg_data/logs/mimicgen_baseline/driver.log
```

단계마다 로그 파일이 따로 있다(`pool_<과제>.log`, `dataset_<과제>.log`, `policy_<과제>_<정책>_seed<시드>.log`). 실패한 단계는 같은
폴더의 `FAILURES`에 한 줄씩 적히고 스크립트는 나머지 작업을 계속한다. 모두 끝나면 `DONE`이 생긴다.

중단되었거나 실패한 단계를 고친 뒤에는 같은 명령을 다시 실행한다. 끝난 풀 묶음, 데이터셋, 학습(`train_result.json`이 있는 실행), 시험
결과가 있는 실행은 건너뛴다. 학습이 중간에 끊긴 실행은 처음부터 다시 학습한다.

## 5. 결과 파일

실험이 끝나면 정책마다 결과 묶음이 `/data/tmg_data/reports/mimicgen_baseline_<날짜>_<정책>/`과 같은 이름의 `.tar.gz`로 만들어진다.
묶음에는 모델 가중치와 영상이 들어가지 않으므로 수백 MB 안쪽이고, 분석에 필요한 것은 모두 들어 있다.

```
summary.json, summary.md              과제별 생성 성공률, 시드별 검증·시험 성공률, 세 시드의 평균과 표준편차
mimicgen/<과제>/ours/none/pool_robosuite_image/image/
  generation/DONE.json                풀 요약: 시도 수, 성공 수, 묶음별 결과
  generation/attempts.jsonl           모든 시도 한 줄씩: 성공 여부, 고른 원본 시연 번호, 하위작업 기록, 걸린 시간
  generation/chunk_logs/              묶음 프로세스 로그
  dataset/*.json, *.yaml              데이터셋에 들어간 시도 목록과 설정
  resets.json                         검증 25장면과 시험 300장면의 초기 배치
  seed<시드>/train_log.jsonl           학습 손실 기록
  seed<시드>/validation.jsonl          체크포인트별 검증 성공률
  seed<시드>/SELECTED.json             고른 체크포인트
  seed<시드>/eval/test_h600/<체크포인트>/eval_summary.json, episodes.jsonl   시험 결과 요약과 에피소드별 기록
driver_logs/                          드라이버 로그
```

한 실행의 시험 결과는 드라이버 로그의 `RESULT` 줄로도 볼 수 있다. 예를 들어 `RESULT coffee 0080000 horizon 600: test success
187/300 = 0.623 (95% CI 0.567-0.676)`는 Coffee에서 8만 번째 체크포인트가 시험 300장면 중 187번 성공했다는 뜻이다. 원본 데이터(풀,
데이터셋, 체크포인트)는 데이터 폴더의 `pools/`, `datasets/`, `runs/`에 그대로 남는다.

## 6. 난수와 재현성

| 시드 | 값 | 쓰이는 곳 |
|---|---|---|
| 생성 묶음 시드 | 30001, 30002, ... (묶음 번호 순) | MimicGen의 `experiment.seed`. 같은 묶음은 같은 시드로 다시 만들어도 같은 시도를 낸다 |
| 고정 장면 시드 | 202600 | 검증 25장면과 시험 300장면의 초기 배치 |
| 학습 시드 | 202601, 202602, 202603 | 모형 초기화, 데이터 순서, 영상 자르기 위치 |
| 데이터셋 시드 | 0 | 성공 시도 중 1,000개를 고르는 순서 (`n1000_seed0.hdf5`) |
| 정책 평가 시드 | 평가 시드, 학습 시드, 장면 번호에서 유도 | Diffusion Policy의 잡음 표본 |

학습은 GPU의 비결정성 때문에 같은 시드라도 비트 단위로 같지 않을 수 있다. 생성과 평가 장면은 결정적이다.

## 7. 단계별 수동 실행

드라이버가 하는 일을 한 단계씩 손으로 돌릴 때 쓴다. 공통 변수는 `GENERATOR=mimicgen IMAGE=robosuite CLUSTER=lab SCENE=ours METHOD=none
GENERATION=pool_robosuite_image OBSERVATION=image`다.

```bash
make sources GENERATOR=mimicgen TASK=coffee CLUSTER=lab
```

```bash
make generate GENERATOR=mimicgen IMAGE=robosuite TASK=coffee SCENE=ours METHOD=none GENERATION=pool_robosuite_image CLUSTER=lab OVERRIDES="generation.attempts=2500"
```

```bash
make resets GENERATOR=mimicgen TASK=coffee SCENE=ours CLUSTER=lab
```

```bash
make dataset GENERATOR=mimicgen TASK=coffee SCENE=ours METHOD=none GENERATION=pool_robosuite_image OBSERVATION=image CLUSTER=lab OVERRIDES="size=1000"
```

```bash
make train GENERATOR=mimicgen TASK=coffee SCENE=ours METHOD=none GENERATION=pool_robosuite_image OBSERVATION=image POLICY=diffusion_policy SEED=202601 CLUSTER=lab OVERRIDES="dataset.size=1000"
```

학습 뒤의 검증, 선택, 시험은 한 실행을 통째로 맡는 스크립트가 있다. 포트 블록(여기서는 8800부터 40개)은 다른 실행과 겹치지 않게 준다.

```bash
CLUSTER=lab GPU=0 GENERATOR=mimicgen IMAGE=robosuite METHOD=none GENERATION=pool_robosuite_image OBSERVATION=image POLICY=diffusion_policy SEED=202601 SKIP_TRAIN=1 scripts/task_policy_pipeline.sh coffee 8800
```

합성한 설정을 눈으로 확인하려면 `make show`를 쓴다.

```bash
make show GENERATOR=mimicgen TASK=coffee SCENE=ours METHOD=none GENERATION=pool_robosuite_image CLUSTER=lab
```

## 8. 측정한 시간과 예상 시간

측정값은 ADS A100 서버(CPU 255코어)에서 2026-10에 저차원 풀(영상을 그리지 않음)을 만들 때 묶음 프로세스 하나가 시도 하나에 쓴 시간이다.
영상 풀은 같은 시도에 두 카메라를 84×84로 그리므로 더 걸리는데 그 배수는 측정하지 않았다(저차원 설정 파일의 주석은 약 1.6배로 적고 있다).

| 과제 | 시도당 시간 (저차원, 프로세스 하나) | 처음 시도 수 | 28프로세스로 영상 풀에 걸리는 시간 (1.6배 가정, 추정) |
|---|---|---|---|
| Coffee | 6.3초 | 2,500 | 약 15분 |
| Threading | 4.4초 | 6,250 | 약 26분 |
| Hammer Cleanup | 6.1초 | 7,000 | 약 41분 |
| Mug Cleanup | 9.1초 | 8,750 | 약 1시간 16분 |
| Stack Three | 6.8초 | 9,500 | 약 1시간 1분 |
| Square | 5.2초 | 15,000 | 약 1시간 14분 |
| Three Piece Assembly | 11.2초 | 17,000 | 약 3시간 1분 |

학습과 평가 시간은 영상 정책으로 측정한 값이 없다. 저차원 BC-RNN은 A100에서 10만 갱신에 약 35분이었다(1만 갱신마다 약 3.5분). 영상
Diffusion Policy는 ResNet18 인코더 두 개와 U-Net을 처음부터 학습하므로 이보다 훨씬 길고, 다른 과제군(Isaac Lab, 128×128 영상)에서는
10만 갱신을 4에서 6시간으로 추정했다. 평가는 체크포인트 10개 × 검증 25장면과 시험 300장면이고 에피소드 하나가 상한 600걸음이다. 3.7절의
축소판 실행이 이 컴퓨터의 실제 속도를 알려 준다. 축소판 로그의 학습 시간(200갱신)을 500배 하면 10만 갱신 시간의 어림이 된다.

전체 실험은 42개 실행(7과제 × 2정책 × 3시드)이다. 실행 하나가 학습 3시간과 평가 1시간이면, 두 개씩 돌릴 때 약 4일이다.

## 9. 알려진 제한과 문제 해결

- `docker: permission denied`. 3.2절의 도커 그룹 등록 뒤 다시 로그인하지 않은 경우다.
- `only N GiB free under ...; cluster.min_free_disk_gib is 20`. 데이터 폴더의 디스크 여유가 20 GB 아래다. 공간을 비우고 같은 명령을 다시 실행한다.
- `finished chunks differ from the requested configuration`. 끝난 풀 묶음이 지금 설정과 다른 설정으로 만들어졌다는 뜻이다. 설정 파일을
  바꾼 뒤에 같은 풀을 이어 돌리면 난다. 의도한 변경이면 `OVERRIDES="generation.on_config_change=rebuild"`를 주어 다시 만들고, 아니면
  설정을 되돌린다.
- 풀의 성공 수가 1,000개에 못 미친다. 드라이버가 시도 수를 25%씩 늘려 이어 돌리므로 그대로 두면 된다. 로그의 `pool: N successes in M
  attempts` 줄로 진행을 본다.
- 정책 서버 포트. 드라이버는 8800번부터 실행 하나에 40개씩 포트를 쓴다. 다른 프로그램이 그 범위를 쓰면 파이프라인이 그 포트를 건너뛰고
  다음 포트를 쓴다. 평가 중에 중단되면 `tmg-serve-<포트>` 이름의 컨테이너가 남을 수 있다. `docker ps`로 보고 `docker stop`으로 멈춘다.
- Hammer Cleanup은 서랍 자산을 robosuite_task_zoo에서 읽는다. 이미지에 들어 있으므로 따로 할 일은 없다.
- Mug Cleanup은 MimicGen이 실행 때 머그잔 자산 폴더에 xml 파일을 쓴다. 이미지 빌드가 그 폴더에 쓰기 권한을 주므로 따로 할 일은 없다.
- 영상 렌더링은 Mesa 소프트웨어 렌더링이라 GPU를 쓰지 않는다. 그래서 생성과 평가 속도는 CPU 코어 수가 정한다.
- 저차원 관측(`OBSERVATION=low_dim`)으로 돌리려면 영상 없는 풀(`GENERATION=pool_robosuite_lowdim`)을 따로 만들어야 하며, 그 설정은
  탐색용 원본 선택 규칙(하위작업별 무작위)을 쓰므로 이 문서의 기준선과 다르다. 기준선은 영상으로만 돌린다.
