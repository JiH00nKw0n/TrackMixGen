# Isaac Lab 기준선 실행 안내 (SoftMimicGen, DexMimicGen)

이 문서 하나로 GPU 한 장짜리 컴퓨터에서 SoftMimicGen과 DexMimicGen의 기준선 실험을 처음부터 끝까지 돌리고, 분석에 필요한 결과 파일을
한 묶음으로 받을 수 있다. 실험은 과제마다 "공식 원본 시연 내려받기, 공식 설정으로 성공 시연 1,000개 생성, 고정 평가 장면 만들기,
학습 데이터셋 1,000개 쓰기, 영상 Diffusion Policy를 시드 3개(202601, 202602, 202603)로 학습, 시드마다 검증 25장면으로 체크포인트
선택, 시험 300장면 평가" 순서로 진행한다. 명령 하나(`scripts/run_isaaclab_baseline.sh`)가 전부를 돌리고, 중단되면 같은 명령을 다시
실행하면 끝난 단계는 건너뛰고 이어서 돈다.

## 차례

1. 돌리는 과제와 공식 설정. 네 과제의 생성기 설정값과 원본 시연 수를 적는다.
2. 요구 사항. 운영체제, 드라이버, 도커, 디스크, 네트워크 조건을 적는다.
3. 준비. 드라이버 확인, git과 Docker 설치, NVIDIA Container Toolkit 설치, 저장소 받기, 데이터 폴더와 머신 설정, 이미지 네 개 빌드, 동작 확인을 순서대로 적는다.
4. 실행. 전체 실행 명령과 진행 상황을 보는 방법, 중단 뒤 재시작 방법을 적는다.
5. 결과 파일. 결과 묶음의 폴더 구조와 각 파일이 담는 내용, 수치를 읽는 방법을 적는다.
6. 난수와 재현성. 어떤 시드가 어디에 쓰이고 무엇이 같게 유지되는지 적는다.
7. 단계별 수동 실행. 문제를 좁힐 때 쓰는 단계별 명령을 적는다.
8. 측정한 시간과 예상 시간. Rope 시범 실행에서 측정한 값과 거기서 추정한 전체 실험 시간을 구분해 적는다.
9. 알려진 제한과 문제 해결. 돌릴 수 없는 과제, 자주 나는 오류와 대처를 적는다.

## 1. 돌리는 과제와 공식 설정

논문 계획의 다섯 과제는 SoftMimicGen의 Rope(밧줄을 U자로 만들기), Towel(수건 접기)와 DexMimicGen의 GR1 Pick Place(사람형 로봇이
물체를 집어 상자에 넣기), Microwave Opening(전자레인지 문 열기), Bottle Placement and Door Closing(병을 냉장고에 넣고 문 닫기)이다.
실행 스크립트는 다섯 과제를 모두 돈다. 뒤의 두 과제는 NVIDIA AutoData(Isaac Lab-Arena 위의 DexMimicGen)로 돌며 네 번째 이미지
`isaaclab-autodata`가 필요하다(3.6절). Jenga 과제 설정도 있지만 계획에 없으므로 기본 목록에서 뺐다(`SOFT_TASKS`로 넣을 수 있다).

생성기 설정은 공식 환경 설정 파일의 값을 그대로 쓴다. 우리 코드는 공식 생성 스크립트에 시도 수, 시드, 입출력 파일만 넘기고 나머지는
건드리지 않는다. 아래 표의 값은 공식 소스에서 읽은 것이다.

| 과제 | 원본 시연 수 | 한 시도의 모든 하위작업이 같은 원본을 쓰는가 | 원본 선택 방식 | 행동 잡음 | 하위작업별 보간 걸음 수 |
|---|---|---|---|---|---|
| Soft Rope | 3 | 그렇다 | 변형체 정합 비용이 가장 낮은 3개 중 무작위 (registration_cost) | 0.0 | 10, 5 |
| Soft Towel | 내려받을 때 매니페스트에 기록 | 그렇다 | 같음 | 0.01 | 5, 5 |
| Dex GR1 Pick Place | 5 | 그렇다 | nearest_neighbor_object | 0.003 | 오른손 0, 3 / 왼손 0 |
| Dex GR1 Microwave Opening (AutoData) | 10 | 아니다 (하위작업마다 고른다) | nearest_neighbor_object | 0.005 | 오른손 5, 5 / 왼손 0 |
| Dex GR1 Bottle Placement and Door Closing (AutoData) | 5 | 아니다 (하위작업마다 고른다) | nearest_neighbor_object | 0.003 | 오른손 5, 5, 5, 5 / 왼손 0, 0 |

Isaac Lab 세 과제는 공식 설정이 "성공 수 보장"(generation_guarantee)이어서, 생성 설정 `pool_isaac`의 시도 수 1,000은 성공 1,000개가
나올 때까지 돈다는 뜻이다(성공 250개짜리 청크 4개). AutoData 두 과제의 공식 기술 파일은 성공을 보장하지 않지만, 우리 청크는 같은 뜻이
되도록 사본에 `guarantee_success: true`를 써 넣는다. 실패 시도도 모두 기록되므로 생성 성공률은 기록에서 계산한다.

정책은 영상 Diffusion Policy 하나다(`configs/experiment/policy/diffusion_policy.yaml`, 10만 갱신, 배치 64, 체크포인트 1만 갱신마다).
관측은 과제의 카메라 두 대(128×128)와 로봇 상태다. 세 과제 모두 영상을 자르지 않는다(SoftMimicGen 공식 Diffusion Policy 설정과
같고, MimicGen 과제의 공식 76×76 자르기는 그쪽에만 남겨 둔다). 평가 상한은 Rope 250걸음, Towel 220걸음(공식 롤아웃 상한),
Pick Place 400걸음(환경 자체 길이)이다. 체크포인트 선택은 검증 25장면 성공률이 가장 높은 것(동률이면 이른 것)이다.

## 2. 요구 사항

- 운영체제. Ubuntu 22.04 또는 24.04를 기준으로 적었다. 아래 명령은 모두 이 두 버전에서 그대로 쓸 수 있다.
- NVIDIA GPU 한 장. 이 문서는 RTX PRO 6000 Ada(메모리 48 GB)를 기준으로 동시 실행 개수를 정했다. Isaac Sim 한 프로세스가 GPU
  메모리를 약 5에서 14 GB 쓰고, 학습 프로세스가 약 10 GB를 쓴다.
- NVIDIA 드라이버 570 이상(이미지가 CUDA 12.8 기반이다). `nvidia-smi` 출력의 "Driver Version"으로 확인한다.
- Docker 24 이상과 NVIDIA Container Toolkit. 설치 방법은 3절에 있다.
- 디스크 약 300 GB. 이미지 세 개가 약 60 GB, 네 과제의 풀(영상 포함)과 데이터셋이 과제당 약 20에서 40 GB, 학습 체크포인트가
  시드당 약 5 GB다.
- 인터넷. 이미지 빌드에 GitHub, PyPI, NVIDIA의 pip 저장소(pypi.nvidia.com)가 필요하고, 원본 시연 내려받기에 Hugging Face(Soft)와
  Isaac 자산 서버 omniverse-content-production.s3-us-west-2.amazonaws.com(Dex)이 필요하다. Isaac Sim은 첫 실행 때 자산을
  자산 서버에서 받는다.
- 호스트에 git, make, Python 3이 있어야 한다(결과 묶음 스크립트는 표준 라이브러리만 쓴다). 그 밖의 모든 것은 이미지 안에서 돈다.
- 저장소 `https://github.com/JiH00nKw0n/TrackMixGen`은 공개 저장소라서 GitHub 계정 없이 받을 수 있다.

## 3. 준비

처음 받는 컴퓨터에서 아래 순서를 그대로 따른다. 이미 갖춘 단계는 건너뛴다. `sudo` 권한이 필요한 단계는 3.1부터 3.3까지다.

### 3.1 드라이버와 GPU 확인

```bash
nvidia-smi
```

GPU 이름과 "Driver Version"이 나와야 한다. 570보다 낮으면 Ubuntu의 드라이버 저장소에서 올린다(예를 들어
`sudo apt install nvidia-driver-570` 뒤 재부팅). 드라이버 설치는 이 문서의 범위 밖이므로 컴퓨터 관리자에게 맡긴다.

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

작업 폴더는 어디든 된다. 아래는 홈 폴더 기준이다.

```bash
cd ~ && git clone https://github.com/JiH00nKw0n/TrackMixGen.git && cd TrackMixGen
```

이 문서가 맞는 버전인지 확인한다. 이 문서와 함께 넘긴 커밋 번호가 있으면 그 커밋으로 맞춘다(없으면 `main`의 최신을 쓴다).

```bash
git log --oneline -1
```

```bash
git checkout <넘겨받은 커밋 번호>
```

이 뒤의 모든 명령은 저장소 최상위 폴더(`~/TrackMixGen`)에서 실행한다.

### 3.5 데이터 폴더 만들기와 머신 설정

실험 자료가 모두 들어갈 폴더를 만들고 현재 사용자 소유로 둔다(컨테이너는 현재 사용자 권한으로 돌아서 이 폴더에 쓴다). 기본 경로는
`/data/tmg_data`이고, 다른 곳을 쓰려면 아래 두 곳의 경로를 함께 바꾼다.

```bash
sudo mkdir -p /data/tmg_data && sudo chown -R $USER:$USER /data/tmg_data
```

`configs/machine/cluster/ada.yaml`의 `data_root`를 그 경로로 맞춘다(기본값이 `/data/tmg_data`이므로 기본 경로를 쓰면 바꿀 것이 없다).
같은 파일의 `workers`(동시에 돌 생성 프로세스 수, 기본 2)와 `device`(기본 `cuda:0`)도 여기서 바꾼다. 설정이 읽히는지 확인한다.
아래 명령이 `/data/tmg_data`를 출력해야 한다.

```bash
make print-data-root CLUSTER=ada
```

### 3.6 이미지 네 개 빌드

저장소 최상위에서 실행하며, 각각 1에서 2시간 걸리고 인터넷이 필요하다. 코드와 설정은 실행 때 컨테이너에 마운트되므로 저장소를
갱신해도 다시 빌드하지 않는다(의존 패키지가 바뀔 때만 다시 빌드한다).

```bash
make build IMAGE=isaaclab-soft
```

```bash
make build IMAGE=isaaclab-dex
```

```bash
make build IMAGE=train
```

```bash
make build IMAGE=isaaclab-autodata
```

끝나면 `docker images`에 `tmg/isaaclab-soft`, `tmg/isaaclab-dex`, `tmg/isaaclab-autodata`, `tmg/train`이 보여야 한다. 네 번째 이미지는
Isaac Sim 6.1 기반이라 가장 크고(약 30 GB) 오래 걸린다.

### 3.7 동작 확인

아래 두 명령은 Rope의 원본 3개를 내려받고, 성공 20개짜리 시범 생성을 한 번 돈다. 끝나면
`/data/tmg_data/pools/softmimicgen/rope/ours/none/pilot20/DONE.json`이 생기고 그 안에 시도 수와 성공 수가 적힌다.

```bash
make sources GENERATOR=softmimicgen IMAGE=isaaclab-soft TASK=rope CLUSTER=ada
```

```bash
make generate GENERATOR=softmimicgen IMAGE=isaaclab-soft TASK=rope SCENE=ours METHOD=none GENERATION=pilot20 CLUSTER=ada
```

첫 실행은 Isaac Sim이 자산을 내려받고 셰이더를 컴파일하느라 10분 넘게 걸릴 수 있다. 로그는
`/data/tmg_data/pools/softmimicgen/rope/ours/none/pilot20/logs/`에 있다. 여기까지 되면 4절의 전체 실행으로 넘어간다.

## 4. 실행

전체 실험은 명령 하나다. 며칠 걸리므로 `nohup`으로 띄우고 터미널을 닫아도 되게 한다.

```bash
CLUSTER=ada nohup scripts/run_isaaclab_baseline.sh > /dev/null 2>&1 &
```

스크립트는 두 줄기를 동시에 돈다. 생성 줄기는 과제를 Rope, Towel, Pick Place, Microwave, Bottle Door 순서로 하나씩 처리하고(원본, 풀, 고정 장면,
데이터셋), 정책 줄기는 데이터셋이 생긴 과제부터 시드 3개를 차례로 학습하고 평가한다. 그래서 한 과제의 학습이 도는 동안 다음 과제의
생성이 같은 GPU에서 함께 돈다.

아래에서 `<data_root>`는 3.5절에서 정한 데이터 폴더(기본 `/data/tmg_data`)다. 진행 상황은 `<data_root>/logs/isaaclab_baseline/driver.log`에 단계별로 적힌다. 단계별 상세 로그는 같은 폴더의
`sources_*.log`, `pool_*.log`, `resets_*.log`, `dataset_*.log`, `policy_<생성기>_<과제>_seed<시드>.log`다. 실패한 단계는
`FAILURES` 파일에 한 줄씩 쌓인다. 전부 끝나면 `DONE` 파일이 생기고 결과 묶음(5절)이 만들어진다.

중단되면 같은 명령을 다시 실행한다. 끝난 풀의 청크, 데이터셋, 체크포인트, 체크포인트별 검증 결과, 시험 결과는 모두 보존되어 건너뛴다.
학습 중에 멈춘 실행은 마지막 체크포인트에서 이어서 학습한다.

과제나 시드를 바꾸려면 환경 변수로 준다. 예를 들어 Rope와 Pick Place만 돌리려면 다음과 같이 한다.

```bash
CLUSTER=ada SOFT_TASKS="rope" DEX_TASKS="pick_place" nohup scripts/run_isaaclab_baseline.sh > /dev/null 2>&1 &
```

GPU 메모리가 부족하면 `VAL_PARALLEL`(동시에 검증하는 체크포인트 수, 기본 2)과 `TEST_SHARDS`(시험 300장면을 나눠 도는 수, 기본 3)를
줄인다. 각각 하나가 Isaac Sim 프로세스 하나다. 생성만 먼저 끝내고 싶으면 `ONLY_GENERATION=1`을 준다.

## 5. 결과 파일

실험이 끝나면 `<data_root>/reports/isaaclab_baseline_<날짜>/`와 같은 이름의 `.tar.gz`가 생긴다. 이 묶음만 받으면 후속 분석을 할 수
있다. 모델 가중치와 영상은 들어 있지 않다. 묶음을 따로 다시 만들려면 `scripts/collect_results.py`를 실행한다(`--help`에 인자가 있다).

| 파일 | 내용 |
|---|---|
| `summary.md`, `summary.json` | 과제별 생성 성공률, 시드별 검증한 체크포인트 수, 최고 검증 성공률, 선택된 체크포인트, 시험 성공률과 95% 구간, 시드 3개의 평균과 표준편차. |
| `<생성기>/<과제>/generation/DONE.json` | 풀 요약. 전체 시도 수, 성공 수, 청크별 결과, 청크 계획(시드, 시도 수). |
| `<생성기>/<과제>/generation/attempts.jsonl` | 모든 시도가 한 줄씩. 성공 여부, 선택된 원본 시연 번호, 하위작업별 기록(원본 번호, 변환된 자세), 걸린 시간, 청크 이름. 생성 성공률과 원본별 성공률, 변환 거리별 성공률은 이 파일에서 계산한다. |
| `<생성기>/<과제>/generation/chunk_logs/` | 청크 프로세스의 로그. |
| `<생성기>/<과제>/generation/resolved_config.yaml`, `provenance.json` | 풀을 만든 설정 전체와 코드 버전, 패키지 버전, 이미지 이름. |
| `<생성기>/<과제>/dataset/*.json`, `*.yaml` | 학습 데이터셋에 들어간 시도의 목록(청크, 시연 번호, 길이)과 설정. |
| `<생성기>/<과제>/resets/val25_test300_seed202600.json` | 고정 장면 325개의 번호, 분할(val 25, test 300), 장면 시드. |
| `<생성기>/<과제>/seed<시드>/train_log.jsonl` | 에폭(500 갱신)마다 손실과 경사 크기. |
| `<생성기>/<과제>/seed<시드>/validation.jsonl` | 체크포인트마다 검증 25장면의 성공 수와 성공률. |
| `<생성기>/<과제>/seed<시드>/SELECTED.json` | 선택된 체크포인트와 선택 규칙. |
| `<생성기>/<과제>/seed<시드>/eval/test_h<상한>/<체크포인트>/eval_summary.json` | 시험 300장면의 성공 수, 성공률, Wilson 95% 구간, 평균 걸음 수, 오류 수. |
| `<생성기>/<과제>/seed<시드>/eval/.../episodes.jsonl` | 에피소드마다 장면 번호, 분할, 성공 여부, 걸음 수, 장면 시드, 정책 시드, 리셋 직후 물체 자세(강체는 위치와 사원수, 변형체는 마디 평균 위치), 걸린 시간, 오류 메시지. 검증 분할(`val_h<상한>/`)도 같은 형식이다. |
| `<생성기>/<과제>/seed<시드>/checkpoints/<갱신 수>.json` | 체크포인트의 갱신 수, 에폭, 손실. |
| `driver_logs/` | 4절의 로그 전부. |

생성 성공률은 `attempts.jsonl`의 성공 줄 수를 전체 줄 수로 나눈 값이고, `summary.md`에 과제별로 적힌다. 정책 성공률은 시드마다 선택된
체크포인트의 `eval_summary.json`에 있는 시험 300장면 값이며, `summary.md`에는 시드별 값과 3개 시드의 평균, 표준편차가 함께 적힌다.

## 6. 난수와 재현성

- 생성 시드. 청크 k의 시드는 `seed_offset + k`(30001, 30002, 30003, 30004)이고 공식 스크립트의 `--seed`로 들어간다.
- 데이터셋 선택 시드. 풀의 성공 시도 가운데 1,000개를 고르는 시드는 0이다(`n1000_seed0.hdf5`).
- 고정 평가 장면. 장면 번호 i의 시드는 `SeedSequence([202600, i])`로 정해지고 파일에 적힌다. 롤아웃은 `env.reset()` 직전에 그 값으로
  환경, 토치, 넘파이, `random`을 모두 초기화한다. 그래서 어느 실행, 어느 정책, 어느 조건에서도 장면 번호 i의 초기 배치는 같다.
  `configs/experiment/evaluation/default.yaml`의 `seed`를 바꾸지 않는 한 모든 과제와 생성기가 같은 장면 시드 목록을 쓴다.
- 학습 시드. 202601, 202602, 202603이다. 정책 시드(에피소드마다 정책 서버에 주는 시드)는 `SeedSequence([202600, 학습 시드, 장면 번호])`다.
- 보장하지 않는 것. Isaac Sim의 물리 결과와 렌더링은 GPU 종류와 드라이버에 따라 달라질 수 있으므로, 같은 장면 시드가 같은 초기 배치를
  주더라도 같은 에피소드 결과를 다른 컴퓨터에서 그대로 재현한다고는 보장하지 않는다. 생성도 같다.

## 7. 단계별 수동 실행

문제를 좁힐 때는 단계를 따로 돈다. 변수는 `GENERATOR`(softmimicgen 또는 dexmimicgen), `IMAGE`(isaaclab-soft 또는 isaaclab-dex),
`TASK`(rope, towel, pick_place, microwave, bottle_door, 그리고 계획 밖의 jenga)다. AutoData 두 과제는 `GENERATOR=autodata IMAGE=isaaclab-autodata`로 돈다. 아래는 Rope 예다.

원본 시연을 내려받는다. 결과는 `<data_root>/sources/softmimicgen/rope/source.hdf5`와 매니페스트(원본 수, md5)다.

```bash
make sources GENERATOR=softmimicgen IMAGE=isaaclab-soft TASK=rope CLUSTER=ada
```

풀을 생성한다. 결과는 `<data_root>/pools/softmimicgen/rope/ours/none/pool_isaac/`이고 `DONE.json`이 생기면 끝이다.

```bash
make generate GENERATOR=softmimicgen IMAGE=isaaclab-soft TASK=rope SCENE=ours METHOD=none GENERATION=pool_isaac CLUSTER=ada
```

고정 평가 장면의 시드 파일을 쓴다(시뮬레이터를 띄우지 않는다).

```bash
make resets GENERATOR=softmimicgen IMAGE=isaaclab-soft TASK=rope SCENE=ours CLUSTER=ada
```

학습 데이터셋을 쓴다. 결과는 `<data_root>/datasets/softmimicgen/rope/ours/none/pool_isaac/image/n1000_seed0.hdf5`다.

```bash
make dataset GENERATOR=softmimicgen IMAGE=isaaclab-soft TASK=rope SCENE=ours METHOD=none GENERATION=pool_isaac OBSERVATION=image CLUSTER=ada OVERRIDES="size=1000"
```

한 시드의 학습, 체크포인트 검증, 선택, 시험을 한 번에 돈다. 두 번째 인자는 정책 서버가 쓸 포트 블록의 시작이다(40개를 쓴다).

```bash
CLUSTER=ada GPU=0 GENERATOR=softmimicgen IMAGE=isaaclab-soft GENERATION=pool_isaac OBSERVATION=image POLICY=diffusion_policy SEED=202601 SCENE=ours VAL_PARALLEL=2 TEST_SHARDS=3 scripts/task_policy_pipeline.sh rope 8800
```

체크포인트 하나만 검증하거나 시험하려면 `make validate`와 `make evaluate`를 쓴다. `RUN`은 학습 결과 폴더, `CKPT`는 그 아래 체크포인트 폴더다.

```bash
make validate GENERATOR=softmimicgen IMAGE=isaaclab-soft TASK=rope SCENE=ours OBSERVATION=image CLUSTER=ada RUN=<data_root>/runs/softmimicgen/rope/ours/none/pool_isaac/image/diffusion_policy/seed202601 CKPT=<그 RUN>/checkpoints/0100000 PORT=8900
```

결과 묶음만 다시 만들려면 호스트에서 다음을 실행한다.

```bash
python3 scripts/collect_results.py --data-root <data_root> --out <data_root>/reports/isaaclab_baseline --job softmimicgen/rope/ours/none/pool_isaac/image --job dexmimicgen/pick_place/ours/none/pool_isaac/image --logs <data_root>/logs/isaaclab_baseline
```

## 8. 측정한 시간과 예상 시간

측정값은 2026-10-28 Isaac 서버(RTX PRO 6000 Blackwell, 메모리 96 GB, 다른 사용자의 작업과 공유)에서 Rope 시범 실행으로 얻은 것이다.
Ada 6000에서는 다를 수 있으므로, 첫 과제가 끝나면 `driver.log`의 시각으로 다시 적는다.

| 단계 | 측정값 | 전체 실험에 대한 뜻 |
|---|---|---|
| Rope 생성 (Isaac Sim 프로세스 1개) | 26회 시도에 834초. Isaac Sim 시작 약 2분 포함, 시도 하나 평균 23.8초, 성공률 20/26 | 성공 1,000개(약 1,300회 시도)는 프로세스 하나로 약 8.5시간, 기본 설정(2개 동시)으로 약 4.5시간 |
| Towel 생성 (프로세스 1개) | 24회 시도에 1,270초, 성공률 20/24 | Rope와 비슷 |
| Pick Place 생성 (프로세스 1개) | 50회 시도에 1,940초(시도당 39초), 성공률 20/50 | 성공 1,000개(약 2,500회 시도)는 프로세스 하나로 약 27시간, 기본 설정(2개 동시)으로 약 14시간 |
| Microwave 생성 (AutoData, 프로세스 1개) | 20회 시도에 1,020초(시도당 약 50초), 성공률 20/20 | 성공 1,000개는 프로세스 하나로 약 14시간, 2개 동시로 약 7시간(성공률이 유지될 때) |
| Microwave 검증 25장면 (상한 250걸음) | 에피소드 하나 평균 68초, 25장면 39분 | 체크포인트 10개를 2개씩 검증하면 시드당 약 3.3시간 |
| 영상 데이터셋 쓰기 | 시연 10개(1,710걸음) 2초 | 1,000개는 수 분 |
| Diffusion Policy 학습 | 200갱신(배치 64, 시연 10개)에 약 100초, 시작 시간 포함 | 10만 갱신은 약 4에서 6시간으로 추정한다(측정하지 않았다) |
| Rope 검증 25장면 (Isaac Sim 프로세스 1개, 상한 400걸음) | 에피소드 하나 평균 89초(모두 실패해 400걸음을 다 돌았다), 25장면 37분 + 시작 2분 | 성공하는 정책은 일찍 끝나 더 빠르다. 체크포인트 10개를 2개씩 동시에 검증하면 시드당 약 3.5시간 |
| Pick Place 검증 25장면 (상한 600걸음, 지금 설정은 400) | 에피소드 하나 약 2.9분, 25장면 72분 | 상한 400으로는 약 50분. 체크포인트 10개를 2개씩 검증하면 시드당 약 4시간 |
| 시험 300장면 | 측정하지 않았다 | 에피소드 89초 기준 3개 분할로 시드당 약 2.5시간 |

에피소드 시간의 상당 부분은 정책 서버의 확산 추론(8걸음마다 100단계 잡음 제거)이다. 시간을 줄여야 하면 `VAL_PARALLEL`과
`TEST_SHARDS`를 GPU 메모리가 허락하는 만큼 늘린다(각각 Isaac Sim 프로세스 하나). GPU 메모리는 Isaac Sim 프로세스당 약 5에서 14 GB,
학습 약 10 GB다. 기본값(생성 2개 + 학습 1개 + 검증 2개 또는 시험 3개)이 48 GB 안에 드는지는 측정하지 않았다. 메모리 초과
(`CUDA out of memory`)가 나면 4절의 변수를 줄인다.

Dex Pick Place의 공식 문서 기준 생성 1,000개는 RTX 6000 Ada에서 약 19분(성공률 80%, 환경 20개 병렬)이다. 우리 설정은 프로세스
2개가 각각 환경 1개로 돌므로 더 걸린다.

## 9. 알려진 제한과 문제 해결

- Microwave Opening과 Bottle Placement and Door Closing은 NVIDIA AutoData(Isaac Lab-Arena)의 DexMimicGen 과제라 `isaaclab-autodata`
  이미지로 돈다. Arena 자산은 실행 때 Isaac 자산 서버와 Lightwheel 자산 서버(api.lightwheel.net)에서 내려받으므로 그 두 곳에 인터넷이
  닿아야 한다. 혼합과 추종 방법은 이 두 과제에 아직 연결되지 않았다(기준선 `method=none`만 된다).
- Isaac Lab의 Nut Pour와 Exhaust Pipe는 환경은 있지만 주석 달린 원본 시연 파일이 공개되어 있지 않아 쓰지 않는다.
- 평가 상한은 Rope 250, Towel 220, Pick Place 400걸음이다(과제 파일의 `horizon`). Rope와 Towel은 공식 설정의 롤아웃 상한이고, Pick Place는 환경 자체의 길이다.
- 원본 시연 수는 Soft Towel은 내려받아야 안다. `<data_root>/sources/softmimicgen/<과제>/source.manifest.json`의 `demos`에 적힌다.
- `docker: Error response from daemon: could not select device driver`는 NVIDIA Container Toolkit이 없다는 뜻이다. 설치 뒤 도커를 다시 시작한다.
- Isaac Sim이 뜨지 않고 EULA를 묻는 로그가 나오면 이미지 환경 변수 `ACCEPT_EULA=Y`가 빠진 것이다. 이미지를 다시 빌드한다.
- `policy server on port N did not come up`은 포트가 이미 쓰이고 있다는 뜻이다. `scripts/task_policy_pipeline.sh`의 두 번째 인자로 다른 시작 포트를 준다.
- Isaac Sim 로그에 `Failed to create any GPU devices`와 `vkCreateInstance failed`가 나오면 컨테이너 안에서 Vulkan이 GPU를 못 찾은 것이다.
  이미지는 NVIDIA 도커 런타임(`--runtime=nvidia`, `NVIDIA_DRIVER_CAPABILITIES=all`)으로 돌아야 드라이버의 Vulkan 드라이버 파일
  (`/etc/vulkan/icd.d/nvidia_icd.json`)이 컨테이너에 들어온다. 3.3절의 `nvidia-ctk runtime configure`가 그 런타임을 등록한다. 다음 명령이
  GPU 이름을 출력해야 한다.

  ```bash
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --entrypoint vulkaninfo tmg/isaaclab-soft:latest --summary | grep deviceName
  ```

- 디스크 여유가 `min_free_disk_gib`(50 GB) 아래로 떨어지면 생성이 시작하지 않는다. 공간을 확보하거나 `ada.yaml`의 값을 바꾼다.
- 생성 청크가 실패하면 풀 폴더의 `logs/seed<N>.log`에 원인이 있고 `DONE.json`은 생기지 않는다. 원인을 고친 뒤 같은 `make generate`를 다시 실행하면 끝난 청크는 유지된다.
