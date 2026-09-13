# Streetview to PLY

지도에서 거리뷰 지점을 고르고 주변 공간을 Gaussian PLY로 만드는 로컬 GUI와 CLI입니다.
현재 GUI에 연결된 생성·가공·정리 코드만 담았습니다.

- 지도 중심과 반경 선택, 촬영본 포함·제외, 제공되는 촬영 시기 선택
- GUI 안에서 거리뷰 보기와 네이버 지도에서 직접 열기
- 파노라마에서 인물·차량 제거 후 원본 해상도 큐브 면으로 합성
- SfM → Brush 0.3 → gsplat 추가 학습
- 깊이 기반 부유물 정리, 큰 Gaussian 제거, 출력 범위 자르기
- 제작 프리셋, 완료 단계 재사용, PLY 다운로드, 설정한 언리얼 프로젝트에서 열기

SHARP/UniSHARP가 예측한 Gaussian은 생성에 사용하지 않습니다. 학습 초기 점은 SfM에서 얻습니다.
반경과 촬영 지점 수에 프로그램 상한은 없으며, 실제 수집 범위와 필요한 자원에 따라 실행 규모가 달라집니다.

## GUI 실행

Python 3.11 이상을 사용합니다. Windows PowerShell:

```powershell
git clone https://github.com/tardis7732/streetview-to-ply.git
cd streetview-to-ply
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m tools.streetview_app.launch
```

설치 후 `Start Streetview.vbs`를 두 번 클릭해도 됩니다. Linux/macOS에서는:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m tools.streetview_app.launch
```

기본 주소는 `http://127.0.0.1:8765/`입니다. 다른 프로그램이 포트를 사용하면 빈 로컬 포트를 선택합니다.
지도와 거리뷰에는 인터넷 연결이 필요합니다. 지도 탐색·선택 저장은 GPU 학습을 시작하지 않습니다.

**GUI 설치만으로 GPU 생성 환경이 설치되지는 않습니다.** 새 설치에서는 연결 전까지 생성 버튼을 사용할 수 없습니다.
모델과 학습 도구를 준비한 뒤 [GPU 환경](docs/GPU_SETUP.md)과 [운영자 설정](docs/CONFIGURATION.md)을 따라 연결합니다.
개인 SSH 설정, 모델 가중치, 실행 환경, 촬영 이미지, 기존 프리셋·작업 결과는 저장소에 포함하지 않습니다.

## 생성 과정

```text
거리뷰 원본 6면
  → 전체 파노라마에서 제거 이미지 생성
  → 원본에 마스크 영역만 정합·페더 합성
  → 원본 해상도 6면
  → SfM 카메라·초기 점
  → Brush 학습 → gsplat + SfM 희소 깊이 추가 학습
  → 선택한 깊이·하늘 정리 → 크기·출력 범위 필터 → PLY
```

객체 마스크는 원본 면에서 검출하고 파노라마에 투영하는 기본 방식을 사용합니다.
생성 모델 입력과 최종 합성 해상도는 서로 다릅니다. 마스크 바깥 원본 RGB는 보존하며 차량 그림자를 별도 확장하지 않습니다.
GPS는 카메라 위치 복원의 보조값입니다. DA3 깊이는 최종 부유물 정리의 보조값이며, 추가 학습에 쓰는 SfM 희소 깊이와 구분합니다.
상공처럼 촬영본에 없는 시점의 표면이나 지붕을 복원했다고 보장하지 않습니다.

`뎁스 기반 부유물 정리`와 `큰 Gaussian 제거`는 별도 옵션입니다.
기존 PLY에서 크기·범위 필터를 실행하면 원본을 보존하고 새 결과를 만듭니다.
현재 GUI의 기존 PLY 정리에는 깊이 재추론 기능이 없습니다.

## CLI

GUI와 동일한 로컬 서버를 사용합니다. `selection.json`은 GUI에서 저장한 선택 파일입니다.

```bash
streetview presets
streetview generate --config selection.json --recipe-id <recipe-id>
streetview status <job-id>
streetview reuse <job-id>
streetview reuse <job-id> --from-stage export
streetview open-unreal <job-id>
```

서버 없이 기존 PLY의 큰 Gaussian 필터만 실행할 수도 있습니다:

```bash
python -m tools.streetview_engine.size_filter --input scene.ply --cameras cameras.json --output-dir outputs/filtered --ratio 0.5
```

카메라 좌표와 PLY가 같은 좌표계인지 확인할 수 있어야 합니다. 출력 디렉터리는 새 경로를 지정합니다.

## 개발

```bash
python -m pip install -e ".[dev]"
python -m pytest tools/streetview_app/tests/test_server.py tools/streetview_app/tests/test_selection.py tools/streetview_engine/tests/test_size_filter.py
```

소스는 `tools/streetview_app`(GUI/API/CLI), `tools/streetview_engine`(생성 단계),
`tools/streetview_geometry`(공통 기하 처리)로 나뉩니다. 선택·작업·운영자 설정은
`tools/streetview_app/data/`에 로컬로 저장하며 Git에서 제외합니다.

제3자 모델·소프트웨어와 지도 데이터는 각각의 제공 조건을 따릅니다.
이 저장소는 네이버, Apple, 모델 제작사 또는 언리얼 플러그인 제작사의 공식 프로젝트가 아닙니다.
모델·도구의 출처와 설치 조건은 [GPU 환경](docs/GPU_SETUP.md)을 참조하세요.
