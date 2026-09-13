# Streetview Studio GUI

로컬 지도에서 중심과 반경을 지정하고 촬영 지점을 선택해 Gaussian PLY를 만드는 인터페이스다. 수집 반경과 선택 개수의 프로그램 상한은 없다. 실제 제공되는 날짜를 선택하고, 지점을 제외하거나 네이버 거리뷰를 직접 열 수 있다.

## 실행

Python 3.11 이상에서 저장소 루트 기준으로 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r tools/streetview_app/requirements.txt
.\.venv\Scripts\python.exe -m tools.streetview_app.launch
```

Linux에서는 `.venv/bin/python`을 사용한다. 실행기는 현재 Python 환경으로 서버를 시작하고 브라우저를 연다. 기본 주소는 `http://127.0.0.1:8765/`다. 기존 서버가 있으면 재사용한다. 브라우저를 열지 않으려면 `python -m tools.streetview_app.server --port 8765`로 실행한다.

지도 조회와 미리보기는 GPU 설정 없이 사용할 수 있다. Leaflet과 OpenStreetMap 지도 타일은 인터넷에서 불러온다. 학습은 운영자가 생성 백엔드를 연결한 뒤 **PLY 생성**을 명시적으로 눌러야 시작한다. 자세한 GPU 설치와 모델 설정은 저장소 루트 README를 참고한다.

## 생성 설정

여러 촬영 지점을 연결하는 SfM 기반 생성만 지원한다. Gaussian 예측 모델의 결과를 초기값으로 사용하지 않는다. 파노라마 이미지 가공, Brush 기본 학습, gsplat 추가 학습, 깊이 기반 부유물 정리와 크기 필터를 운영자 프리셋으로 연결할 수 있다.

- **인물·차량 처리, 하늘 처리**: 연결된 전처리 경로의 옵션이다. 파노라마 제거·합성 경로와 RGB 학습 마스킹 경로의 의미를 프리셋 설명에서 구분한다.
- **뎁스 기반 부유물 정리**: 지원되는 생성 프리셋에서 작업마다 켜거나 끈다. 끄면 DA3 추론과 해당 최종 정리를 생략한다. SfM 깊이를 사용하는 추가 학습과 마지막 크기 필터는 별도 설정을 따른다.
- **큰 Gaussian 제거**: 실제 촬영 위치 반경에 대한 Gaussian 표준편차 비율로 크기를 제한한다. 깊이맵 설정과 별개다.
- **제작 프리셋**: 위치 선택과 분리된 처리 설정을 저장한다. 대표 이미지 한 장은 참고용이며 새 작업의 결과가 아니다. 기본 프리셋이나 사용자 촬영 데이터는 저장소에 포함하지 않는다.

코드·설정이 변경되면 기존 프리셋의 실행 스냅샷이 달라져 재등록이 필요할 수 있다. 변경된 코드로 과거 프리셋을 조용히 실행하지 않는다.

## 클라우드 연결

SSH 키와 호스트 설정을 먼저 준비하고, 운영자 설정 JSON을 로컬에 만든다. 원격 설치에는 `run_python.sh`와 `code/tools/streetview_engine/remote_worker.py`가 필요하다.

```powershell
python -m tools.streetview_app.cloud_setup --host my-gpu --remote-root /srv/streetview-to-ply --settings tools/streetview_app/data/engine_settings.json --output tools/streetview_app/data/backend.json
```

이 명령은 설정 파일만 만든다. 연결 시험이나 학습을 실행하지 않는다. 새 작업은 GPU 확인 후 작업별 디렉터리·프로세스·임대 시간을 관리한다. 서버가 재시작되면 중단된 작업을 자동 재개하지 않는다.

## 결과와 CLI

완료된 PLY는 다운로드하거나 **범위·크기 정리**로 새 파일을 만들 수 있다. 원본은 보존한다. 연결된 카메라와 좌표 근거가 필요하며, 범위 자르기는 기본 꺼짐이다.

**언리얼에서 열기**는 Windows의 Unreal Editor와 사용자가 설치한 MLSLabsRenderer, PythonScriptPlugin, EditorScriptingUtilities가 있는 프로젝트를 `data/unreal_profile.json`에 연결했을 때 사용할 수 있다. 플러그인과 언리얼 프로젝트는 이 저장소에 포함하지 않는다.

GUI 서버가 실행 중이면 같은 동작을 CLI로 실행할 수 있다. `selection.json`은 GUI에서 저장한 선택 파일이다.

```powershell
python -m tools.streetview_app.workflow_cli presets
python -m tools.streetview_app.workflow_cli save-preset --name "내 설정" --config selection.json
python -m tools.streetview_app.workflow_cli generate --config selection.json --recipe-id "<recipe-id>"
python -m tools.streetview_app.workflow_cli status "<job-id>"
python -m tools.streetview_app.workflow_cli reuse "<job-id>"
python -m tools.streetview_app.workflow_cli reuse "<job-id>" --from-stage export
python -m tools.streetview_app.workflow_cli open-unreal "<job-id>"
```

`reuse`는 `--from-stage` 없이 실행하면 가능 여부만 조회한다. 완료 단계의 입력·설정·코드·산출물 해시가 모두 맞아야 재사용한다. 새 설정으로 학습을 이어가는 체크포인트 기능은 아니다.

`data/`에는 캐시, 선택, 프리셋, 작업, 로그와 로컬 실행 설정이 저장된다. Git에는 포함하지 않는다. HTTP는 로컬 동일 출처 요청만 받고, HTTP 요청에 임의 실행 명령이나 모델 경로를 받지 않는다.
