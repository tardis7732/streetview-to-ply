"""Resolve only verified completed application artifacts for explicit actions."""
import hashlib
import json
from pathlib import Path
import re
import threading

_RESOLVE_LOCK = threading.RLock()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, relative):
    if not isinstance(relative,str) or not relative:
        raise ValueError('완료된 작업의 파일 경로가 올바르지 않습니다.')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('완료된 작업의 파일을 찾을 수 없습니다.')
    return path


def resolve_artifact(app, payload):
    # Parallel UI/CLI actions may request the same legacy camera binding.
    with _RESOLVE_LOCK:
        return _resolve_artifact(app, payload)


def _resolve_artifact(app, payload):
    if not isinstance(payload, dict) or set(payload) != {'kind', 'job_id'}:
        raise ValueError('작업 종류와 ID가 필요합니다.')
    kind, job_id = payload['kind'], payload['job_id']
    if kind not in ('filter', 'generation') or not isinstance(job_id, str) or not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('올바른 완료 작업을 선택해 주세요.')
    manager = app.filters if kind == 'filter' else app.jobs
    job = manager.get(job_id)
    if job.get('status') != 'completed' or not job.get('artifact'):
        raise ValueError('완료된 PLY만 사용할 수 있습니다.')
    root = (manager.root / job_id).resolve()
    if not root.is_relative_to(manager.root.resolve()):
        raise ValueError('작업 폴더가 저장 범위를 벗어났습니다.')
    source = inside(root, job['artifact']['path'])
    if sha(source) != job['artifact'].get('sha256'):
        raise ValueError('완료 후 PLY가 변경됐습니다.')
    if kind == 'filter':
        from ..streetview_engine.ply_cleanup import verify_cleanup_report, write_output_camera_reference
        record = job.get('report') or {}
        report_path = inside(root, record.get('path', ''))
        if sha(report_path) != record.get('sha256'):
            raise ValueError('정리 결과의 검증 기록이 변경됐습니다.')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        camera = inside(root, 'result/cameras.json')
        verify_cleanup_report(report, Path(job['source_ply']), source, camera, inside(root, 'result/selection.npz'))
        reference = report['camera_reference']
        output_camera = root / 'result/artifact_cameras.json'
        if not output_camera.resolve().is_relative_to(root):
            raise ValueError('출력 카메라 경로가 작업 폴더를 벗어났습니다.')
        if not output_camera.exists():
            write_output_camera_reference(dict(report, artifact=dict(report['artifact'], path=str(source))), camera, output_camera)
        expected_doc=json.loads(camera.read_text(encoding='utf-8-sig'))
        expected_doc.update(coordinate_frame=reference['coordinate_frame'],units=reference['units'],
            camera_convention=reference['camera_convention'],
            ply_binding=dict(sha256=job['artifact']['sha256'],coordinate_frame=reference['coordinate_frame'],units=reference['units']),
            cleanup_provenance=dict(source_camera_sha256=reference['sha256'],source_ply_sha256=report['source']['sha256'],
                artifact_ply_sha256=job['artifact']['sha256'],unchanged_coordinate_frame=True,exact_retained_row_bytes=True))
    else:
        report_path=inside(root,'export/report.json')
        recorded_report=job.get('quality_report') or {}
        if recorded_report.get('sha256') and sha(report_path)!=recorded_report['sha256']:
            raise ValueError('완료 후 내보내기 기록이 변경됐습니다.')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if (report.get('status')!='completed' or report.get('artifact', {}).get('sha256') != job['artifact']['sha256']
                or report.get('artifact',{}).get('path')!=job['artifact']['path']):
            raise ValueError('PLY와 내보내기 기록이 다릅니다.')
        candidates = ['export/cameras.json', 'export/size_filter/cameras.json', 'sfm/dataset/transforms_train.json']
        camera = next((inside(root,name) for name in candidates if (root / name).is_file()), None)
        if camera is None:
            raise ValueError('이전 작업에는 내려받은 카메라 좌표가 없습니다. 새 작업부터 좌표를 함께 보관합니다.')
        training_path=inside(root,'training/manifest.json')
        training = json.loads(training_path.read_text(encoding='utf-8'))
        if (training.get('status')!='completed' or report.get('training_manifest_sha256')!=sha(training_path)
                or report.get('selection')!=training.get('selection')):
            raise ValueError('내보내기 기록이 완료된 학습 결과와 일치하지 않습니다.')
        from ..streetview_engine.size_filter import read_size_filter_options
        if report.get('size_filter') is not None or read_size_filter_options(job.get('config',{}))['enabled']:
            from ..streetview_engine.size_filter import verify_filtered_export_lineage
            verify_filtered_export_lineage(root,job['artifact']['path'],job.get('config',{}))
        elif training.get('selection',{}).get('accepted_model_sha256')!=job['artifact']['sha256']:
            raise ValueError('출력 PLY가 학습에서 선택된 모델과 다릅니다.')
        if sha(camera) != training.get('provenance', {}).get('inputs', {}).get('transforms_train.json'):
            raise ValueError('카메라 좌표가 학습 입력과 일치하지 않습니다.')
        original = json.loads(camera.read_text(encoding='utf-8-sig'))
        reference = {key: original.get(key, report.get(key)) for key in ('coordinate_frame', 'units', 'camera_convention')}
        if not all(reference.values()):
            raise ValueError('카메라의 좌표계와 단위가 명시되지 않았습니다.')
        output_camera = root / 'export/artifact_cameras.json'
        if not output_camera.resolve().is_relative_to(root):
            raise ValueError('출력 카메라 경로가 작업 폴더를 벗어났습니다.')
        expected_doc=dict(original)
        expected_doc.update(**reference, ply_binding=dict(sha256=job['artifact']['sha256'], coordinate_frame=reference['coordinate_frame'], units=reference['units']),
                source_camera_sha256=sha(camera), output_artifact_sha256=job['artifact']['sha256'])
        if not output_camera.exists():
            output_camera.write_text(json.dumps(expected_doc, ensure_ascii=False, indent=2), encoding='utf-8')
    output_doc = json.loads(output_camera.read_text(encoding='utf-8-sig'))
    # Intrinsics and world_up can live at document level; frame equality alone
    # would permit altered crop axes or camera rays in a previously bound file.
    if output_doc != expected_doc:
        raise ValueError('출력 PLY의 카메라 좌표 연결이 변경됐습니다.')
    if any(output_doc.get(key) != reference[key] for key in ('coordinate_frame', 'units', 'camera_convention')):
        raise ValueError('출력 카메라의 좌표계와 단위가 변경됐습니다.')
    return dict(source_ply=str(source), camera_json=str(output_camera), source_sha256=sha(source),
        camera_sha256=sha(output_camera), coordinate_frame=reference['coordinate_frame'], units=reference['units'],
        camera_convention=reference['camera_convention'], label=job.get('label') or f'{kind}_{job_id[:8]}',
        source_ref=dict(kind=kind, job_id=job_id))
