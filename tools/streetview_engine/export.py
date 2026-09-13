"""Standard binary Gaussian PLY serialization and hash-bound delivery."""
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from .processing_options import read_processing_options


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def validate_ply(path):
    path = Path(path)
    properties, count = [], None
    with path.open("rb") as stream:
        if stream.readline() != b"ply\n" or stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Expected standard binary little-endian PLY")
        while True:
            if stream.tell() > 65536:
                raise ValueError("Oversized PLY header")
            raw = stream.readline(8192)
            if not raw:
                raise ValueError("Truncated PLY header")
            items = raw.decode("ascii").split()
            if not items or items[0] in ("comment", "obj_info"):
                continue
            if items == ["end_header"]:
                break
            if items[0] == "element" and len(items) == 3 and items[1] == "vertex" and count is None:
                count = int(items[2])
            elif items[0] == "property" and len(items) == 3 and items[1] in ("float", "float32") and items[2] not in properties:
                properties.append(items[2])
            else:
                raise ValueError("Unsupported PLY declaration")
        offset = stream.tell()
    required = {"x", "y", "z", "opacity", *(f"f_dc_{i}" for i in range(3)),
                *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4))}
    if not count or count < 0 or not required <= set(properties):
        raise ValueError("Missing Gaussian PLY fields or vertices")
    rest = [name for name in properties if name.startswith("f_rest_")]
    if set(rest) != {f"f_rest_{i}" for i in range(len(rest))} or len(rest) % 3:
        raise ValueError("Malformed spherical harmonic fields")
    basis = len(rest) // 3 + 1
    if int(math.isqrt(basis)) ** 2 != basis:
        raise ValueError("Incomplete spherical harmonic degree")
    if path.stat().st_size != offset + count * len(properties) * 4:
        raise ValueError("PLY size differs from declared float payload")
    data = np.memmap(path, mode="r", dtype="<f4", offset=offset, shape=(count, len(properties)))
    try:
        for start in range(0, count, 65536):
            chunk = data[start:start + 65536]
            if not np.isfinite(chunk).all():
                raise ValueError("Nonfinite Gaussian PLY value")
            quat = chunk[:, [properties.index(f"rot_{i}") for i in range(4)]]
            if np.any(np.linalg.norm(quat, axis=1) < 1e-8):
                raise ValueError("Zero Gaussian quaternion")
    finally:
        data._mmap.close()
        del data
    return dict(sha256=sha256(path), bytes=path.stat().st_size, vertex_count=count,
                sh_degree=math.isqrt(basis) - 1, format="binary_little_endian_gaussian_ply",
                fields=properties, validation="serialization_and_finite_parameters_not_visual_quality")


def write_model(path, *, means, log_scales, quats, opacity_logits, sh0, shN):
    """Write EDN metre centers, log metre scales, WXYZ and INRIA SH ordering."""
    path = Path(path)
    means, log_scales, quats, opacity_logits, sh0, shN = [np.asarray(x) for x in (means, log_scales, quats, opacity_logits, sh0, shN)]
    count = len(means)
    if sh0.shape == (count, 1, 3):
        sh0 = sh0[:, 0]
    if means.shape != (count, 3) or log_scales.shape != means.shape or quats.shape != (count, 4) or opacity_logits.shape != (count,) or sh0.shape != means.shape or shN.ndim != 3 or shN.shape[0] != count or shN.shape[2] != 3:
        raise ValueError("Invalid Gaussian export shapes")
    if not count or not all(np.isfinite(x).all() for x in (means, log_scales, quats, opacity_logits, sh0, shN)):
        raise ValueError("Invalid Gaussian export values")
    norms = np.linalg.norm(quats, axis=1)
    if (norms < 1e-8).any() or math.isqrt(shN.shape[1] + 1) ** 2 != shN.shape[1] + 1:
        raise ValueError("Invalid quaternion or spherical harmonic degree")
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(shN.shape[1] * 3)]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate([means, np.zeros_like(means), sh0, shN.transpose(0, 2, 1).reshape(count, -1),
                           opacity_logits[:, None], log_scales, quats / norms[:, None]], axis=1).astype("<f4")
    if not np.isfinite(data).all():
        raise ValueError("Export overflows float32")
    header = "ply\nformat binary_little_endian 1.0\ncomment coordinates EDN metres; world_up 0 -1 0\ncomment quaternion WXYZ; scales log metres; opacity logits\n"
    header += f"element vertex {count}\n" + "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header.encode("ascii"))
        data.tofile(stream)
    result = validate_ply(temporary)
    temporary.replace(path)
    return result


def run(config, job_dir, settings):
    """Deliver an accepted artifact, optionally followed by a bound size cap."""
    from .size_filter import read_size_filter_options, run_size_filter
    root = Path(job_dir).resolve()
    size_options = read_size_filter_options(config)
    if config.get('generation_mode', 'multi_view') != 'multi_view':
        raise ValueError('Only multi_view generation is available')
    source_manifest = root / "training/manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf8"))
    processing = read_processing_options(config)
    if read_processing_options(manifest) != processing:
        raise ValueError('Export processing options differ from the trained artifact')
    if manifest.get("status") != "completed" or not manifest.get("selection", {}).get("accepted_model"):
        raise ValueError("Training has no completed accepted model")
    source = (root / "training" / manifest["selection"]["accepted_model"]).resolve()
    if not source.is_relative_to(root / "training"):
        raise ValueError("Accepted model escapes training directory")
    actual = validate_ply(source)
    if actual["sha256"] != manifest["selection"].get("accepted_model_sha256"):
        raise ValueError("Accepted model changed since evaluation")
    inspection = None
    options = settings.get("inspection", {})
    if not isinstance(options, dict) or (options and type(options.get("enabled")) is not bool):
        raise ValueError("Inspection settings require explicit enabled=true/false")
    if options.get("enabled"):
        from .inspection import InspectionConfig, inspect_scene
        inspection_config = InspectionConfig(**{key:value for key,value in options.items() if key != "enabled"})
        has_sky = bool((manifest.get("quality") or {}).get("sky_component"))
        inspection = inspect_scene(source, root / "sfm/dataset",
            components=source_manifest if has_sky else None, config=inspection_config)
        if inspection["input_ply"]["sha256"] != actual["sha256"]:
            raise ValueError("Inspection belongs to another model")
    output = root / "export"
    output.mkdir(exist_ok=True)
    size_filter = None
    accepted = actual
    if size_options['enabled']:
        camera_json = root / 'sfm/dataset/transforms_train.json'
        if manifest.get('provenance', {}).get('inputs', {}).get('transforms_train.json') != sha256(camera_json):
            raise ValueError('Size-filter camera reference differs from accepted training inputs')
        if (output / 'scene.ply').exists() or (output / 'size_filter').exists():
            raise ValueError('Filtered export requires a new export target')
        unfiltered = output / 'source.ply'
        if unfiltered.exists() and sha256(unfiltered) != actual['sha256']:
            raise ValueError('Refusing to replace a different unfiltered source')
        if not unfiltered.exists():
            shutil.copyfile(source, unfiltered)
        if sha256(unfiltered) != actual['sha256']:
            raise ValueError('Unfiltered export source copy differs')
        size_filter = run_size_filter(unfiltered, camera_json,
                                      output / 'size_filter', size_options)
        source = Path(size_filter['artifact']['path'])
        actual = validate_ply(source)
    target = output / "scene.ply"
    if target.exists() and sha256(target) != actual["sha256"]:
        raise ValueError("Refusing to overwrite a different exported scene")
    if not target.exists():
        temporary = output / "scene.ply.tmp"
        shutil.copyfile(source, temporary)
        if sha256(temporary) != actual["sha256"]:
            raise ValueError("Export copy hash mismatch")
        temporary.replace(target)
    report = dict(status="completed", artifact=dict(path="export/scene.ply", **actual),
                  training_manifest_sha256=sha256(source_manifest), selection=manifest["selection"],
                  processing_options=processing,
                  quality=manifest.get("quality"), coordinate_frame="EDN", units="metres", world_up=[0, -1, 0],
                  interoperability="Standard Gaussian PLY fields; application-specific Nuke import not asserted by serialization checks",
                  sky_policy=manifest.get('sky_policy', "Shared trained Gaussian scene; no per-photo background overlay"))
    if size_filter is not None:
        report['source'] = dict(path='export/source.ply', **accepted)
        report['size_filter'] = dict(status='completed', options=size_options,
            report='export/size_filter/report.json', sha256=sha256(output / 'size_filter/report.json'),
            source_path='export/source.ply', camera_path='export/size_filter/cameras.json',
            selection_path='export/size_filter/selection.npz', removed_rows=size_filter['removed_rows'],
            remaining_rows=size_filter['remaining_rows'])
        report['quality_scope'] = 'Training quality belongs to the accepted unfiltered source; filtered visual quality has not been evaluated'
        report['source_quality'] = report.pop('quality')
        report['quality'] = None
    if inspection is not None:
        write_json(output / "inspection.json", inspection)
        report["inspection"] = dict(report="export/inspection.json", sha256=sha256(output / "inspection.json"),
            status=inspection["status"], all_rows=inspection["all_rows"],
            foreground_rows=inspection["foreground"]["counts"]["rows"],
            review_candidates=inspection["foreground"]["review_candidates"],
            ground_support={key:value for key,value in inspection["ground_support"].items() if key != "frames"},
            interpretation="Read-only evidence, not an approval to delete flagged Gaussians")
        if size_filter is not None:
            report['inspection']['scope'] = 'Accepted unfiltered source; not a filtered-result inspection'
    camera_source = root / 'sfm/dataset/transforms_train.json'
    if camera_source.is_file() and manifest.get('provenance', {}).get('inputs', {}).get('transforms_train.json') == sha256(camera_source):
        camera_target = output / 'cameras.json'
        if camera_target.exists() and sha256(camera_target) != sha256(camera_source):
            raise ValueError('Export camera copy differs from trained inputs')
        if not camera_target.exists():
            shutil.copyfile(camera_source, camera_target)
        report['camera_reference'] = dict(path='export/cameras.json', sha256=sha256(camera_target),
            frame_unchanged=True, source='accepted training transforms_train.json')
    write_json(output / "report.json", report)
    return report
