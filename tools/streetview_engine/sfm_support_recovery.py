"""Additional visual matching for missing rigs, without invented camera poses."""
from pathlib import Path
import time

from .sfm_geometry import candidate_capture_pairs


def additional_image_pairs(captures, mapping, registered, neighbors, existing):
    """Expand the candidate graph only across missing/registered capture rigs."""
    registered=set(registered)
    by_capture={str(row['pano_id']):[] for row in captures}
    for name,row in mapping.items():
        by_capture[str(row['pano_id'])].append(name)
    if not registered <= set(by_capture):
        raise ValueError('Registered captures are outside the prepared roster')
    prior={tuple(sorted(pair)) for pair in existing}
    result=set()
    for first,second in candidate_capture_pairs(captures,neighbors):
        a,b=str(captures[first]['pano_id']),str(captures[second]['pano_id'])
        if (a in registered)==(b in registered):
            continue
        for left in by_capture[a]:
            for right in by_capture[b]:
                pair=tuple(sorted((left,right)))
                if pair not in prior:result.add(pair)
    return sorted(result)


def recover_with_additional_matches(pc, output, rec, mapping, captures, settings, device, initial):
    """Reuse stored SIFT features and keep native pose/inlier/BA acceptance rules."""
    from .sfm import recover_capture_rigs, save, sha
    output=Path(output)
    registered={mapping[im.name]['pano_id'] for im in rec.images.values() if im.has_pose and im.name in mapping}
    expected={row['pano_id'] for row in captures}
    if (registered==expected or settings.recovery_match_neighbors==0 or settings.recovery_rounds==0):
        return rec,initial
    previous=output/'match_pairs.txt'
    existing=[line.split() for line in previous.read_text(encoding='utf8').splitlines() if line.strip()]
    if any(len(pair)!=2 for pair in existing):raise ValueError('Invalid original visual pair list')
    pairs=additional_image_pairs(captures,mapping,registered,settings.recovery_match_neighbors,existing)
    report=dict(status='no_additional_pairs',neighbors=settings.recovery_match_neighbors,
        missing_capture_ids=sorted(expected-registered),additional_image_pairs=len(pairs),
        additional_capture_pairs=len({tuple(sorted((mapping[a]['pano_id'],mapping[b]['pano_id']))) for a,b in pairs}),
        base_pair_list_sha256=sha(previous),feature_extractions=0,
        policy='Existing SIFT features; only missing/registered rig pairs; native matcher and pose thresholds unchanged; no GPS pose substitution')
    if not pairs:
        save(output/'additional_matching_report.json',report)
        return rec,dict(initial,additional_matching=report)
    path=output/'recovery_match_pairs.txt'
    path.write_text(''.join(a+' '+b+'\n' for a,b in pairs),encoding='utf8')
    report.update(pair_list_sha256=sha(path),database_sha256_before=sha(output/'database.db'))
    started=time.monotonic()
    matching=pc.FeatureMatchingOptions(num_threads=settings.threads,use_gpu=device=='cuda',guided_matching=True,skip_image_pairs_in_same_frame=True)
    verification=pc.TwoViewGeometryOptions();verification.ransac.random_seed=settings.random_seed
    pc.match_image_pairs(output/'database.db',matching_options=matching,
        pairing_options=pc.ImportedPairingOptions(match_list_path=path,block_size=128),
        verification_options=verification,device=getattr(pc.Device,device))
    report.update(status='matched',elapsed_seconds=time.monotonic()-started,database_sha256_after=sha(output/'database.db'))
    rec,recovery=recover_capture_rigs(pc,output,rec,mapping,settings)
    report.update(recovery_status=recovery['status'],registered_after=recovery['after'])
    save(output/'additional_matching_report.json',report)
    return rec,dict(recovery,initial_recovery=initial,additional_matching=report)
