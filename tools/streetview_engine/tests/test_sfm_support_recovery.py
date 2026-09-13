from dataclasses import replace

import pytest

from tools.streetview_engine.sfm import SfMSettings
from tools.streetview_engine.sfm_support_recovery import additional_image_pairs, recover_with_additional_matches


def scene():
    captures=[dict(pano_id=pid,station_id=group,lat=37.0,lng=127.0+index*.0001)
              for index,(pid,group) in enumerate([('a','ga'),('b','gb'),('a_copy','ga'),('c','gc')])]
    mapping={pid+'_'+face:dict(pano_id=pid,station_id=row['station_id'])
             for row in captures for pid in [row['pano_id']] for face in ['F','R']}
    return captures,mapping


def test_expansion_only_connects_missing_to_registered_and_never_same_physical_station():
    captures,mapping=scene()
    pairs=additional_image_pairs(captures,mapping,{'b','a_copy','c'},4,[('b_F','a_F')])
    assert len(pairs)==7
    assert ('a_F','b_F') not in pairs
    assert all('a_copy' not in a and 'a_copy' not in b for a,b in pairs)
    assert all(a.startswith('a_') and b.startswith(('b_','c_')) for a,b in pairs)


def test_capture_order_and_added_world_coordinates_do_not_change_pair_selection():
    captures,mapping=scene()
    expected=additional_image_pairs(captures,mapping,{'b','a_copy','c'},4,[])
    altered=[dict(row,world_center=[1e7+i*100,-40,30],world_scale=1e6) for i,row in enumerate(reversed(captures))]
    assert additional_image_pairs(altered,dict(reversed(list(mapping.items()))),{'b','a_copy','c'},4,[])==expected


def test_no_registered_or_no_missing_captures_produces_no_pairs():
    captures,mapping=scene()
    assert additional_image_pairs(captures,mapping,set(),4,[])==[]
    assert additional_image_pairs(captures,mapping,{row['pano_id'] for row in captures},4,[])==[]
    with pytest.raises(ValueError,match='outside'):
        additional_image_pairs(captures,mapping,{'unknown'},4,[])


def test_disabled_recovery_does_not_read_or_write_database(tmp_path):
    captures,mapping=scene()
    class Reconstruction:
        images={}
    rec=Reconstruction(); original=dict(status='insufficient_verified_support')
    result,report=recover_with_additional_matches(None,tmp_path,rec,mapping,captures,SfMSettings(),'cpu',original)
    assert result is rec and report is original
    assert list(tmp_path.iterdir())==[]
    for value in [-1,501,True,1.5]:
        with pytest.raises(ValueError,match='recovery_match_neighbors'):
            replace(SfMSettings(),recovery_match_neighbors=value)
