"""Mask geometry checks use analytic rays and synthetic boolean ERP patterns."""
import copy

import numpy as np
import pytest

from tools.streetview_engine.imaging import FACES, cube_camera_to_station_cv
from tools.streetview_engine.panorama_native_mapping import native_uv_alpha
from tools.streetview_engine.panorama_mask_projection import project_erp_mask, project_erp_mask_to_cubes, projection_policy


def camera(face, size=33):
    return dict(face=face,w=size,h=size,fl_x=size/2,fl_y=size/2,cx=size/2,cy=size/2,
        camera_to_station_cv=cube_camera_to_station_cv(face).tolist())


@pytest.mark.parametrize('face,x,y',[('F',6,3),('R',9,3),('B',0,3),('L',3,3),('U',6,0),('D',6,5)])
def test_exact_face_center_axes_and_poles(face,x,y):
    mask=np.zeros((6,12),bool);mask[y,x]=True
    actual=project_erp_mask(mask,camera(face,3),sampling='nearest')
    assert actual[1,1]


def test_back_face_bilinear_wrap_reads_both_actual_seam_columns():
    left=np.zeros((32,64),bool);left[:,0]=True
    right=np.zeros_like(left);right[:,-1]=True
    back=camera('B',33)
    for mask in (left,right):
        projected=project_erp_mask(mask,back)
        assert projected[16,16]
        assert not project_erp_mask(mask,camera('F',33)).any()
    # Nearest chooses column zero at the +pi center; it does not clamp to W-1.
    assert project_erp_mask(left,back,sampling='nearest')[16,16]
    assert not project_erp_mask(right,back,sampling='nearest')[16,16]


def test_pole_edges_replicate_without_wrapping_top_to_bottom():
    north=np.zeros((32,64),bool);north[0]=True
    assert project_erp_mask(north,camera('U'))[16,16]
    assert not project_erp_mask(north,camera('D'))[16,16]
    south=north[::-1].copy()
    assert project_erp_mask(south,camera('D'))[16,16]
    assert not project_erp_mask(south,camera('U'))[16,16]


def test_every_face_matches_compositor_support_chunking_and_preserves_input():
    mask=np.random.default_rng(42).random((64,128))<.07;original=mask.copy()
    cameras=[camera(face,47) for face in FACES]
    projected=project_erp_mask_to_cubes(mask,cameras,chunk_rows=7)
    assert list(projected)==list(FACES) and np.array_equal(mask,original)
    for row in cameras:
        expected=native_uv_alpha(row,mask.astype(float),chunk_rows=13)[1]>0
        assert projected[row['face']].dtype==bool
        assert np.array_equal(projected[row['face']],expected)
        assert np.array_equal(projected[row['face']],project_erp_mask(mask,row,chunk_rows=47))
        nearest=project_erp_mask(mask,row,sampling='nearest')
        assert np.all(projected[row['face']][nearest])


def test_nearest_matches_independent_scalar_camera_ray_oracle():
    mask=np.indices((18,36)).sum(0)%3==0
    for face in FACES:
        row=camera(face,11);actual=project_erp_mask(mask,row,sampling='nearest')
        rotation=np.asarray(row['camera_to_station_cv'])[:3,:3]
        for y,x in [(0,0),(0,10),(4,6),(7,2),(10,10)]:
            ray=rotation @ np.array([(x+.5-row['cx'])/row['fl_x'],(y+.5-row['cy'])/row['fl_y'],1.])
            ray/=np.linalg.norm(ray)
            longitude=np.arctan2(ray[0],ray[2]);latitude=np.arctan2(ray[1],np.hypot(ray[0],ray[2]))
            col=int(np.floor((longitude+np.pi)/(2*np.pi)*36))%36
            line=np.clip(int(np.floor((latitude+np.pi/2)/np.pi*18)),0,17)
            assert actual[y,x]==mask[line,col]


def test_native_1536_mask_alignment_uses_original_pixel_centers():
    mask=np.zeros((1024,2048),bool)
    mask[500:520,1000:1048]=True
    row=camera('F',1536)
    output=project_erp_mask(mask,row)
    # The exact native compositor's alpha uses this same unshifted ray grid.
    reference=native_uv_alpha(row,mask.astype(float),chunk_rows=64)[1]
    assert output.shape==(1536,1536) and output.any()
    assert np.array_equal(output,reference>0)


@pytest.mark.parametrize('mask',[np.zeros((4,8),np.uint8),np.zeros((4,7),bool),np.zeros((0,0),bool),np.zeros((4,8,1),bool)])
def test_bad_mask_format_fails(mask):
    with pytest.raises(ValueError,match='2:1 boolean'):project_erp_mask(mask,camera('F'))


def test_invalid_calibration_options_and_incomplete_cube_fail():
    mask=np.zeros((4,8),bool)
    for size in (0,-1,1.5,True):
        with pytest.raises(ValueError):project_erp_mask(mask,dict(camera('F'),w=size))
    for chunk in (0,-1,1.5,True):
        with pytest.raises(ValueError):project_erp_mask(mask,camera('F'),chunk_rows=chunk)
    with pytest.raises(ValueError):project_erp_mask(mask,camera('F'),sampling='dilate')
    with pytest.raises(ValueError):project_erp_mask_to_cubes(mask,[camera('F')])
    cameras=[camera(face) for face in FACES];cameras[0]['fl_x']+=1
    with pytest.raises(ValueError,match='90-degree'):project_erp_mask_to_cubes(mask,cameras)
    moved=camera('F');moved['camera_to_station_cv'][0][3]=1
    with pytest.raises(ValueError,match='Non-centered'):project_erp_mask(mask,moved)
    assert not projection_policy()['entire_pixel_footprint_conservative']
