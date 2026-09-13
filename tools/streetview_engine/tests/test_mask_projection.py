"""Masked resize must not import excluded source RGB into valid supervision."""
import copy

import numpy as np
from PIL import Image
import pytest
from scipy.ndimage import minimum_filter

from tools.streetview_engine.mask_projection import resize_lanczos_valid_mask
from tools.streetview_engine.training import prepare_frame, masked_losses, supervised_frame_schedule


def test_identity_is_exact_independent_copy():
    mask = np.array([[True, False, True], [False, True, False]])
    result = resize_lanczos_valid_mask(mask, mask.shape)
    np.testing.assert_array_equal(result, mask)
    assert not np.shares_memory(result, mask)


@pytest.mark.parametrize('source_shape,target_shape', [
    ((97,131),(41,52)), ((19,43),(11,43)), ((19,43),(19,17)),
    ((19,43),(37,85)), ((128,128),(1,1)), ((1,11),(7,3)), ((11,1),(3,7))])
def test_all_valid_and_all_excluded_preserve_border_semantics(source_shape, target_shape):
    assert resize_lanczos_valid_mask(np.ones(source_shape,bool), target_shape).all()
    assert not resize_lanczos_valid_mask(np.zeros(source_shape,bool), target_shape).any()


@pytest.mark.parametrize('source_shape,target_shape', [
    ((97,131),(41,52)), ((19,43),(11,43)), ((19,43),(19,17)),
    ((19,43),(37,85)), ((128,128),(16,16)), ((1,43),(7,17)), ((43,1),(17,7))])
def test_odd_anisotropic_and_border_rgb_exclusion_invariance(source_shape, target_shape):
    rng = np.random.default_rng(492)
    mask = np.ones(source_shape, bool)
    # Isolated exclusions at interior and border: nearest can miss the former,
    # and treating outside-image support as invalid would over-erode the latter.
    mask[0,0] = False
    mask[source_shape[0]//2,source_shape[1]//2] = False
    a = rng.integers(40, 190, (*source_shape,3), dtype=np.uint8)
    b = a.copy(); b[~mask] = 250
    valid = resize_lanczos_valid_mask(mask, target_shape)
    assert valid.any()
    resized = [np.asarray(Image.fromarray(rgb).resize(target_shape[::-1], Image.Resampling.LANCZOS)) for rgb in (a,b)]
    np.testing.assert_array_equal(resized[0][valid], resized[1][valid])


def test_unchanged_axis_does_not_acquire_spurious_neighbor_erosion():
    mask = np.ones((19,43), bool)
    mask[:,20] = False
    result = resize_lanczos_valid_mask(mask, (11,43))
    assert not result[:,20].any()
    assert result[:,:20].all() and result[:,21:].all()


def synthetic_frame(shape, filename, **extra):
    h,w = shape
    return dict(station_id='synthetic_station', file_path=filename, mask_path='mask.png',
                w=w, h=h, fl_x=w/2, fl_y=h/2, cx=w/2, cy=h/2,
                transform_matrix=np.eye(4).tolist(), **extra)


def test_prepare_frame_removes_thin_stripe_rgb_loss_and_gradient_leak(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    shape = (128,128)
    mask = np.ones(shape,bool); mask[:,64] = False
    a = np.full((*shape,3),90,np.uint8)
    b = a.copy(); b[~mask] = 250
    for name, values in [('a.png',a),('b.png',b),('mask.png',mask.astype(np.uint8)*255)]:
        Image.fromarray(values).save(tmp_path/name)
    prepared = [prepare_frame(synthetic_frame(shape,name),tmp_path,64,np.zeros(3),1.) for name in ('a.png','b.png')]
    pa,pb = prepared
    # RGB remains the exact existing Lanczos result. Only supervision changes.
    np.testing.assert_array_equal(pa['rgb'],np.asarray(Image.fromarray(a).resize((64,64),Image.Resampling.LANCZOS)))
    nearest = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((64,64),Image.Resampling.NEAREST)) == 255
    assert nearest.all()
    delta = np.any(pa['rgb'] != pb['rgb'],axis=2)
    assert delta.any() and not pa['mask'][delta].any()
    np.testing.assert_array_equal(pa['mask'],pb['mask'])
    torch_results = []
    for item in prepared:
        prediction = (torch.tensor(pa['rgb'],dtype=torch.float64)/255).requires_grad_(True)
        l1,dssim = masked_losses(prediction,torch.tensor(item['rgb'],dtype=torch.float64)/255,
                                torch.tensor(item['mask']),torch.tensor(item['ssim_mask']))
        loss = .8*l1+.2*dssim
        loss.backward()
        torch_results.append((loss.detach(),prediction.grad))
    assert torch.equal(torch_results[0][0],torch_results[1][0])
    assert torch.equal(torch_results[0][1],torch_results[1][1])
    assert pa['mask_resampling']['signed_coefficients'] == 'positive_and_negative_contributors_required'
    assert pa['mask_resampling']['output_shape'] == [64,64]


def test_negative_lobes_cannot_be_accepted_by_float_mask_overshoot():
    mask = np.ones((128,128),bool); mask[:,64] = False
    float_coverage = np.asarray(Image.fromarray(mask.astype(np.float32)).resize((64,64),Image.Resampling.LANCZOS))
    assert float_coverage.max() > 1
    a = np.full((128,128,3),90,np.uint8)
    b = a.copy(); b[~mask] = 250
    ra,rb = [np.asarray(Image.fromarray(rgb).resize((64,64),Image.Resampling.LANCZOS)) for rgb in (a,b)]
    unsafe = (float_coverage>=1)&np.any(ra!=rb,axis=2)
    assert unsafe.any()
    assert not resize_lanczos_valid_mask(mask,(64,64))[unsafe].any()


def test_category_evidence_requires_full_support_independently_of_photo_mask(tmp_path):
    shape=(128,128)
    Image.fromarray(np.full((*shape,3),90,np.uint8)).save(tmp_path/'photo.png')
    Image.fromarray(np.full(shape,255,np.uint8)).save(tmp_path/'mask.png')
    category=np.full(shape,255,np.uint8);category[:,64]=0
    Image.fromarray(category).save(tmp_path/'category.png')
    frame=synthetic_frame(shape,'photo.png',foreground_mask_path='category.png',sky_mask_path='category.png')
    result=prepare_frame(frame,tmp_path,64,np.zeros(3),1.)
    assert result['mask'].all()
    assert not result['foreground'].all() and result['foreground'].any()
    np.testing.assert_array_equal(result['foreground'],result['sky'])
    assert not result['foreground'][:,30:35].any()


def test_resize_preserves_camera_station_and_world_scale_invariance(tmp_path):
    shape=(97,131)
    Image.fromarray(np.full((*shape,3),90,np.uint8)).save(tmp_path/'photo.png')
    mask=np.full(shape,255,np.uint8);mask[40:48,64]=0
    Image.fromarray(mask).save(tmp_path/'mask.png')
    frame=synthetic_frame(shape,'photo.png')
    frame['transform_matrix'][0][3]=3
    center=np.array([1.,2.,3.]);radius=7.
    before=prepare_frame(frame,tmp_path,52,center,radius)
    moved=copy.deepcopy(frame);shift=np.array([43.,-15.,108.]);scale=5.
    pose=np.asarray(moved['transform_matrix']);pose[:3,3]=pose[:3,3]*scale+shift
    moved['transform_matrix']=pose.tolist()
    after=prepare_frame(moved,tmp_path,52,center*scale+shift,radius*scale)
    assert before['station_id']==after['station_id']==frame['station_id']
    np.testing.assert_array_equal(before['K'],after['K'])
    np.testing.assert_allclose(before['view'],after['view'],rtol=0,atol=1e-7)
    np.testing.assert_array_equal(before['rgb'],after['rgb'])
    np.testing.assert_array_equal(before['mask'],after['mask'])


def test_insufficient_support_is_reported_without_losing_physical_station(tmp_path):
    shape=(128,128)
    Image.fromarray(np.full((*shape,3),90,np.uint8)).save(tmp_path/'photo.png')
    mask=np.zeros(shape,np.uint8);mask[64,64]=255
    Image.fromarray(mask).save(tmp_path/'mask.png')
    frame=synthetic_frame(shape,'photo.png')
    with pytest.raises(ValueError,match='no valid photometric'):
        prepare_frame(frame,tmp_path,32,np.zeros(3),1.)
    unsupported=prepare_frame(frame,tmp_path,32,np.zeros(3),1.,allow_empty=True)
    assert not unsupported['mask'].any()
    with pytest.raises(ValueError,match='physical station'):
        supervised_frame_schedule([unsupported],20,42)


@pytest.mark.parametrize('shape', [(0,3),(-1,3),(True,3),(3.5,4),(3,),[2,3,4],'2,3',np.array(3)])
def test_invalid_target_shape_rejected(shape):
    with pytest.raises(ValueError):
        resize_lanczos_valid_mask(np.ones((5,7),bool),shape)


def test_nonbinary_mask_representation_is_explicit():
    with pytest.raises(ValueError,match='bool'):
        resize_lanczos_valid_mask(np.full((5,7),255,np.uint8),(2,3))
