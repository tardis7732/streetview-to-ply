"""Selection and transport preserve an explicit export-only size filter."""
import copy
import io
import json
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tools.streetview_app.jobs import validate_config
from tools.streetview_app.plans import PlanStore, freeze_selection
from tools.streetview_app.remote_jobs import source_bundle
from tools.streetview_app.tests.test_jobs import selection
from tools.streetview_app.tests.test_server import Provider, app_server, payload, post
from tools.streetview_engine.size_filter import read_size_filter_options


@pytest.mark.parametrize('enabled,ratio', [(False, .5), (True, .0001), (True, .5), (True, 10)])
def test_filter_survives_frozen_plan_config_and_cloud_bundle(tmp_path, enabled, ratio):
    options = dict(enabled=enabled,max_sigma_camera_radius_ratio=ratio)
    request = payload(size_filter=options)
    original = copy.deepcopy(request)
    frozen = freeze_selection(request,Provider())
    assert request == original
    assert frozen['size_filter'] == options
    normalized = validate_config(frozen)
    assert normalized['size_filter'] == options
    store = PlanStore(tmp_path/'plans')
    saved = store.save(frozen)
    assert store.get(saved['id'])['size_filter'] == store.list()[0]['size_filter'] == options
    packed,_ = source_bundle(normalized,{},(),SimpleNamespace(lease_seconds=90,timeout_seconds=600))
    with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as archive:
        assert json.load(archive.extractfile('config.json'))['size_filter'] == options
        assert archive.getmember('code/tools/streetview_engine/size_filter.py').isfile()


def test_missing_filter_stays_absent_and_legacy_config_unchanged(tmp_path):
    config = selection()
    original = copy.deepcopy(config)
    normalized = validate_config(config)
    assert config == original and 'size_filter' not in normalized
    frozen = freeze_selection(payload(),Provider())
    assert 'size_filter' not in frozen
    store = PlanStore(tmp_path/'plans'); saved = store.save(frozen)
    assert 'size_filter' not in store.get(saved['id']) and 'size_filter' not in store.list()[0]
    assert read_size_filter_options(normalized) == dict(enabled=False,max_sigma_camera_radius_ratio=.5)


@pytest.mark.parametrize('bad', [None,False,[],{'unknown':True},{'enabled':'false'},
    {'max_sigma_camera_radius_ratio':True},{'max_sigma_camera_radius_ratio':0},
    {'max_sigma_camera_radius_ratio':11},{'max_sigma_camera_radius_ratio':float('nan')}])
def test_invalid_filter_rejected_before_provider_or_job(bad):
    class NoProvider:
        def get_panorama(self,_):
            pytest.fail('Invalid filter must fail before provider requests')
    with pytest.raises(ValueError): freeze_selection(payload(size_filter=bad),NoProvider())
    with pytest.raises(ValueError): validate_config(dict(selection(),size_filter=bad))




def test_http_save_download_never_starts_generation(app_server):
    app,base = app_server
    options = dict(enabled=True,max_sigma_camera_radius_ratio=.25)
    code,saved = post(base+'/api/plans',payload(size_filter=options),origin=base)
    assert code == 201
    with urlopen(base+saved['download_url']) as response:
        assert json.load(response)['size_filter'] == options
    assert app.jobs.list() == []
    with pytest.raises(HTTPError) as failed:
        post(base+'/api/plans',payload(size_filter={'enabled':'yes'}),origin=base)
    assert failed.value.code == 400
    assert len(app.plans.list()) == 1
