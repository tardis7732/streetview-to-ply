"""Finite positive collection radius, without a scene-size ceiling."""
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tools.streetview_app.jobs import validate_config
from tools.streetview_app.plans import selection_settings
from tools.streetview_app.tests.test_discovery_pagination import ChainProvider
from tools.streetview_app.tests.test_jobs import selection
from tools.streetview_app.tests.test_server import app_server, payload, post
from tools.streetview_engine.collection import _validate_selection


@pytest.mark.parametrize('radius', [500, 501, 2000, 10000.5])
def test_large_finite_radius_is_preserved(tmp_path, radius):
    config = dict(selection(), radius_m=radius)
    assert selection_settings(config)['radius_m'] == radius
    assert validate_config(config)['radius_m'] == radius
    assert _validate_selection(config)[0] == config['panorama_ids']
    assert ChainProvider(tmp_path, count=2).discover(35, 128, radius)['radius_m'] == radius


@pytest.mark.parametrize('radius', [9, True, float('inf'), float('nan')])
def test_invalid_radius_rejected_before_collection_or_discovery(tmp_path, radius):
    config = dict(selection(), radius_m=radius)
    for validate in (selection_settings, validate_config, _validate_selection):
        with pytest.raises((ValueError, TypeError)):
            validate(config)
    provider = ChainProvider(tmp_path, count=2)
    with pytest.raises(ValueError):
        provider.discover(35, 128, radius)
    assert provider.calls == []


def test_server_preserves_large_radius_and_rejects_invalid_plan(app_server):
    app, base = app_server
    status, saved = post(base+'/api/plans', payload(radius_m=2000), origin=base)
    assert status == 201 and saved['plan']['radius_m'] == 2000
    old_plans = app.plans.list()
    with urlopen(base+'/api/discover?lat=35&lng=128&radius_m=2000') as response:
        assert response.status == 200
    with pytest.raises(HTTPError) as error:
        post(base+'/api/plans', payload(radius_m=9), origin=base)
    assert error.value.code == 400
    assert app.plans.list() == old_plans
