import copy
from io import BytesIO
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.streetview_engine import collection
from tools.streetview_engine.imaging import FACES, cube_camera_to_station_cv, group_physical_stations, inside, sha256


def image_bytes(color, size=(512, 512)):
    output = BytesIO()
    Image.new('RGB', size, color).save(output, format='PNG')
    return output.getvalue()


def selection():
    return dict(center=dict(lat=12., lng=34.), radius_m=100., panorama_ids=['capture/A=='],
        panoramas=[dict(id='capture/A==', lat=12., lng=34., captured_at='2024-03-02 11:22:33', heading=30., projection='cubic')])


def source_response(url, timeout, session=None):
    if '/metadataV3/basic/' in url:
        return json.dumps(dict(id='capture/A==', latitude=12., longitude=34., altitude=20.,
            camera_angle=[0, 30, 0], proj_type='cubic', info=dict(photodate='2024-03-02 11:22:33'))).encode()
    x, y = map(int, url.rsplit('/', 2)[-2:])
    return image_bytes((x*50, y*60, 100))


class CollectionTests(unittest.TestCase):
    def test_native_tiles_stitch_exactly_and_ids_are_encoded(self):
        tiles = {(x, y): image_bytes((x*90, y*100, 50)) for y in range(2) for x in range(2)}
        face = collection.assemble_face(tiles)
        self.assertEqual(face.shape, (1024, 1024, 3))
        np.testing.assert_array_equal(face[100, 600], [90, 0, 50])
        np.testing.assert_array_equal(face[600, 100], [0, 100, 50])
        self.assertIn('capture%2FA%3D%3D', collection.tile_url('capture/A==', 'D', 1, 0))
        with self.assertRaises(ValueError):
            collection.tile_url('capture', 'F', True, 0)
        with self.assertRaises(ValueError):
            collection.assemble_face({(0, 0): image_bytes((0, 0, 0))})
        with self.assertRaises(ValueError):
            collection.assemble_face({(x,y): image_bytes((0,0,0), (256,256)) for y in range(2) for x in range(2)})

    def test_all_six_originals_provenance_and_resume_are_bound(self):
        config = selection(); untouched = copy.deepcopy(config)
        with tempfile.TemporaryDirectory() as directory, patch.object(collection, '_request', side_effect=source_response) as network:
            root = Path(directory)
            result = collection.run(config, root, {})
            self.assertEqual(network.call_count, 25)
            self.assertEqual(result['face_order'], list(FACES))
            self.assertEqual(set(result['stations'][0]['faces']), set(FACES))
            for face in FACES:
                item = result['stations'][0]['faces'][face]
                self.assertEqual(item['sha256'], sha256(root/item['file_path']))
                self.assertEqual(len(item['tiles']), 4)
                with Image.open(root/item['file_path']) as image:
                    self.assertEqual(image.size, (1024,1024))
                    self.assertEqual(image.getpixel((800,800)), (100,120,100))
            again = collection.run(config, root, {})
            self.assertEqual(network.call_count, 25)
            self.assertEqual(again, result)
            source = root/result['stations'][0]['metadata_source']['file_path']
            source.write_bytes(source.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                collection.run(config, root, {})
        self.assertEqual(config, untouched)

    def test_wrong_metadata_or_changed_selection_never_reuses_images(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(collection, '_request', side_effect=source_response):
            root = Path(directory); config = selection()
            collection.run(config, root, {})
            changed = copy.deepcopy(config); changed['radius_m'] = 90
            with self.assertRaisesRegex(ValueError, 'another selection'):
                collection.run(changed, root, {})
        with tempfile.TemporaryDirectory() as directory, patch.object(collection, '_request', return_value=b'{"id":"different"}'):
            with self.assertRaisesRegex(ValueError, 'ID does not match'):
                collection.run(selection(), Path(directory), {})
            self.assertFalse((Path(directory)/'collection/manifest.json').exists())

    def test_outside_selection_and_path_escape_rejected(self):
        config=selection();config['panoramas'][0]['lat']+=.1
        with self.assertRaisesRegex(ValueError, 'outside'):
            collection._validate_selection(config)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):inside(Path(directory),'../escape.png')

    def test_serial_parallel_bytes_hashes_and_order_match_without_extra_requests(self):
        with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as parallel_dir:
            with patch.object(collection, '_request', side_effect=source_response) as network:
                serial = collection.run(selection(), Path(serial_dir), {'max_workers': 1})
                self.assertEqual(network.call_count, 25)
            with patch.object(collection, '_request', side_effect=source_response) as network:
                parallel = collection.run(selection(), Path(parallel_dir), {'max_workers': 4})
                self.assertEqual(network.call_count, 25)
                collection.run(selection(), Path(parallel_dir), {'max_workers': 8})
                self.assertEqual(network.call_count, 25)
            self.assertEqual(serial['input_fingerprint'], parallel['input_fingerprint'])
            for face in FACES:
                a, b = serial['stations'][0]['faces'][face], parallel['stations'][0]['faces'][face]
                self.assertEqual(a['sha256'], b['sha256'])
                self.assertEqual([x['sha256'] for x in a['tiles']], [x['sha256'] for x in b['tiles']])
                self.assertEqual([(x['x'], x['y']) for x in b['tiles']], [(0,0),(1,0),(0,1),(1,1)])
                self.assertEqual((Path(serial_dir)/a['file_path']).read_bytes(), (Path(parallel_dir)/b['file_path']).read_bytes())

    def test_worker_limits_rejected_before_requests(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(collection.requests, 'Session') as session:
            for value in [0, 9, True, 2.5, '4']:
                with self.assertRaisesRegex(ValueError, 'max_workers'):
                    collection.run(selection(), Path(directory), {'max_workers': value})
            session.assert_not_called()


class ActualSessionTransportTests(unittest.TestCase):
    def test_http_keepalive_worker_bound_order_cache_and_error_cleanup(self):
        # Real local HTTP/1.1 requests exercise Requests' connection pool. No
        # provider traffic or assumption about remote wall-clock speed is needed.
        calls = []; ports = set(); lock = threading.Lock(); active = [0, 0]
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args): pass
            def do_GET(self):
                with lock:
                    calls.append(self.path); ports.add(self.client_address[1]); active[0] += 1; active[1] = max(active)
                try:
                    time.sleep(.02)
                    payload = self.path.encode()
                    self.send_response(503 if self.path == '/fail' else 200)
                    self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
                finally:
                    with lock: active[0] -= 1
        original_session = collection.requests.Session; sessions = []
        def session_factory():
            session = original_session(); original_close = session.close; session.closed_for_test = False
            def close():
                session.closed_for_test = True; original_close()
            session.close = close; sessions.append(session); return session
        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server, tempfile.TemporaryDirectory() as directory:
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True); thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with patch.object(collection.requests, 'Session', side_effect=session_factory):
                    with collection._RequestPool(2) as network:
                        for face in ('F','R'):
                            paths = [f'/{face}/{i}' for i in range(4)]
                            arguments = [(Path(directory), f'{face}_{i}.bin', url+path, 2) for i,path in enumerate(paths)]
                            result = network.fetch_face(arguments)
                            self.assertEqual([content for content, _ in result], [path.encode() for path in paths])
                        self.assertEqual(len(calls), 8)
                        network.fetch_face(arguments)
                        self.assertEqual(len(calls), 8)
                    self.assertEqual(len(sessions), 2); self.assertTrue(all(session.closed_for_test for session in sessions))
                    self.assertEqual(len(ports), 2); self.assertEqual(active[1], 2)
                    with self.assertRaises(collection.requests.HTTPError):
                        with collection._RequestPool(1) as network:
                            network.fetch(Path(directory), 'failure.bin', url+'/fail', 2)
                    self.assertEqual(calls.count('/fail'), 1)
                    self.assertTrue(all(session.closed_for_test for session in sessions))
                    self.assertFalse((Path(directory)/'failure.bin').exists())
                    self.assertFalse((Path(directory)/'failure.bin.source.json').exists())
            finally:
                server.shutdown(); thread.join(timeout=2)


class CubeAndGroupingTests(unittest.TestCase):
    def test_cube_rays_are_proper_and_seams_agree(self):
        expected=dict(F=[0,0,1],R=[1,0,0],B=[0,0,-1],L=[-1,0,0],U=[0,-1,0],D=[0,1,0])
        for face in FACES:
            matrix=cube_camera_to_station_cv(face)
            np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3))
            self.assertAlmostEqual(np.linalg.det(matrix[:3,:3]),1.)
            np.testing.assert_array_equal(matrix[:3,2],expected[face])
        # Front right edge and right-face left edge describe one world ray.
        f=cube_camera_to_station_cv('F')[:3,:3]
        r=cube_camera_to_station_cv('R')[:3,:3]
        u=cube_camera_to_station_cv('U')[:3,:3]
        d=cube_camera_to_station_cv('D')[:3,:3]
        np.testing.assert_array_equal(f@[1,0,1],r@[-1,0,1])
        np.testing.assert_array_equal(f@[0,-1,1],u@[0,1,1])
        np.testing.assert_array_equal(f@[0,1,1],d@[0,-1,1])

    def test_grouping_is_order_independent_and_does_not_chain(self):
        degree_per_meter=180/(math.pi*6371008.8)
        points=[dict(id=key,lat=0.,lng=distance*degree_per_meter) for key,distance in [('a',0),('b',.2),('c',.4),('d',5)]]
        original=copy.deepcopy(points)
        a=group_physical_stations(points,.25);b=group_physical_stations(list(reversed(points)),.25)
        self.assertEqual(a,b);self.assertEqual(a['a'],a['b']);self.assertNotEqual(a['a'],a['c']);self.assertNotEqual(a['c'],a['d'])
        self.assertEqual(points,original)
        with self.assertRaises(ValueError):group_physical_stations(points,float('nan'))
        with self.assertRaises(ValueError):group_physical_stations(points+[points[0]],.25)


if __name__=='__main__':unittest.main()
