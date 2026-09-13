"""No-network remote lifecycle tests; POSIX cases launch only a fake CLI.

All files and subprocesses belong to TemporaryDirectory fixtures. No SSH,
provider request, model loading, cloud setup or GPU operation is performed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.streetview_app import remote_jobs
from tools.streetview_engine import remote_worker as worker


def tar_bytes(entries):
    """Entries: (name, bytes) for ordinary files, or a configured TarInfo."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for value in entries:
            if isinstance(value, tarfile.TarInfo):
                archive.addfile(value)
            else:
                name, payload = value
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def archive_file(root, entries):
    path = root / 'fixture_results.tar'
    path.write_bytes(tar_bytes(entries))
    return path


def source_workspace(root):
    source = root / 'source'
    paths = ['tools/__init__.py', 'tools/streetview_engine/__init__.py',
             'tools/streetview_engine/__main__.py', 'tools/streetview_engine/remote_worker.py',
             'tools/streetview_geometry/__init__.py', 'tools/streetview_geometry/ground.py',
             *('tools/streetview_app/' + name for name in ('__init__.py', 'jobs.py', 'plans.py', 'provider.py', 'recipes.py', 'reuse.py'))]
    for name in paths:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source fixture\n', encoding='utf8')
    return source, paths


class ArchiveTests(unittest.TestCase):
    def test_input_archive_allows_only_confined_ordinary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker.extract_archive(io.BytesIO(tar_bytes([('code/tools/module.py', b'pass\n'), ('config.json', b'{}')])), root)
            self.assertEqual((root/'code/tools/module.py').read_bytes(), b'pass\n')
        for name in ('../escaped.txt', '/absolute.txt', 'code/../../escaped.txt', 'code\\escape.txt', 'C:/escape.txt'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    worker.extract_archive(io.BytesIO(tar_bytes([(name, b'payload')])), root/'job')
                self.assertFalse((root/'escaped.txt').exists())

    def test_input_archive_rejects_links_directories_duplicates_and_size(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                member = tarfile.TarInfo('code/item'); member.type = kind; member.linkname = '../outside'
                with self.assertRaises(ValueError):
                    worker.extract_archive(io.BytesIO(tar_bytes([member])), Path(directory))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                worker.extract_archive(io.BytesIO(tar_bytes([('same', b'a'), ('./same', b'b')])), Path(directory))
            self.assertEqual((Path(directory)/'same').read_bytes(), b'a')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                worker.extract_archive(io.BytesIO(tar_bytes([('large', b'1234')])), Path(directory), limit=3)
            self.assertFalse((Path(directory)/'large').exists())

    def test_results_confine_paths_and_require_ordinary_allowed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'job'; output.mkdir()
            archive = archive_file(root, [('logs/stage.log', b'actual fixture log'), ('export/scene.ply', b'fixture')])
            remote_jobs.extract_results(archive, output)
            self.assertEqual((output/'logs/stage.log').read_bytes(), b'actual fixture log')
        for name in ('logs/../../escaped.txt', '/export/absolute.ply', 'logs\\escaped.txt', 'export/C:escape', 'config.json', '.ssh/id_ed25519'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); output = root/'job'; output.mkdir()
                archive = archive_file(root, [(name, b'bad')])
                with self.assertRaises(ValueError): remote_jobs.extract_results(archive, output)
                self.assertFalse((root/'escaped.txt').exists())
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); member = tarfile.TarInfo('export/link'); member.type = kind; member.linkname = '../secret'
                with self.assertRaises(ValueError): remote_jobs.extract_results(archive_file(root, [member]), root/'job')

    @unittest.skipUnless(os.name == 'posix', 'Symlink confinement checked on the POSIX cloud test host')
    def test_results_cannot_follow_preexisting_receiving_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'job';(output/'export').mkdir(parents=True)
            outside=root/'outside.txt';outside.write_bytes(b'keep unchanged')
            (output/'export/scene.ply.receiving').symlink_to(outside)
            archive=archive_file(root,[('export/scene.ply',b'new artifact')])
            try:
                remote_jobs.extract_results(archive,output)
            except (ValueError, FileExistsError):
                pass
            self.assertEqual(outside.read_bytes(),b'keep unchanged','Temporary extraction path followed a symlink outside the job')

    @unittest.skipUnless(os.name == 'posix', 'Symlink confinement checked on the POSIX cloud test host')
    def test_existing_symlink_directory_is_not_an_extraction_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);outside=root/'outside';outside.mkdir();job=root/'job';job.mkdir()
            (job/'logs').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):
                worker.extract_archive(io.BytesIO(tar_bytes([('logs/escape',b'bad')])),job)
            with self.assertRaises(ValueError):
                remote_jobs.extract_results(archive_file(root,[('logs/escape',b'bad')]),job)
            self.assertFalse((outside/'escape').exists())


class SourceAndOwnerTests(unittest.TestCase):
    def test_pidfd_signal_requires_current_start_identity_and_owned_ancestry(self):
        owner=worker.OwnedDescendants.__new__(worker.OwnedDescendants)
        record=dict(pid=123,start=456,ppid=1,state='S')
        with patch.object(worker.os,'pidfd_open',return_value=999,create=True), patch.object(worker.os,'close') as close, patch.object(worker.signal,'pidfd_send_signal',create=True) as send:
            with patch.object(owner,'_record',return_value=dict(record,start=457)), patch.object(owner,'snapshot',return_value={123:record}):
                owner.send(record,signal.SIGTERM)
            send.assert_not_called();close.assert_called_with(999)
            with patch.object(owner,'_record',return_value=record),patch.object(owner,'snapshot',return_value={}):
                owner.send(record,signal.SIGTERM)
            send.assert_not_called()
            with patch.object(owner,'_record',return_value=record),patch.object(owner,'snapshot',return_value={123:record}):
                owner.send(record,signal.SIGTERM)
            send.assert_called_once_with(999,signal.SIGTERM)

    def test_source_hash_manifest_is_required_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'job';job.mkdir();(job/'code').mkdir()
            source=job/'code/main.py';source.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError,'nonempty'):
                worker.verify_code(job,{})
            recipe=dict(code_sha256={'code/main.py':hashlib.sha256(b'original').hexdigest()})
            worker.verify_code(job,recipe)
            source.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                worker.verify_code(job,recipe)
            source.write_bytes(b'original');(job/'code/unlisted.py').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'roster differs'):
                worker.verify_code(job,recipe)

    def test_source_bundle_excludes_data_credentials_caches_and_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workspace,paths=source_workspace(root)
            private=[('.env',b'PRIVATE_ENV_MARKER'),('.ssh/id_ed25519',b'PRIVATE_KEY_MARKER'),
                     ('tools/streetview_app/data/remote_owner.json',b'OWNER_TOKEN_MARKER'),
                     ('tools/streetview_engine/data/scene.ply',b'PRIVATE_DATA_MARKER'),
                     ('tools/streetview_geometry/__pycache__/module.pyc',b'BYTECODE_MARKER'),
                     ('tools/streetview_engine/tests/test_secret_fixture.py',b'TEST_FIXTURE_MARKER')]
            for name,data in private:
                path=workspace/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            stages=[SimpleNamespace(name=name,outputs=(name+'/manifest.json',)) for name in worker.STAGES]
            profile=SimpleNamespace(lease_seconds=30,timeout_seconds=60)
            config=dict(panorama_ids=['synthetic_capture']);settings=dict(cpu_threads=2)
            blob,hashes=remote_jobs.source_bundle(config,settings,stages,profile,workspace=workspace)
            with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
                members=archive.getmembers();names={member.name for member in members}
                self.assertEqual(names,{'code/'+name for name in paths}|{'config.json','settings.json','recipe.json'})
                self.assertTrue(all(member.isfile() for member in members))
                contents={member.name:archive.extractfile(member).read() for member in members}
            for name,digest in hashes.items():self.assertEqual(hashlib.sha256(contents[name]).hexdigest(),digest)
            self.assertEqual(json.loads(contents['recipe.json'])['code_sha256'],hashes)
            self.assertEqual(json.loads(contents['config.json']),config)
            for _,secret in private:self.assertFalse(any(secret in payload for payload in contents.values()))

    @unittest.skipUnless(os.name == 'posix', 'Source symlink refusal tested on POSIX')
    def test_source_bundle_refuses_source_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workspace,_=source_workspace(root);source=workspace/'tools/streetview_geometry/ground.py'
            source.unlink();secret=root/'secret.txt';secret.write_text('private');source.symlink_to(secret)
            with self.assertRaises(ValueError):
                remote_jobs.source_bundle({}, {}, [], SimpleNamespace(lease_seconds=30,timeout_seconds=60),workspace=workspace)

    def test_invalid_owner_cannot_read_renew_start_or_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory);token=secrets.token_hex(32)
            worker.write_json(job/'owner.json',dict(token_sha256=hashlib.sha256(token.encode()).hexdigest()))
            worker.write_json(job/'remote_state.json',dict(status='prepared',stage=None))
            (job/'lease').touch();os.utime(job/'lease',(100,100));before=(job/'lease').stat().st_mtime_ns
            wrong=secrets.token_hex(32)
            for operation in (lambda:worker.authenticate(job,wrong),lambda:worker.status(job,wrong),lambda:worker.start_job(job,wrong),lambda:worker.pack_results(job,wrong)):
                with self.assertRaises(PermissionError):operation()
            self.assertEqual((job/'lease').stat().st_mtime_ns,before)
            self.assertFalse((job/'started.lock').exists())
            for invalid in ('',token[:-1],'g'*64,token+'\n'):
                with self.assertRaises(ValueError):worker.authenticate(job,invalid)
            self.assertEqual(worker.status(job,token,renew=False)['status'],'prepared')
            self.assertEqual((job/'lease').stat().st_mtime_ns,before)
            with self.assertRaises(RuntimeError):worker.pack_results(job,token)

    def test_inspect_cli_does_not_renew_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/secrets.token_hex(16);job.mkdir();token=secrets.token_hex(32)
            worker.write_json(job/'owner.json',dict(token_sha256=hashlib.sha256(token.encode()).hexdigest()))
            worker.write_json(job/'remote_state.json',dict(status='prepared',stage=None))
            (job/'lease').touch();os.utime(job/'lease',(100,100));before=(job/'lease').stat().st_mtime_ns
            command=[sys.executable,str(Path(worker.__file__)), '--root',str(root),'--job-id',job.name,'--token',token,'inspect']
            result=subprocess.run(command,capture_output=True,timeout=5,check=True)
            self.assertEqual(json.loads(result.stdout)['status'],'prepared')
            self.assertEqual((job/'lease').stat().st_mtime_ns,before)


# A genuine executable module, but every output and process is synthetic. The
# descendant has a cooperative fixture cleanup flag even if production group
# cancellation is broken, so a failing test cannot leave a daemon behind.
FAKE_ENGINE = r'''
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--job-config');p.add_argument('--job-dir');p.add_argument('--settings');a=p.parse_args()
root=Path(a.job_dir);config=json.loads(Path(a.job_config).read_text());mode=config.get('fixture_mode','complete')
if a.stage=='collect' and mode in ('block','orphan'):
 child_code="""import os,signal,sys,time\nfrom pathlib import Path\nr=Path(sys.argv[1]);signal.signal(signal.SIGTERM,signal.SIG_IGN)\n(r/'fixture_child.pid').write_text(str(os.getpid()))\nwhile not (r/'fixture_cleanup').exists():\n (r/'fixture_heartbeat').write_text(str(time.time()))\n time.sleep(.05)\n"""
 child=subprocess.Popen([sys.executable,'-c',child_code,str(root)])
 (root/'fixture_parent.pid').write_text(str(os.getpid()))
 deadline=time.monotonic()+3
 while not (root/'fixture_heartbeat').exists():
  if time.monotonic()>deadline:raise RuntimeError('fixture child failed to start')
  time.sleep(.01)
 if mode=='block':
  while not (root/'fixture_cleanup').exists():time.sleep(.05)
  child.wait(timeout=5)
if a.stage=='collect' and mode=='fail':raise SystemExit(7)
recipe=json.loads((root/'recipe.json').read_text())
stage=next(row for row in recipe['stages'] if row['name']==a.stage)
if mode!='missing':
 for relative in stage['outputs']:
  target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps({'fixture':True,'stage':a.stage}))
print('synthetic fixture stage '+a.stage,flush=True)
'''


@unittest.skipUnless(sys.platform == 'linux', 'Real supervisor pidfd/subreaper tests require Linux; run on cloud CPU')
class PosixSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.job=self.root/secrets.token_hex(16);self.token=secrets.token_hex(32);self.supervisor_pid=None

    def tearDown(self):
        if self.job.exists():
            (self.job/'fixture_cleanup').touch()
            (self.job/'cancel').touch()
            deadline=time.monotonic()+8
            while self.supervisor_pid is not None and time.monotonic()<deadline:
                try:
                    done,_=os.waitpid(self.supervisor_pid,os.WNOHANG)
                    if done:break
                except ChildProcessError:break
                time.sleep(.05)
        self.temporary.cleanup()

    def prepare(self,mode='complete'):
        recipe=dict(stages=[dict(name=name,outputs=[name+'/manifest.json']) for name in worker.STAGES],lease_seconds=15,timeout_seconds=30,cpu_threads=1)
        entries=[('config.json',json.dumps(dict(fixture_mode=mode)).encode()),('settings.json',b'{}'),
                 ('code/tools/__init__.py',b''),('code/tools/streetview_engine/__init__.py',b''),
                 ('code/tools/streetview_engine/__main__.py',FAKE_ENGINE.encode()),
                 ('code/tools/streetview_engine/remote_worker.py',Path(worker.__file__).read_bytes())]
        recipe['code_sha256']={name:hashlib.sha256(content).hexdigest() for name,content in entries if name.startswith('code/')}
        entries.append(('recipe.json',json.dumps(recipe).encode()))
        worker.init_job(self.job,self.token,io.BytesIO(tar_bytes(entries)))
        start=worker.start_job(self.job,self.token);self.supervisor_pid=start['supervisor_pid']

    def await_file(self,name,timeout=8):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if (self.job/name).exists():return
            time.sleep(.05)
        self.fail('Fixture did not reach '+name+'; state='+str(worker.status(self.job,self.token,renew=False)))

    def terminal(self,timeout=12):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            value=worker.status(self.job,self.token,renew=False)
            if value['status'] in ('completed','failed','cancelled'):return value
            time.sleep(.05)
        self.fail('Supervisor did not reach a bounded terminal state')

    def assert_descendant_stopped(self):
        marker=self.job/'fixture_heartbeat'
        before=marker.read_bytes();time.sleep(.25)
        self.assertEqual(marker.read_bytes(),before,'Owned grandchild continued after supervisor reached terminal state')

    def test_actual_five_stage_completion_and_result_archive(self):
        self.prepare();result=self.terminal()
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual([row['name'] for row in result['stages']],list(worker.STAGES))
        self.assertTrue(all(row['status']=='completed' and row['exit_code']==0 for row in result['stages']))
        packed=worker.pack_results(self.job,self.token)
        self.assertEqual(Path(packed['path']).parent,self.job)
        self.assertEqual(hashlib.sha256(Path(packed['path']).read_bytes()).hexdigest(),packed['sha256'])
        with tarfile.open(packed['path']) as archive:
            names={member.name for member in archive.getmembers()}
        self.assertIn('export/manifest.json',names);self.assertIn('logs/collect.log',names)
        self.assertNotIn('owner.json',names);self.assertNotIn('config.json',names)
        with self.assertRaises(FileExistsError):worker.start_job(self.job,self.token)

    def test_cancel_stops_owned_term_ignoring_descendant(self):
        unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            self.prepare('block');self.await_file('fixture_heartbeat')
            command=[sys.executable,str(Path(worker.__file__)), '--root',str(self.root),'--job-id',self.job.name,'--token',self.token,'cancel']
            request=subprocess.run(command,capture_output=True,timeout=5,check=True)
            self.assertIn('status',json.loads(request.stdout))
            result=self.terminal();self.assertEqual(result['status'],'cancelled',result)
            self.assert_descendant_stopped()
            self.assertIsNone(unrelated.poll(),'Unrelated sibling process was signalled by job cleanup')
        finally:
            unrelated.terminate();unrelated.wait(timeout=5)

    def test_expired_lease_stops_owned_compute_without_renewal(self):
        self.prepare('block');self.await_file('fixture_heartbeat')
        old=time.time()-100;os.utime(self.job/'lease',(old,old))
        result=self.terminal();self.assertEqual(result['status'],'failed',result)
        self.assertIn('lease expired',result.get('error','').lower())
        self.assert_descendant_stopped()

    def test_successful_leader_exit_still_cleans_orphan_descendant(self):
        self.prepare('orphan');self.await_file('fixture_heartbeat')
        result=self.terminal();self.assertEqual(result['status'],'completed',result)
        self.assert_descendant_stopped()

    def test_failed_stage_cannot_complete(self):
        self.prepare('fail');result=self.terminal()
        self.assertEqual(result['status'],'failed');self.assertEqual(result['stages'][0]['exit_code'],7)
        self.assertEqual(len(result['stages']),1)

    def test_missing_stage_artifact_cannot_complete(self):
        self.prepare('missing');result=self.terminal()
        self.assertEqual(result['status'],'failed');self.assertIn('did not produce',result.get('error',''))
        self.assertEqual(len(result['stages']),1)


if __name__=='__main__':unittest.main()
