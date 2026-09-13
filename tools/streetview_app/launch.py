"""Open the local GUI, reusing its server or starting a hidden helper."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / 'data'


def ready(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
        return False
    try:
        with urlopen(url.rstrip('/') + '/api/status', timeout=2) as response:
            return json.load(response).get('name') == 'Streetview to PLY'
    except Exception:
        return False


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    record = DATA / 'server.json'
    if record.exists():
        try:
            url = json.loads(record.read_text(encoding='utf8'))['url']
            if ready(url):
                webbrowser.open(url)
                return
        except (OSError, ValueError, KeyError):
            pass
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 8765))
        except OSError:
            probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    url = f'http://127.0.0.1:{port}/'
    with (DATA / 'server.log').open('ab') as log:
        options = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if os.name == 'nt':
            options['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options['start_new_session'] = True
        process = subprocess.Popen([sys.executable, '-B', '-m', 'tools.streetview_app.server', '--port', str(port), '--data-dir', str(DATA)], **options)
    (DATA / 'server_process.json').write_text(json.dumps(dict(pid=process.pid, url=url)), encoding='utf8')
    for _ in range(60):
        if ready(url):
            webbrowser.open(url)
            return
        if process.poll() is not None:
            break
        time.sleep(.25)
    raise RuntimeError('Streetview server did not start; see tools/streetview_app/data/server.log')


if __name__ == '__main__':
    main()
