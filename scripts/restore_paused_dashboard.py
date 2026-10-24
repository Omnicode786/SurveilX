"""Restore the paused local dashboard after this owned training runner exits."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import psutil


def restore(campaign, runner_pid, created):
    workspace = Path(__file__).resolve().parents[1]
    campaign = Path(campaign).resolve()
    if campaign.parent != workspace / 'data/generations':
        raise ValueError('Expected a local campaign directory')
    report = campaign / 'dashboard-restoration.json'
    try:
        runner = psutil.Process(runner_pid)
        if runner.create_time() != created or 'scripts.train_accuracy_campaign' not in runner.cmdline():
            raise ValueError('Runner identity changed; refusing to watch an unrelated process')
        if str(campaign.relative_to(workspace) / 'plan.json').replace('\\', '/') not in [
                arg.replace('\\', '/') for arg in runner.cmdline()]:
            raise ValueError('Runner belongs to a different campaign')
        runner.wait()
        with socket.socket() as connection:
            connection.settimeout(2)
            if connection.connect_ex(('127.0.0.1', 8000)) == 0:
                report.write_text(json.dumps({'state': 'already_available', 'checked': time.time()}, indent=2))
                return
        saved = json.loads((campaign / 'paused-api-restore.json').read_text())
        if Path(saved['cwd']).resolve() != workspace or saved['command'][1:4] != ['-m', 'uvicorn', 'surveilx.api:app']:
            raise ValueError('Saved dashboard command is incompatible')
        env = os.environ.copy()
        env.update(saved['environment'])
        env.update(CUDA_VISIBLE_DEVICES='-1', SURVEILX_START_WORKERS='false', SURVEILX_DEMO='false',
            OPENBLAS_NUM_THREADS='3', OMP_NUM_THREADS='3', MKL_NUM_THREADS='3')
        command = [str(workspace / '.venv/Scripts/python.exe'), *saved['command'][1:]]
        with (campaign / 'restored-api.log').open('a', encoding='utf-8') as stdout, (
                campaign / 'restored-api-error.log').open('a', encoding='utf-8') as stderr:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        report.write_text(json.dumps({'state': 'launched', 'time': time.time(), 'pid': process.pid,
            'command': command, 'workers_paused': True, 'cuda_visible_devices': '-1'}, indent=2))
    except Exception as exc:
        report.write_text(json.dumps({'state': 'failed', 'error': str(exc), 'time': time.time()}, indent=2))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign')
    parser.add_argument('runner_pid', type=int)
    parser.add_argument('created', type=float)
    args = parser.parse_args()
    restore(args.campaign, args.runner_pid, args.created)
