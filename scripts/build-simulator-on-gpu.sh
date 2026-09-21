#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

config = json.loads(Path('/var/lib/physicalai/live-runtime.json').read_text())
update = json.loads('__CODE_UPDATE_JSON__')
subprocess.run(['docker', 'stop', '--time', '10', 'physicalai-simulator'], check=True)
subprocess.run(['az', 'login', '--identity', '--client-id', config['AZURE_CLIENT_ID'],
                '--allow-no-subscriptions', '--output', 'none'], check=True)
subprocess.run(['az', 'acr', 'login', '--name', config['REGISTRY_NAME']], check=True)
subprocess.run(['docker', 'pull', update['code_image']], check=True)
build = Path('/var/lib/physicalai/builds') / update['code_digest'].split(':')[-1]
build.mkdir(parents=True, exist_ok=True)
(build / 'app').mkdir(exist_ok=True)
extraction = subprocess.run(
    ['docker', 'create', update['code_image'], '/not-executed'],
    text=True, capture_output=True, check=True,
).stdout.strip()
try:
    subprocess.run(['docker', 'cp', extraction + ':/app/.', str(build / 'app')], check=True)
    subprocess.run(['docker', 'cp', extraction + ':/build/Dockerfile', str(build / 'Dockerfile')], check=True)
    if update.get('runtime_version') == '6.0.0':
        subprocess.run(['docker', 'cp', extraction + ':/build/Dockerfile.six',
                        str(build / 'Dockerfile.six')], check=True)
finally:
    subprocess.run(['docker', 'rm', extraction], check=True)
if update.get('runtime_version') == '6.0.0':
    subprocess.run(['docker', 'build', '--build-arg', 'ASSET_IMAGE=' + update['base_image'],
                    '--file', str(build / 'Dockerfile.six'),
                    '--tag', update['target_image'], str(build / 'app')], check=True)
else:
    subprocess.run(['docker', 'build', '--build-arg', 'SIMULATOR_BASE_IMAGE=' + update['base_image'],
                    '--tag', update['target_image'], str(build)], check=True)
subprocess.run(['docker', 'push', update['target_image']], check=True)
result = json.loads(subprocess.run(['docker', 'image', 'inspect', update['target_image']],
                                  text=True, capture_output=True, check=True).stdout)[0]
print('PHYSICALAI_GPU_IMAGE ' + json.dumps({
    'image_id': result['Id'], 'repo_digests': result['RepoDigests'],
    'base_image': update['base_image'], 'code_image': update['code_image'],
}))
PY
