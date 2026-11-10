#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the fork's Swift shell and package it with a compatible oMLX runtime."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
MAC = ROOT / 'apps/omlx-mac'


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor', type=Path, required=True,
                        help='Installed oMLX 0.7.3 app supplying Python, MLX and compiled assets')
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--skip-compile', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'dist/meeting-workbench')
    args = parser.parse_args()
    donor = args.donor.expanduser().resolve()
    donor_info = plistlib.loads((donor/'Contents/Info.plist').read_bytes())
    if donor_info.get('CFBundleShortVersionString') != '0.7.3':
        parser.error('Use a 0.7.3 donor to match this fork and its MLX ABI.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = ROOT/'.artifacts/native-build'
    state.mkdir(parents=True, exist_ok=True)
    flags = ['--package-path', str(MAC), '--cache-path', str(state/'cache'),
             '--config-path', str(state/'config'), '--security-path', str(state/'security'),
             '--sdk', str(args.sdk), '-c', 'release', '-j', '2']
    if not args.skip_compile:
        print('Compiling the native Swift shell…', flush=True)
        run('/usr/bin/swift', 'build', *flags)
    binpath = Path(subprocess.check_output(['/usr/bin/swift', 'build', *flags, '--show-bin-path'],
                                          text=True).strip())
    binary = binpath/'oMLXMeeting'
    if not binary.is_file():
        parser.error('Native executable is missing; compile before packaging.')
    app = output/'oMLX Meeting Workbench.app'
    dmg = output/'oMLX-Meeting-Workbench-0.7.3-arm64.dmg'
    with tempfile.TemporaryDirectory(prefix='stage-', dir=output) as folder:
        stage = Path(folder)
        staged_app = stage/app.name
        print('Copying the compatible Python/MLX runtime and compiled assets…', flush=True)
        run('/usr/bin/ditto', '--noqtn', donor, staged_app)
        contents = staged_app/'Contents'
        resources = contents/'Resources'
        old_executable = contents/'MacOS'/donor_info['CFBundleExecutable']
        old_executable.unlink()
        shutil.copy2(binary, contents/'MacOS/oMLXMeeting')
        info = dict(donor_info)
        info.update(CFBundleIdentifier='io.github.dmytroshkvyra.omlx-meeting',
                    CFBundleExecutable='oMLXMeeting', CFBundleName='oMLX Meeting Workbench',
                    CFBundleDisplayName='oMLX Meeting Workbench',
                    CFBundleShortVersionString='0.7.3', CFBundleVersion='20261110.1',
                    OMLXDistributionFlavor='meeting',
                    NSMicrophoneUsageDescription='Record your microphone for meeting transcription and voice input.')
        info['CFBundleURLTypes'] = [{'CFBundleURLName': info['CFBundleIdentifier']+'.url',
                                    'CFBundleURLSchemes': ['omlxmeeting']}]
        (contents/'Info.plist').write_bytes(plistlib.dumps(info))
        for source, destination in [(ROOT/'omlx', resources/'omlx'),
                (ROOT/'apps/omlx-web/omlx_web', resources/'omlx_web')]:
            # Preserve matching donor native kernels, while replacing Python/UI
            # source. Matching 0.7.3 runtime keeps the compiled MLX ABI unchanged.
            run('/usr/bin/rsync', '-a', '--delete', '--exclude=__pycache__', '--exclude=*.pyc',
                '--exclude=*.so', '--exclude=*.dylib', '--exclude=*.metallib',
                '--exclude=csrc', '--exclude=include', str(source)+'/', str(destination)+'/')
        for bundle in binpath.glob('*.bundle'):
            shutil.copytree(bundle, resources/bundle.name, dirs_exist_ok=True)
        for name in ('omlx-cli', 'omlx-cluster-python'):
            wrapper = contents/'MacOS'/name
            if wrapper.exists():
                text = wrapper.read_text()
                text = text.replace('set -eu\n', '''set -eu
export OMLX_BASE_PATH="${OMLX_BASE_PATH:-$HOME/.omlx-meeting}"
export OMLX_PORT="${OMLX_PORT:-18081}"
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"
''', 1)
                wrapper.write_text(text)
                wrapper.chmod(0o755)
                run('/usr/bin/codesign', '--force', '--sign', '-', wrapper)
        metadata = {'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    'source_dirty': bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)),
                    'native_shell': 'compiled from fork source with SwiftPM',
                    'runtime_donor': donor.name, 'architecture': 'arm64',
                    'signature': 'local ad-hoc; not notarized by Apple'}
        (resources/'meeting-build.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('Signing and verifying the application…', flush=True)
        run('/usr/bin/codesign', '--force', '--sign', '-', '--entitlements',
            MAC/'Resources/oMLX.entitlements', staged_app)
        run('/usr/bin/codesign', '--verify', '--deep', '--strict', staged_app)
        if app.exists():
            shutil.rmtree(app)
        staged_app.rename(app)
    with tempfile.TemporaryDirectory(prefix='dmg-', dir=output) as folder:
        image_root = Path(folder)
        # The app is copied into the image; the Applications link enables drag-install.
        run('/usr/bin/ditto', '--noqtn', app, image_root/app.name)
        (image_root/'Applications').symlink_to('/Applications')
        print('Creating the disk image…', flush=True)
        run('/usr/bin/hdiutil', 'create', '-ov', '-volname', 'oMLX Meeting Workbench',
            '-srcfolder', image_root, '-format', 'UDZO', dmg)
    print(json.dumps({'app': str(app), 'dmg': str(dmg)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
