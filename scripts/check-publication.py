"""Check a staged snapshot or a Git revision for accidental private material.

This is a conservative repository-specific guard, not a complete secret scanner.
It reports categories and filenames without printing potential secret values.
"""
import argparse
from pathlib import PurePosixPath
import re
import subprocess


PRIVATE_ROOTS = {
    '.portal', '.tools', '.venv', 'inputExamples', 'outputs', 'reports',
    'backups', 'skill-identification-handover', 'wifi-reconnect',
}
MEDIA = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm', '.jpg', '.jpeg', '.png', '.gif'}
APPROVED_MEDIA = {
    'docs/assets/red-tuck-backs.mp4', 'docs/assets/red-tuck-backs.gif',
    'docs/assets/upload-demo.mp4', 'docs/assets/analysis-demo.mp4',
}
PATTERNS = {
    'private key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'GitHub token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})',
    'AWS access key': rb'\bAKIA[0-9A-Z]{16}\b',
    'personal Windows path': rb'(?i)C:[\\/]+Users[\\/]+[^\\/\s"\x27]+',
    'personal Unix path': rb'/(?:Users|home)/[A-Za-z0-9_.-]+/',
    'credential in URL': rb'https?://[^\s/:]+:[^\s/@]+@',
}


def git(*args):
    return subprocess.check_output(['git', *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', help='Revision to check; default is the staged index')
    args = parser.parse_args()
    if args.ref:
        paths = git('ls-tree', '-r', '--name-only', '-z', args.ref)
    else:
        paths = git('ls-files', '-z')
    issues = []
    files = [name.decode('utf-8') for name in paths.split(b'\0') if name]
    for name in files:
        path = PurePosixPath(name)
        if path.parts[0] in PRIVATE_ROOTS or name in {'PLAN.md', 'skills.csv'}:
            issues.append((name, 'local-only file'))
        if path.name.startswith('.env') and path.name != '.env.example':
            issues.append((name, 'environment file'))
        if path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx', '.pth', '.pt', '.bundle'}:
            issues.append((name, 'private key, checkpoint or archive'))
        if path.suffix.lower() in MEDIA and name not in APPROVED_MEDIA:
            issues.append((name, 'unreviewed media'))
        data = git('show', f'{args.ref or ""}:{name}')
        if len(data) > 10 * 1024 * 1024:
            issues.append((name, 'file exceeds 10 MiB publication limit'))
        if path.suffix.lower() not in MEDIA:
            for label, pattern in PATTERNS.items():
                if re.search(pattern, data):
                    issues.append((name, label))
            emails = re.findall(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', data)
            if any(not email.endswith((b'@users.noreply.github.com', b'@example.com')) for email in emails):
                issues.append((name, 'email address requiring review'))
    for name, reason in issues:
        print(f'{reason}: {name}')
    if issues:
        raise SystemExit(1)
    print(f'Publication guard passed: {len(files)} files checked.')


if __name__ == '__main__':
    main()
