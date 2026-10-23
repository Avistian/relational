"""Read-only checks of the lesson manifest against authored repository assets."""
import json
from pathlib import Path

# CURRICULUM.md marks 179 as planned, not yet available. Piotr approved
# checking authored lessons 1–200 only; planned lessons 201–240 are out of scope.
AUTHORED_IDS = set(range(1, 201)) - {179}


def validate(root, manifest_path):
    """Return actionable integrity errors; never modify the manifest or assets.

    manifest_path may be a disposable copy; references always resolve from root.
    """
    root = Path(root)
    lessons = json.loads(Path(manifest_path).read_text())['lessons']
    errors = []
    seen_ids = set()
    listed_files = set()
    numeric_ids = set()
    for entry in lessons:
        identity = entry['id']
        if identity in seen_ids:
            errors.append(f'duplicate id: {identity}')
        seen_ids.add(identity)
        if type(identity) is int:
            numeric_ids.add(identity)
        lesson_file = f"{entry['slug']}.html"
        listed_files.add(lesson_file)
        if not (root / 'lessons' / lesson_file).is_file():
            errors.append(f'missing lesson file: lessons/{lesson_file}')
        lab_path = entry.get('labPath')
        if lab_path and not (root / lab_path).is_file():
            errors.append(f'missing lab file: {lab_path} (lesson {identity})')
    for path in sorted((root / 'lessons').glob('*.html')):
        if path.name not in listed_files:
            errors.append(f'unlisted lesson: {path.name}')
    for identity in sorted(AUTHORED_IDS - numeric_ids):
        errors.append(f'authored gap: {identity}')
    return errors
