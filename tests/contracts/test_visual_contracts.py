"""Expose the original script contracts through standard unittest discovery."""
import importlib
from pathlib import Path
import sys
import unittest


MODULES = (
    'check_course_visuals',
    'check_lesson_visuals',
    'check_visual_details',
    'check_lesson_openings',
)


def load_tests(loader, tests, pattern):
    scripts = str(Path(__file__).resolve().parents[2] / 'scripts')
    sys.path.insert(0, scripts)
    try:
        return unittest.TestSuite(
            loader.loadTestsFromModule(importlib.import_module(name))
            for name in MODULES
        )
    finally:
        sys.path.pop(0)
