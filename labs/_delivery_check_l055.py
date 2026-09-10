"""Compatibility entry point for the repaired current L055 package."""
import runpy
from pathlib import Path

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).with_name('_delivery_check_l055_v2.py')),run_name='__main__')
