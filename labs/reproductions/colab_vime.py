"""Install a separate historical interpreter on x86_64 Colab, without replacing its kernel."""
import os
import platform
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def historical_python(root):
    root=Path(root).resolve();prefix=root/'vime-python37'
    python=prefix/'bin/python'
    if python.exists():return python
    if platform.machine()!='x86_64':
        raise RuntimeError('TF 1.15 requires x86_64. On ARM use the supplied Dockerfile with --platform linux/amd64.')
    archive=root/'micromamba.tar.bz2'
    urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/linux-64/1.5.8',archive)
    with tarfile.open(archive) as tar:
        member=tar.getmember('bin/micromamba')
        dest=root/'micromamba';dest.write_bytes(tar.extractfile(member).read());dest.chmod(0o755)
    env=dict(os.environ,MAMBA_ROOT_PREFIX=str(root/'mamba-root'))
    subprocess.run([str(dest),'create','-y','-p',str(prefix),'-c','conda-forge','python=3.7.12','pip=23.1.2'],env=env,check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(root/'requirements-vime.txt')],check=True)
    return python

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    python=historical_python(root)
    output=root/'vime-release-results.json'
    env=dict(os.environ,TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
    subprocess.run([str(python),str(root/'vime_release.py'),'--output',str(output)],cwd=root/'sources/vime',env=env,check=True)
