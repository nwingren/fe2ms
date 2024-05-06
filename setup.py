import subprocess
from setuptools import setup

subprocess.run('mamba env update', shell=True, check=False)
subprocess.run('python -m pip install https://github.com/Excalibur-SLE/AdaptOctree/archive/27e49e142463eb0114fb37aad013f51680aa0c0f.tar.gz -v --no-deps', shell=True, check=False)
subprocess.run('python -m pip install https://github.com/nwingren/demcem4py/archive/refs/tags/v1.1.0.tar.gz -v --no-deps', shell=True, check=False)

setup(name='fe2ms',
      version='0.1.0',
      description='A finite element-boundary integral code for electromagnetic scattering',
      url='http://github.com/nwingren/fe2ms',
      author='Niklas Wingren',
      packages=['fe2ms'],
      classifiers=[
          "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
      ]
)
