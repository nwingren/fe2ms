from setuptools import setup

setup(name='fe2ms',
      version=0.1,
      description='A finite element-boundary integral code for electromagnetic scattering',
      url='http://github.com/nwingren/fe2ms',
      author='Niklas Wingren',
      author_email='niklas.wingren@eit.lth.se',
      license='GPLv3+',
      packages=['fe2ms', 'fe2ms.bindings'],
      zip_safe=False,
      include_package_data=True)
