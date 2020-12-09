from setuptools import setup, find_packages
from glob import glob

scripts = glob('bin/*')
scripts = [f for f in scripts if '~' not in f and 'old' not in f]

setup(
    name="compare_y1_y3",
    version="0.1.0",
    packages=find_packages(),
    scripts=scripts,
    author='Erin Sheldon',
    author_email='erin.sheldon@gmail.com',
    url='https://github.com/esheldon/compare-y1-y3',
)
