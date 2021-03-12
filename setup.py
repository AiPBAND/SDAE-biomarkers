from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>2.2',
                     'pandas',
                     'argparse',
                     'sklearn']

setup(
    name='unsupervised',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='My training application.'
)
