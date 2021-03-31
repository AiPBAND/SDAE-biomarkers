from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>2.2',
                     'pandas',
                     'argparse',
                     'sklearn']

setup(
    name='unsupervised',
    author=[
        'Jeremy Georges-Filteau',
        'Birbal Prasad'
    ],
    author_email=['jeremy.geo@gmail.com', 'birbalprasad22@gmail.com'],
    url='http://www.github.com/u/aipband/',
    version='0.1',
    install_required='REQUIRED_PACKAGES',
    packages=find_packages(),
    include_package_data=True,
    description='SDAE'
)
