###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################

from setuptools import setup

v = '1.0.0'
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    README = f.read()

setup(
    name="FFEAT",
    version=v,
    description="Framework for Evolutionary Algorithms in Torch",
    url="https://github.com/PatrikValkovic/FEAT",
    download_url='https://github.com/PatrikValkovic/FEAT/archive/v' + v + '.tar.gz',
    long_description=README,
    long_description_content_type="text/markdown",
    author="Patrik Valkovic",
    license="GNU LGPLv3",
    packages=[
        "ffeat",
        "ffeat/flow"
    ],
    install_requires=[
        "numpy",
        "torch",
    ],
)