from setuptools import setup

import os
here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='FruitAPI',
    version=1.0,
    url='',
    license='GNU License',
    author='Duy Nguyen',
    author_email='garlicdevs@gmail.com',
    description='A universal framework for deep reinforcement learning',
    long_description=long_description,
    packages=['fruit'],
    long_description_content_type="text/markdown",
    include_package_data=True,
    platforms='any',
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: Alpha',
        'Environment :: Machine Learning',
        'Intended Audience :: Developers/Researchers',
        'License :: GNU License',
        'Operating System :: OS Independent',
        ]
)
