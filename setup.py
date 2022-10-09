from setuptools import setup, find_packages

setup(
    name='activedet',
    version='0.3.0',
    description="Active Learning for Object Detection",
    packages=find_packages(include=['activedet',]),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)