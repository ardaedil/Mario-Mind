from setuptools import setup, find_packages

setup(
    name='mariomind',
    version='0.1.0',
    description='Experimental RL lab for Mario World 1-1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[],
)
