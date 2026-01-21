from setuptools import setup, find_packages

setup(
    name='bev-perception',
    version='0.1.0',
    author='Meet Jain',
    description='Multi-View BEV Perception for Autonomous Driving',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0,<2.0',
        'opencv-python>=4.8.0',
        'nuscenes-devkit>=1.1.11',
    ],
)
