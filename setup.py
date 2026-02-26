"""
Setup script for RIEnet package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from version.py file
def read_version():
    version_file = os.path.join(this_directory, 'rienet', 'version.py')
    with open(version_file, 'r') as f:
        version_content = f.read()
    version_line = [line for line in version_content.split('\n') if line.startswith('__version__')]
    if version_line:
        return version_line[0].split('=')[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string.")

setup(
    name="rienet",
    version=read_version(),
    author="Christian Bongiorno",
    author_email="christian.bongiorno@centralesupelec.fr",
    description="A Compact Recurrent-Invariant Eigenvalue Network for Portfolio Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bongiornoc/RIEnet",
    project_urls={
        "Bug Tracker": "https://github.com/bongiornoc/RIEnet/issues",
        "Documentation": "https://github.com/bongiornoc/RIEnet",
        "Source Code": "https://github.com/bongiornoc/RIEnet",
    },
    packages=find_packages(include=["rienet", "rienet.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "keras>=2.10.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "jupyter>=1.0.0",
        ],
    },
    keywords=[
        "portfolio optimization",
        "neural networks", 
        "finance",
        "machine learning",
        "tensorflow",
        "eigenvalue decomposition",
        "recurrent neural networks",
        "covariance estimation",
    ],
    license="MIT",
    zip_safe=False,
    include_package_data=True,
)
