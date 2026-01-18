#!/usr/bin/env python3
"""
Setup script for dat_ml - Discrete Alignment Theory ML Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dat-ml",
    version="0.1.0",
    author="Bryan Solomon",
    description="Golden ratio spectral filtering for time series prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SolomonB14D3/dat-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "full": [
            "matplotlib>=3.3.0",
            "scipy>=1.5.0",
            "pandas>=1.1.0",
        ],
    },
    keywords="machine-learning, time-series, spectral-filtering, golden-ratio, weather-prediction",
)
