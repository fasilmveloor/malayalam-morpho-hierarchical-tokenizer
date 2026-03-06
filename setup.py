"""
Malayalam Morpho-Hierarchical Tokenizer

A novel morphologically-aware tokenizer for Malayalam combining
Finite State Transducers with Phoneme-Aware Bi-LSTM.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="malayalam-morpho-tokenizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A morphologically-aware tokenizer for Malayalam",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/malayalam-tokenizer",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/malayalam-tokenizer/issues",
        "Documentation": "https://github.com/yourusername/malayalam-tokenizer#readme",
        "Source Code": "https://github.com/yourusername/malayalam-tokenizer",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "mlmorph": ["mlmorph>=1.0.0"],
        "transformers": ["transformers>=4.30.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    keywords=[
        "nlp",
        "tokenization",
        "malayalam",
        "morphology",
        "dravidian",
        "linguistics",
        "neural-network",
        "huggingface",
    ],
)
