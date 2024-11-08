from setuptools import setup, find_packages

# Read in the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="set_transformer",
    version="1.1.0",  # Update the version as needed
    author="Brendan Crowe",
    author_email="brendancrowe98@gmail.com",
    description="A PyTorch implementation of Set Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/set_transformer",  # Update with your repo URL
    packages=find_packages(),  # Automatically finds all packages under `set_transformer/`
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",  # Adjust as needed for compatibility
    install_requires=[
        "torch>=1.0",           # PyTorch requirement
        "numpy>=1.18.0",        # Add other dependencies as required
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",     # Development dependencies, e.g., for testing
            "flake8",
            "black",
        ],
    },
)