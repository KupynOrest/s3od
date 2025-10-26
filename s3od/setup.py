from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="s3od",
    version="0.1.0",
    author="S3OD Team",
    description="Simple and efficient background removal using S3OD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/organization/s3od",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "huggingface-hub>=0.16.0",
    ],
)

