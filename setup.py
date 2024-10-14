from setuptools import setup, find_packages
import os
import subprocess


def get_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_version():
    with open("VERSION", "r") as f:
        return f.read().strip()


setup(
    name="faceswap",
    version=get_version(),
    author="Omela Health",
    author_email="info@omelahealth.com",
    description="A face swapping tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/omelahealth/faceswap-demo",
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    py_modules=["faceswap"],
)
