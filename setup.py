from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="BinaryVectorDB",
    version="0.0.2",
    author="Nils Reimers",
    author_email="nils@cohere.com",
    description="Efficient vector DB using binary & int8 embeddings",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/cohere-ai/BinaryVectorDB",
    download_url="https://github.com/cohere-ai/BinaryVectorDB/archive/v0.0.1.zip",
    packages=find_packages(),
    install_requires=[
        'faiss-cpu==1.8.0',
        'numpy',
        'rocksdict',
        'cohere==4.57',
        'packaging'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Vector Database"
)