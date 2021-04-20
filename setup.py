import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mctorch-lib",
    version="0.1.0",
    author="McTorch Team",
    author_email="mayankmeghwanshi@gmail.com",
    description="McTorch, a manifold optimization library for deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mctorch/mctorch_lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["torch", "numpy"]
)