import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["numpy>=1.15"]
    
setuptools.setup(
    name="noisy-state-lib-maxastyler", # Replace with your own username
    version="0.0.1",
    author="Max Tyler",
    author_email="maxastyler@gmail.com",
    description="Noisy state functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HWQuantum/noisy-state-lib",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
