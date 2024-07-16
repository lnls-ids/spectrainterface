import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spectrainterface",
    version="0.2.2",
    author="Gabriel Rezende da Ascencao",
    author_email="gabriel.ascencao@lnls.br",
    description="Python interface for SPECTRA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lnls-ids/spectrainterface",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'mathphys>=2.9'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.0, <3.9.0',

)
