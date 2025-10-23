import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spectrainterface",
    version="0.2.4",
    author="Gabriel Rezende da Ascencao, Jefferson Barros Vieira, Sergio Augusto Lordano Luiz",
    author_email="gabriel.ascencao@lnls.br, jefferson.vieira@lnls.br, sergio.lordano@lnls.br",
    maintainer="Brazilian Synchrotron Light Laboratory (LNLS) - Insertion Devices and Photon Diagnostics group (IDS)",
    maintainer_email="ids@lnls.br",
    description="Python interface for SPECTRA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lnls-ids/spectrainterface",
    project_urls={
        "Organization": "https://lnls.cnpem.br/grupos/accelerator-division/",
        "Source Code": "https://github.com/lnls-ids/spectrainterface",
        "Documentation": "https://github.com/lnls-ids/spectrainterface#readme",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'mathphys>=2.9', 'scipy>=1.10'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Physics :: Scientific/Engineering",
    ],
    python_requires='>=3.8.0, <3.9.0',

)
