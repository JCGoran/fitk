from setuptools import setup

setup(
    name="fitk",
    version="1.0.0",
    description="The Fisher Information ToolKit",
    author="Goran Jelic-Cizmek",
    author_email="goran.jelic-cizmek@unige.ch",
    install_requires=[
        "matplotlib>=3.3",
        "numpy>=1.16.0",
        "scipy>=1.2.0",
    ],
    packages=["fitk"],
)
