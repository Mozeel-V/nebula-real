from setuptools import setup, find_packages

setup(
    name="nebula_mini",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
