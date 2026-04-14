from setuptools import setup

setup(
    name="modelcascade",
    version="0.1.0",
    py_modules=["modelcascade"],
    python_requires=">=3.10",
    description="Route local. Escalate smart. Never overspend.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Wayne Colt",
    author_email="wayne@wayneia.com",
    license="MIT",
    url="https://github.com/wayneColt/modelcascade",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
