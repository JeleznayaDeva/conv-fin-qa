import os

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


def get_version() -> str:
    # Read from env variable
    return os.environ.get("APP_VERSION", "0.1.dev0")


setup(
    name="conv-fin-qa",
    version=get_version(),
    description="Prototype for financial question answering using Mistral and OpenAI's GPT-3.5",
    long_description=readme,
    author="Firuza Mamedova",
    author_email="firuza.mamedova@gmail.com",
    license="",
    url="",
    scripts=[],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    namespace_packages=[],
    py_modules=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={},
    data_files=[],
    install_requires=install_requires,
    dependency_links=[],
    zip_safe=True,
    keywords="",
    python_requires=">=3.8",
    obsoletes=[],
)
