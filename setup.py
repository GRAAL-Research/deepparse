import os
import subprocess

from setuptools import setup, find_packages

current_file_path = os.path.abspath(os.path.dirname(__file__))


def get_readme():
    readme_file_path = os.path.join(current_file_path, "README.md")
    with open(readme_file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_version():
    version_file_path = os.path.join(current_file_path, "version.txt")
    with open(version_file_path, "r", encoding="utf-8") as f:
        version = f.read().strip()

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:  # pylint: disable=broad-except
        sha = "Unknown"

    if os.getenv("DEEPPARSE_RELEASE_BUILD") != "1":
        version += ".dev1"
        if sha != "Unknown":
            version += "+" + sha[:7]
    return version


def write_version_python_file(version):
    version_python_file = os.path.join(current_file_path, "deepparse", "version.py")
    with open(version_python_file, "w", encoding="utf-8") as f:
        f.write(f"__version__ = {repr(version)}\n")


def main():
    readme = get_readme()

    version = get_version()
    print("Building version", version)
    write_version_python_file(version)

    packages = find_packages()
    setup(
        name="deepparse",
        version=version,
        author="Marouane Yassine, David Beauchemin",
        author_email="marouane.yassine.1@ulaval.ca, david.beauchemin.5@ulaval.ca",
        url="https://deepparse.org/",
        download_url="https://github.com/GRAAL-Research/deepparse/archive/v" + version + ".zip",
        license="LGPLv3",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        packages=packages,
        install_requires=[
            "numpy",
            "torch",
            "bpemb",
            "gensim>=4.0.0",
            "requests",
            "fasttext-wheel",
            "pymagnitude-light",
            "poutyne",
            "pandas==2.0.3",
            "urllib3",
            "cloudpathlib[s3, gs, azure]",
        ],
        python_requires=">=3.8",
        description="A library for parsing multinational street addresses using deep learning.",
        long_description=readme,
        long_description_content_type="text/markdown",
        extras_require={
            "colorama": "colorama>=0.4.3",
            "app": ["fastapi[all]==0.99.1", "uvicorn==0.22.0", "sentry-sdk[fastapi]==1.28.1", "python-decouple==3.8"],
            "all": [
                "colorama>=0.4.3",  # colorama
                "fastapi[all]==0.99.1",  # app requirements
                "uvicorn==0.22.0",
                "sentry-sdk[fastapi]>=1.28.1",
                "python-decouple==3.8",
                "black==23.3.0",  # code formatting requirements
                "pylint==2.16.2",
                "pylint-django[with_django]==2.5.3",
                "pre-commit==3.3.3",
                "pycountry==22.3.5",  # tests requirements
                "pytest==7.4.0",
                "pytest-asyncio==0.21.1",
                "pytest_cov==4.1.0",
                "pytest-env==0.8.2",
                "pytest-mock==3.11.1",
                "pytest-xdist[psutil]==3.3.1",
                "tensorboard==2.13.0",
                "sphinx==6.2.1",  # documentation requirements
                "sphinx_rtd_theme==1.2.2",
            ],
        },
    )


if __name__ == "__main__":
    main()
