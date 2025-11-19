# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# KMFODA
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import codecs
import os
import re
import subprocess
from io import open
from os import path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


def custom_command():
    import pip

    pip.main(
        [
            "install",
            "git+https://github.com/learning-at-home/hivemind.git@3a4cc15e29ce51b20c5d415a4c579abbae435718",
        ]
    )
    pip.main(["install", "bittensor==9.12.2"])
    pip.main(["install", "py-multihash==2.0.1"])

    # Install Go and HFDownloader
    try:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "golang"], check=True)
        subprocess.run(
            ["go", "install", "github.com/lxe/hfdownloader@latest"], check=True
        )

        # Add Go bin to PATH in venv activate script
        if "VIRTUAL_ENV" in os.environ:
            activate_script = os.path.join(os.environ["VIRTUAL_ENV"], "bin", "activate")
            if os.path.exists(activate_script):
                with open(activate_script, "a") as f:
                    f.write('\nexport PATH="$HOME/go/bin:$PATH"\n')
                # Also update current session's PATH
                go_bin_path = os.path.expanduser("~/go/bin")
                if go_bin_path not in os.environ["PATH"]:
                    os.environ["PATH"] = f"{go_bin_path}:{os.environ['PATH']}"

    except Exception as e:
        raise RuntimeError(f"Failed to install Go and HFDownloader: {str(e)}")


class CustomInstallCommand(install):
    def run(self):
        custom_command()
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        custom_command()
        develop.run(self)


class CustomEggInfoCommand(egg_info):
    def run(self):
        custom_command()
        egg_info.run(self)


with open("requirements.txt") as requirements_file:
    requirements = list(map(str, parse_requirements(requirements_file)))

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(
    os.path.join(here, "distributed_training/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="distributed_training",
    version=version_string,
    description="distributed_training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dstrbtd/DistributedTraining",
    author="KMFODA",
    packages=find_packages(),
    include_package_data=True,
    author_email="",
    license="MIT",
    python_requires=">=3.8",
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
    setup_requires=["pip"],
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
