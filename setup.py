#! /usr/bin/env python

import os
import re
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from sysconfig import get_paths
from tempfile import TemporaryDirectory
from typing import List

import requests
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, source_dir: Path, libraries: List[str] = []):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[], libraries=libraries)
        self.sourcedir = source_dir.resolve()


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        pattern = re.compile(r".*lib.python\d.\d+$")
        for path in sys.path:
            if pattern.match(path):
                src_dir = path

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        fst_v = "openfst-1.8.3"
        fst_url = f"https://www.openfst.org/twiki/pub/FST/FstDownload/{fst_v}.tar.gz"

        with TemporaryDirectory() as tmp_dir:
            res = requests.get(fst_url)
            with tarfile.open(fileobj=BytesIO(res.content)) as tar:
                tar.extractall(tmp_dir)
            del res

            # example of cmake args
            config = "Debug" if self.debug else "Release"
            cmake_args = [
                "-DLIBRARY_OUTPUT_DIRECTORY=" + src_dir,
                "-DCMAKE_BUILD_TYPE=" + config,
                "-DPYTHON_INCLUDE_DIR=" + str(get_paths()["include"]),
                "-DPYTHON_EXECUTABLE=" + str(sys.executable),
                "-DCMAKE_INSTALL_PREFIX=" + str(build_temp),
                "-DFST_DIR=" + str(Path(tmp_dir) / fst_v),
            ]

            # example of build args
            build_args = ["--config", config, "--", "-j" + str(os.cpu_count())]

            subprocess.run(
                ["cmake", str(ext.sourcedir), *cmake_args], cwd=build_temp, check=True
            )
            subprocess.run(
                ["cmake", "--build", ".", "--target", "install", *build_args],
                cwd=build_temp,
                check=True,
            )

            ext.library_dirs = [src_dir]

setup(
    name="zctc",
    version="0.1",
    description="A fast and efficient CTC beam decoder with C++ backend.",
    author="Durgai Vel Selvan M",
    author_email="durgaivel0309@gmail.com",
    packages=find_packages(str(Path(__file__).parent)),
    # package_data={},
    include_package_data=True,
    # libraries=[],
    ext_modules=[
        CMakeExtension(
            "_zctc",
            Path(__file__).parent,
            [
                "_zctc",
                "kenlm_filter",
                "kenlm_builder",
                "kenlm_util",
                "kenlm"
            ]
        )
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
)
