#! /usr/bin/env python

import os
import sys
import subprocess

from pathlib import Path
from sysconfig import get_paths
from distutils.extension import Extension
from distutils.command.build_ext import build_ext


class CMakeExtension(Extension):

    def __init__(self, name, source_dir: Path):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = source_dir.resolve()


class CMakeBuild(build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        cwd = Path().absolute()
        src_dir = ext.sourcedir

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DPYTHON_INCLUDE_DIR=' + str(get_paths()['include']),
            '-DPYTHON_EXECUTABLE=' + str(sys.executable),
            '-DCMAKE_INSTALL_PREFIX=' + str(src_dir / "zctc"),
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j' + str(os.cpu_count())
        ]

        subprocess.run(
            ["cmake", str(ext.sourcedir), *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install", *build_args], cwd=build_temp, check=True
        )

def build(setup_kwargs):

    setup_kwargs.update(
        {"ext_modules": [CMakeExtension("_zctc", Path(__file__).parent)], "cmdclass": {"build_ext": CMakeBuild}}
    )
      