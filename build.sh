#! /bin/bash

if [$# -gt 2]; then
    echo "Usage: $0 [Release | Debug(default)]"
    exit 1
fi

PWD=$(pwd)
cd "$(realpath "$(dirname "$0")")" || exit
mkdir build
WORKER=$(($(($(nproc) / 2)) > 15 ? 15 : $(($(nproc) / 2))))

if [$# -eq 1]; then
    BUILD_TYPE="Debug"
else
    BUILD_TYPE=$1
fi

cmake -B ./build -DPYTHON_EXECUTABLE:PATH="$(command -v python)" -DCMAKE_INSTALL_PREFIX:PATH="$(realpath ./zctc)" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DPYTHON_INCLUDE_DIR:PATH="$(python -c "from sysconfig import get_paths; print(get_paths()['include'])")" .

cd build || exit
make "-j$WORKER" && make install

cd .. || exit 0
rm -rf ./build

cd $PWD
