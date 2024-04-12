#! /bin/bash

PWD=$(pwd)
cd "$(realpath "$(dirname "$0")")" || exit
mkdir build
WORKER=$(($(($(nproc) / 2)) > 15 ? 15 : $(($(nproc) / 2))))

cmake -B ./build -DPYTHON_EXECUTABLE:PATH="$(command -v python)" -DCMAKE_INSTALL_PREFIX:PATH="$(realpath ./zctc)" \
    -DCMAKE_BUILD_TYPE=Release -DPYTHON_INCLUDE_DIR:PATH="$(python -c "from sysconfig import get_paths; print(get_paths()['include'])")" .

cd build || exit
make "-j$WORKER" && make install

cd .. || exit 0
rm -rf ./build

cd $PWD
