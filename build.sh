#! /bin/bash

if [ $# -gt 2 ]; then
    echo "Usage: $0 [Release | Debug(default)]"
    exit 1
fi

X="sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev"

PWD=$(pwd)
cd "$(realpath "$(dirname "$0")")" || exit
mkdir build
WORKER=$(($(($(nproc) / 2)) > 20 ? 20 : $(($(nproc) / 2))))

if [ $# -eq 0 ]; then
    BUILD_TYPE="Debug"
else
    BUILD_TYPE=$1
fi

fst_v="openfst-1.8.3"
fst_url="https://www.openfst.org/twiki/pub/FST/FstDownload/$fst_v.tar.gz"

wget $fst_url
tar -xzf $fst_v.tar.gz

cmake -B ./build -DPYTHON_EXECUTABLE:PATH="$(command -v python)" -DCMAKE_INSTALL_PREFIX:PATH="$(realpath ./build)" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DPYTHON_INCLUDE_DIR:PATH="$(python -c "from sysconfig import get_paths; print(get_paths()['include'])")" \
    -DFST_DIR="$(realpath ./$fst_v)" -DLIBRARY_OUTPUT_DIRECTORY:PATH="$(realpath ./zctc)" .

cd build || exit
make "-j$WORKER" && make install

cd .. || exit 0
rm -rf ./build ./$fst_v ./$fst_v.tar.gz

cd $PWD
