#!/bin/bash

set -e

BUILD_ROOT="$(dirname "$(realpath "$0")")"
cd $BUILD_ROOT

if [ $(id -u) -ne 0 ]; then
    echo "Require root permission,try sudo ./build.sh"
    exit 1
fi


if rpm -q etcd >/dev/null 2>&1; then
    echo "etcd installed."
else
    echo "sslverify=0" >> /etc/yum.conf
    yum install -y etcd
fi


if pip show lmcache >/dev/null 2>&1; then
    echo "lmcahce installed."
else
    git config --global http.sslVerify false
    git clone https://github.com/LMCache/LMCache.git

    PATCH_ROOT=${1:-$BUILD_ROOT/../patch/}
    LMCACHE_PATH=${2:-$BUILD_ROOT/LMCache}
    MOONCAKE_PATH=${3:$BUILD_ROOT/Mooncake}

    # install lmcache and ascend_lmcache
    cd ${LMCACHE_PATH}
    git reset --hard
    git clean -fd
    git checkout v0.3.2
    git apply --whitespace=nowarn $PATCH_ROOT/npu_adaptor.patch

    export NO_CUDA_EXT=1
    pip install -e .

    cd $BUILD_ROOT/..
    python setup.py bdist_wheel
    cd $BUILD_ROOT
    pip install ../dist/ascend_lmcache-*.whl

    # install mooncake
    bash $BUILD_ROOT/install_mooncake.sh
fi

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib' >> ~/.bashrc