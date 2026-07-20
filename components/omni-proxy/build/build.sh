#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -e

PKG_VERSION=1.0
PKG_RELEASE=1
NGINX_VERSION=1.28.0
MSGPACK_VERSION=6.1.0

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
RPMBUILD=$WORKDIR/rpmbuild
SPEC_FILE="omni-proxy.spec"

cd $WORKDIR

mkdir -p $WORKDIR/SOURCES

rm -rf $RPMBUILD
mkdir -p $RPMBUILD/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

if [ ! -f SOURCES/nginx-${NGINX_VERSION}.tar.gz ]; then
    echo "nginx-${NGINX_VERSION}.tar.gz not found, downloading..."
    wget --no-check-certificate https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz -O SOURCES/nginx-${NGINX_VERSION}.tar.gz
fi

# Always regenerate the source tarball from the current tree (caching it would
# build the RPM from stale source). Heavy build artifacts are excluded to keep it
# lean; the spec rebuilds the modules and the Rust wheel from source.
echo "creating omni_proxy.tar.gz from current source..."
rm -f SOURCES/omni_proxy.tar.gz
tar --exclude=build --exclude=target --exclude=__pycache__ --exclude='*.pyc' \
    -czf SOURCES/omni_proxy.tar.gz -C ../ omni_proxy

cp SOURCES/omni_proxy.tar.gz $RPMBUILD/SOURCES/
cp SOURCES/nginx-${NGINX_VERSION}.tar.gz $RPMBUILD/SOURCES/
cp SPECS/${SPEC_FILE} $RPMBUILD/SPECS/

echo "start to build rpm in $RPMBUILD"
rpmbuild --define "ngx_version ${NGINX_VERSION}" --define "libmsgpack_c_version ${MSGPACK_VERSION}" --define "_topdir $RPMBUILD" --define "debug_package %{nil}" -ba $RPMBUILD/SPECS/${SPEC_FILE}

ARCH=$(uname -m)
echo "RPM Packages has been built in $RPMBUILD/RPMS/$ARCH/"
ls -lh $RPMBUILD/RPMS/$ARCH/

yum remove -y global-proxy
rpm -Uvh --replacepkgs $RPMBUILD/RPMS/$ARCH/*.rpm
