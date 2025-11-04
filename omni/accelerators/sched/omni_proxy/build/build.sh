#!/bin/bash
set -e

PKG_VERSION=1.0
PKG_RELEASE=1
NGINX_VERSION=1.28.0
MSGPACK_VERSION=6.1.0

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
RPMBUILD=$WORKDIR/rpmbuild
SPEC_FILE="omni-proxy.spec"

mkdir -p $WORKDIR/SOURCES

rm -rf $RPMBUILD
mkdir -p $RPMBUILD/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

if [ ! -f SOURCES/nginx-${NGINX_VERSION}.tar.gz ]; then
    echo "nginx-${NGINX_VERSION}.tar.gz not found, downloading..."
    wget --no-check-certificate https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz -O SOURCES/nginx-${NGINX_VERSION}.tar.gz
fi

if [ ! -f SOURCES/omni_proxy.tar.gz ]; then
    echo "omni_proxy.tar.gz not found, creating..."
    tar --exclude=build -czf SOURCES/omni_proxy.tar.gz -C ../.. global_proxy omni_proxy
fi

cp SOURCES/omni_proxy.tar.gz $RPMBUILD/SOURCES/
cp SOURCES/nginx-${NGINX_VERSION}.tar.gz $RPMBUILD/SOURCES/
cp SPECS/${SPEC_FILE} $RPMBUILD/SPECS/

echo "start to build rpm in $RPMBUILD"
rpmbuild --define "ngx_version ${NGINX_VERSION}" --define "libmsgpack_c_version ${MSGPACK_VERSION}" --define "_topdir $RPMBUILD" --define "debug_package %{nil}" -ba $RPMBUILD/SPECS/${SPEC_FILE}

ARCH=$(uname -m)
echo "RPM Packages has been built in $RPMBUILD/RPMS/$ARCH/"
ls -lh $RPMBUILD/RPMS/$ARCH/

DIST_DIR="$(cd "$(dirname "$0")"/../../../../.. && pwd)/build/dist"
mkdir -p "$DIST_DIR"
cp $RPMBUILD/RPMS/$ARCH/*.rpm "$DIST_DIR"
echo "RPM packages copied to $DIST_DIR"
