Name:           omni-proxy
Version:        1.0
Release:        1%{?dist}
Summary:        Nginx custom global proxy modules

%{!?ngx_version:%global ngx_version 1.28.0}
%{!?libmsgpack_c_version:%global libmsgpack_c_version 6.1.0}

License:        MIT
Source0:        nginx-%{ngx_version}.tar.gz
Source1:        omni_proxy.tar.gz

BuildRequires:  gcc make zlib-devel pcre-devel openssl-devel zeromq zeromq-devel boost-devel

Conflicts:      nginx

%description
Custom global proxy modules for Nginx, built for PD Disaggregation.

%prep
%setup -q -c -T -a 0
%setup -q -c -n %{name}-%{version} -D -T -a 1

%build
cd %{_builddir}/%{name}-%{version}/nginx-%{ngx_version}
CFLAGS="-O0 -g" ./configure --sbin-path=/usr/sbin/ \
    --add-dynamic-module=../omni_proxy/modules \
    --add-dynamic-module=../omni_proxy/modules/ngx_http_set_request_id_module \
    --without-http_gzip_module
make -j 4
make modules

# Rust tokenizer accelerator (tokenizers_chunk_ext) -- optional, non-fatal, mirrors
# omni_proxy/build.sh. Skipped (the proxy falls back to AutoTokenizer tokenization)
# when the embedded python3.11 or a Rust toolchain is absent on the build host.
EMBED_PY=/usr/local/bin/python3.11
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env" || true
if [ -x "$EMBED_PY" ] && command -v cargo >/dev/null 2>&1; then
    ( cd ../omni_proxy/tools/tokenizers_chunk_ext \
      && "$EMBED_PY" -m pip install --upgrade maturin patchelf \
      && "$EMBED_PY" -m maturin build --release -i "$EMBED_PY" -o dist ) \
      || echo "WARNING: chunk accelerator not built; proxy falls back to AutoTokenizer"
fi

%install
mkdir -p %{buildroot}/usr/sbin
cp nginx-%{ngx_version}/objs/nginx %{buildroot}/usr/sbin/nginx

mkdir -p %{buildroot}/usr/local/nginx/
cp -a nginx-%{ngx_version}/conf  %{buildroot}/usr/local/nginx/
cp -a nginx-%{ngx_version}/html  %{buildroot}/usr/local/nginx/
cp -a nginx-%{ngx_version}/objs/nginx %{buildroot}/usr/local/nginx/nginx
mkdir -p %{buildroot}/usr/local/nginx/logs/
mkdir -p %{buildroot}/usr/local/nginx/modules/
cp nginx-%{ngx_version}/objs/*.so %{buildroot}/usr/local/nginx/modules/

mkdir -p %{buildroot}/usr/lib64
cp /usr/lib64/libmsgpack-c.so.2.0.0 %{buildroot}/usr/lib64
ln -sf libmsgpack-c.so.2.0.0 %{buildroot}/usr/lib64/libmsgpack-c.so.2
ln -sf libmsgpack-c.so.2.0.0 %{buildroot}/usr/lib64/libmsgpack-c.so

mkdir -p %{buildroot}/usr/local/lib
cp /usr/local/lib/libpython3.11.so.1.0 %{buildroot}/usr/local/lib
ln -sf libpython3.11.so.1.0 %{buildroot}/usr/local/lib/libpython3.11.so
mkdir -p %{buildroot}/usr/local/lib/python3.11/site-packages
cp %{_builddir}/%{name}-%{version}/omni_proxy/modules/omni_tokenizer.py %{buildroot}/usr/local/lib/python3.11/site-packages/omni_tokenizer.py

# Rust tokenizer accelerator wheel if it was built in %build -- installs into
# site-packages alongside omni_tokenizer.py (packaged by the site-packages glob below)
CHUNK_WHL=$(ls %{_builddir}/%{name}-%{version}/omni_proxy/tools/tokenizers_chunk_ext/dist/tokenizers_chunk_ext-*.whl 2>/dev/null | head -1)
if [ -n "$CHUNK_WHL" ]; then
    /usr/local/bin/python3.11 -m pip install --no-deps --no-compile \
        --target %{buildroot}/usr/local/lib/python3.11/site-packages "$CHUNK_WHL"
fi

%files
/usr/sbin/nginx
/usr/local/nginx/nginx
/usr/local/nginx/conf/*
/usr/local/nginx/html/*
%dir /usr/local/nginx/logs
/usr/local/nginx/modules/*.so
/usr/lib64/libmsgpack-c.so.2.0.0
/usr/lib64/libmsgpack-c.so.2
/usr/lib64/libmsgpack-c.so
/usr/local/lib/libpython3.11.so.1.0
/usr/local/lib/libpython3.11.so
/usr/local/lib/python3.11/site-packages/*

%changelog
* Thu Jun 12 2026 Omni - 1.0-2
  tokenizers_chunk_ext Rust accelerator, matching omni_proxy/build.sh.
* Mon Oct 27 2025 Huawei - 1.0-1
- Initial build
