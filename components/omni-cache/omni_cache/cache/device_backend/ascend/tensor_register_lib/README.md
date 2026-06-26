# C++ tensor registration library for Ascend NPU

This directory contains the C++ extension code for tensor registration.

## Files

- `tensor_register.cpp`: Main C++ library for tensor registration using ACL
- `zero_copy_npu.cpp`: PyTorch C++ extension for zero-copy NPU tensor creation
- `setup.py`: Python setuptools configuration for building the extension
- `build.sh`: Build script for compiling the libraries

## Building

Run the build script:

```bash
bash build.sh
```

Or manually with setuptools:

```bash
python setup.py build_ext --inplace
```

## Requirements

- CANN toolkit (Ascend)
- PyTorch
- torch_npu
