#!/bin/bash
set -ueo pipefail

echo "start build opencv-cc-arm64"
OPENCV_VERSION="4.9.0"
PYTHON3_REALPATH=$(realpath /usr/bin/python3)
PYTHON3_BASENAME=$(basename "${PYTHON3_REALPATH}")
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
        cmake -S opencv-"${OPENCV_VERSION}" \
              -B build4-full_arm64 \
              -D CMAKE_BUILD_TYPE=RELEASE \
              -D CMAKE_TOOLCHAIN_FILE=/tmp/work/opencv-"${OPENCV_VERSION}"/platforms/linux/aarch64-gnu.toolchain.cmake \
              -D OPENCV_EXTRA_MODULES_PATH=/tmp/work/opencv_contrib-"${OPENCV_VERSION}"/modules \
              -D INSTALL_C_EXAMPLES=OFF\
              -D ENABLE_NEON=ON \
              -D XNNPACK_ENABLE_ARM_BF16=OFF \
              -D OPENCV_ENABLE_NONFREE=ON \
              -D INSTALL_PYTHON_EXAMPLES=OFF \
              -D BUILD_EXAMPLES=OFF \
              -D PYTHON3_NUMPY_INCLUDE_DIRS="/usr/local/lib/${PYTHON3_BASENAME}/dist-packages/numpy/core/include/" \
              -D PYTHON3_INCLUDE_PATH="/usr/include/${PYTHON3_BASENAME};/usr/include/" \
              -D PYTHON3_LIBRARIES=$(find /usr/lib/aarch64-linux-gnu/ -name libpython*.so) \
              -D PYTHON3_EXECUTABLE="/usr/bin/${PYTHON3_BASENAME}" \
              -D PYTHON3_CVPY_SUFFIX=".so" \
              -G Ninja

cmake --build build4-full_arm64
cmake --install build4-full_arm64

tar czvf opencv_arm64.tgz -C build4-full_arm64/install .