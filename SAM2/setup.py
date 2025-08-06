# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

# Package metadata
NAME = "SAM-2"
VERSION = "1.0"
DESCRIPTION = "SAM 2: Segment Anything in Images and Videos"
URL = "https://github.com/facebookresearch/sam2"
AUTHOR = "Meta AI"
AUTHOR_EMAIL = "segment-anything@meta.com"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "jittor>=1.3.9",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
]

EXTRA_PACKAGES = {
    "notebooks": [
        "matplotlib>=3.9.1",
        "jupyter>=1.0.0",
        "opencv-python>=4.7.0",
        "eva-decord>=0.6.1",
    ],
    "interactive-demo": [
        "Flask>=3.0.3",
        "Flask-Cors>=5.0.0",
        "av>=13.0.0",
        "dataclasses-json>=0.6.7",
        "eva-decord>=0.6.1",
        "gunicorn>=23.0.0",
        "imagesize>=1.4.1",
        "pycocotools>=2.0.8",
        "strawberry-graphql>=0.243.0",
    ],
    "dev": [
        "black==24.2.0",
        "usort==1.0.2",
        "ufmt==2.0.0b2",
        "fvcore>=0.1.5.post20221221",
        "pandas>=2.2.2",
        "scikit-image>=0.24.0",
        "tensorboard>=2.17.0",
        "pycocotools>=2.0.8",
        "tensordict>=0.6.0",
        "opencv-python>=4.7.0",
        "submitit>=1.5.1",
    ],
}

# By default, we also build the SAM 2 CUDA extension.
# You may turn off CUDA build with `export SAM2_BUILD_CUDA=0`.
BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
# By default, we allow SAM 2 installation to proceed even with build errors.
# You may force stopping on errors with `export SAM2_BUILD_ALLOW_ERRORS=0`.
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

# Catch and skip errors during extension building and print a warning message
# (note that this message only shows up under verbose build mode
# "pip install -v -e ." or "python setup.py build_ext -v")
CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the SAM 2 CUDA extension due to the error above. "
    "You can still use SAM 2 and it's OK to ignore the error above, although some "
    "post-processing functionality may be limited (which doesn't affect the results in most cases; "
    "(see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).\n"
)


def compile_cuda_extension():
    if not BUILD_CUDA:
        return

    try:
        import jittor as jt
        from jittor import Function
        jt.flags.use_cuda = 1
        # 设置编译参数
        jt.flags.nvcc_flags = """
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        """
        # 创建编译目录
        # cuda_src_dir = os.path.join(os.path.dirname(__file__), "SAM2", "csrc")
        build_dir = os.path.join(os.path.dirname(__file__), "_C")
        os.makedirs(build_dir, exist_ok=True)
        # 编译 CUDA 代码
        print("Compiling CUDA extension for Jittor...")
        # 将 CUDA 代码转换为 Jittor 兼容格式
        # cuda_src = os.path.join(cuda_src_dir, "connected_components_jittor.cu")
        # jt.compiler.compile(cuda_src, output=build_dir,obj_dirname="cc2d_text")

        # 获取当前目录
        _C_DIR = os.path.join(os.path.dirname(__file__), "SAM2", "csrc")
        _CUDA_FILE = os.path.join(_C_DIR, "connected_components_jittor.cu")
        # 读取CUDA代码
        with open(_CUDA_FILE, 'r') as f:
            cuda_src = f.read()
        init_file = os.path.join(build_dir, "__init__.py")
        init_content = '''
        import os
        import jittor as jt
        
        # Get the directory containing the compiled extension
        _C_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Define the CUDA source
        # Compile the CUDA kernel using Jittor
        
        def get_connected_componnets(inputs):
            """
            Get connected components for binary masks
            
            Args:
                inputs: Jittor tensor of shape (N, 1, H, W) with dtype uint8
                
            Returns:
                labels: Jittor tensor of shape (N, 1, H, W) with connected component labels
                counts: Jittor tensor of shape (N, 1, H, W) with area of connected components
            """
            assert inputs.ndim == 4 and inputs.shape[1] == 1, "Input must be (N, 1, H, W)"
            assert inputs.dtype == 'uint8', "Input must be uint8"
            
            N, C, H, W = inputs.shape
            assert H % 2 == 0 and W % 2 == 0, "Height and width must be even"
            
            # Create output tensors
            labels = jt.zeros((N, C, H, W), dtype='int32')
            counts = jt.zeros((N, C, H, W), dtype='int32')
            
            # Create a temporary buffer (use double size for labels to store counts_init)
            temp_labels = jt.zeros((N * 2, C, H, W), dtype='int32')
            
            # Compile and execute CUDA kernel
            labels, counts = jt.code(
                inputs=[inputs],
                cuda_src="""
                namespace cc2d {

                template <typename T>
                __device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
                    return (bitmap >> pos) & 1;
                }
                
                __device__ int32_t find(const int32_t* s_buf, int32_t n) {
                    while (s_buf[n] != n)
                        n = s_buf[n];
                    return n;
                }
                
                __device__ int32_t find_n_compress(int32_t* s_buf, int32_t n) {
                    const int32_t id = n;
                    while (s_buf[n] != n) {
                        n = s_buf[n];
                        s_buf[id] = n;
                    }
                    return n;
                }
                
                __device__ void union_(int32_t* s_buf, int32_t a, int32_t b) {
                    bool done;
                    do {
                        a = find(s_buf, a);
                        b = find(s_buf, b);
                
                        if (a < b) {
                            int32_t old = atomicMin(s_buf + b, a);
                            done = (old == b);
                            b = old;
                        } else if (b < a) {
                            int32_t old = atomicMin(s_buf + a, b);
                            done = (old == a);
                            a = old;
                        } else
                            done = true;
                    } while (!done);
                }
                
                __global__ void init_labeling(int32_t* label, const uint32_t W, const uint32_t H) {
                    const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
                    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
                    const uint32_t idx = row * W + col;
                
                    if (row < H && col < W)
                        label[idx] = idx;
                }
                
                __global__ void merge(uint8_t* img, int32_t* label, const uint32_t W, const uint32_t H) {
                    const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
                    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
                    const uint32_t idx = row * W + col;
                
                    if (row >= H || col >= W)
                        return;
                
                    uint32_t P = 0;
                
                    if (img[idx])
                        P |= 0x777;
                    if (row + 1 < H && img[idx + W])
                        P |= 0x777 << 4;
                    if (col + 1 < W && img[idx + 1])
                        P |= 0x777 << 1;
                
                    if (col == 0)
                        P &= 0xEEEE;
                    if (col + 1 >= W)
                        P &= 0x3333;
                    else if (col + 2 >= W)
                        P &= 0x7777;
                
                    if (row == 0)
                        P &= 0xFFF0;
                    if (row + 1 >= H)
                        P &= 0xFF;
                
                    if (P > 0) {
                        if (hasBit(P, 0) && img[idx - W - 1]) {
                            union_(label, idx, idx - 2 * W - 2);
                        }
                
                        if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
                            union_(label, idx, idx - 2 * W);
                
                        if (hasBit(P, 3) && img[idx + 2 - W])
                            union_(label, idx, idx - 2 * W + 2);
                
                        if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
                            union_(label, idx, idx - 2);
                    }
                }
                
                __global__ void compression(int32_t* label, const int32_t W, const int32_t H) {
                    const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
                    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
                    const uint32_t idx = row * W + col;
                
                    if (row < H && col < W)
                        find_n_compress(label, idx);
                }
                
                __global__ void final_labeling(const uint8_t* img, int32_t* label, const int32_t W, const int32_t H) {
                    const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
                    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
                    const uint32_t idx = row * W + col;
                
                    if (row >= H || col >= W)
                        return;
                
                    int32_t y = label[idx] + 1;
                
                    if (img[idx])
                        label[idx] = y;
                    else
                        label[idx] = 0;
                
                    if (col + 1 < W) {
                        if (img[idx + 1])
                            label[idx + 1] = y;
                        else
                            label[idx + 1] = 0;
                
                        if (row + 1 < H) {
                            if (img[idx + W + 1])
                                label[idx + W + 1] = y;
                            else
                                label[idx + W + 1] = 0;
                        }
                    }
                
                    if (row + 1 < H) {
                        if (img[idx + W])
                            label[idx + W] = y;
                        else
                            label[idx + W] = 0;
                    }
                }
                
                __global__ void init_counting(const int32_t* label, int32_t* count_init, const int32_t W, const int32_t H) {
                    const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y);
                    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x);
                    const uint32_t idx = row * W + col;
                
                    if (row >= H || col >= W)
                        return;
                
                    int32_t y = label[idx];
                    if (y > 0) {
                        int32_t count_idx = y - 1;
                        atomicAdd(count_init + count_idx, 1);
                    }
                }
                
                __global__ void final_counting(const int32_t* label, const int32_t* count_init,
                                              int32_t* count_final, const int32_t W, const int32_t H) {
                    const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y);
                    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x);
                    const uint32_t idx = row * W + col;
                
                    if (row >= H || col >= W)
                        return;
                
                    int32_t y = label[idx];
                    if (y > 0) {
                        int32_t count_idx = y - 1;
                        count_final[idx] = count_init[count_idx];
                    } else {
                        count_final[idx] = 0;
                    }
                }
                
                std::vector<Var> get_connected_components(const Var& inputs) {
                    // 检查输入
                    ASSERT(inputs.is_cuda()) << "inputs must be a CUDA Var";
                    ASSERT(inputs.ndim() == 4) << "inputs must be [N, 1, H, W] shape";
                    ASSERT(inputs.dtype() == ns_uint8) << "inputs must be a uint8 type";
                
                    const uint32_t N = inputs.shape[0];
                    const uint32_t C = inputs.shape[1];
                    const uint32_t H = inputs.shape[2];
                    const uint32_t W = inputs.shape[3];
                
                    ASSERT(C == 1) << "inputs must be [N, 1, H, W] shape";
                    ASSERT((H % 2) == 0) << "height must be an even number";
                    ASSERT((W % 2) == 0) << "width must be an even number";
                
                    // 创建输出张量
                    Var labels = zeros({N, C, H, W}, ns_int32);
                    Var counts_init = zeros({N, C, H, W}, ns_int32);
                    Var counts_final = zeros({N, C, H, W}, ns_int32);
                
                    // 确保张量在GPU上
                    if (!labels.is_cuda()) {
                        labels = labels.cuda();
                    }
                    if (!counts_init.is_cuda()) {
                        counts_init = counts_init.cuda();
                    }
                    if (!counts_final.is_cuda()) {
                        counts_final = counts_final.cuda();
                    }
                
                    // 计算CUDA网格和块大小
                    dim3 grid(
                        ((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS,
                        ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS
                    );
                    dim3 block(BLOCK_COLS, BLOCK_ROWS);
                    dim3 grid_count(
                        (W + BLOCK_COLS) / BLOCK_COLS,
                        (H + BLOCK_ROWS) / BLOCK_ROWS
                    );
                    dim3 block_count(BLOCK_COLS, BLOCK_ROWS);
                
                    // 获取CUDA流
                    cudaStream_t stream = 0; // JitTor通常使用默认流
                
                    // 同步以确保数据准备就绪
                    inputs.sync();
                    labels.sync();
                    counts_init.sync();
                    counts_final.sync();
                
                    for (int n = 0; n < N; n++) {
                        uint32_t offset = n * H * W;
                
                        // 获取数据指针
                        uint8_t* inputs_ptr = inputs.ptr<uint8_t>();
                        int32_t* labels_ptr = labels.ptr<int32_t>();
                        int32_t* counts_init_ptr = counts_init.ptr<int32_t>();
                        int32_t* counts_final_ptr = counts_final.ptr<int32_t>();
                
                        // 调用CUDA kernels
                        cc2d::init_labeling<<<grid, block, 0, stream>>>(
                            labels_ptr + offset, W, H);
                        cc2d::merge<<<grid, block, 0, stream>>>(
                            inputs_ptr + offset,
                            labels_ptr + offset,
                            W, H);
                        cc2d::compression<<<grid, block, 0, stream>>>(
                            labels_ptr + offset, W, H);
                        cc2d::final_labeling<<<grid, block, 0, stream>>>(
                            inputs_ptr + offset,
                            labels_ptr + offset,
                            W, H);
                
                        // 计算每个像素的计数
                        cc2d::init_counting<<<grid_count, block_count, 0, stream>>>(
                            labels_ptr + offset,
                            counts_init_ptr + offset,
                            W, H);
                        cc2d::final_counting<<<grid_count, block_count, 0, stream>>>(
                            labels_ptr + offset,
                            counts_init_ptr + offset,
                            counts_final_ptr + offset,
                            W, H);
                    }
                
                    // 等待CUDA操作完成
                    cudaStreamSynchronize(stream);
                
                    // 返回结果
                    std::vector<Var> outputs;
                    outputs.push_back(labels);
                    outputs.push_back(counts_final);
                    return outputs;
                }
                
                } // namespace cc2d
                """,
                cuda_header="""
                #include <jittor/jittor.h>
                #include <jittor/var.h>
                #include <jittor/op.h>
                #include <jittor/misc/cuda_flags.h>
                #include <cuda.h>
                #include <cuda_runtime.h>
                #include <vector>
                
                // 2d
                #define BLOCK_ROWS 16
                #define BLOCK_COLS 16
                """
            )
            
            return labels, counts
        
        # Export the function
        __all__ = ['get_connected_componnets']
        '''
        with open(init_file, 'w') as f:
            f.write(init_content)
        print("CUDA extension compiled successfully!")
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
        else:
            raise e


class CustomBuildExt(build_ext):
    def run(self):
        compile_cuda_extension()
        super().run()


class CustomDevelop(develop):
    def run(self):
        compile_cuda_extension()
        super().run()


class CustomInstall(install):
    def run(self):
        compile_cuda_extension()
        super().run()

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(exclude="notebooks"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
    cmdclass={
        'build_ext': CustomBuildExt,
        'develop': CustomDevelop,
        'install': CustomInstall,
    },
)
