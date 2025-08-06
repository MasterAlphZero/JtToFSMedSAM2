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