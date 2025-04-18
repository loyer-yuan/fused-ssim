#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_pipeline.h>


#define G_00 0.001028380123898387f
#define G_01 0.0075987582094967365f
#define G_02 0.036000773310661316f
#define G_03 0.10936068743467331f
#define G_04 0.21300552785396576f
#define G_05 0.26601171493530273f
#define G_06 0.21300552785396576f
#define G_07 0.10936068743467331f
#define G_08 0.036000773310661316f
#define G_09 0.0075987582094967365f
#define G_10 0.001028380123898387f

#define WARP_SIZE 32
#define WARP_NUM 1

#define WINDOW_SIZE 11
#define SLIDING_STEPS 1
#define SLIDING_TIMES (8*9)
#define SLIDING_LEN (SLIDING_TIMES * SLIDING_STEPS)

// #define TILE_SIZE 4
// #define TILE_SIZE 3
#define TILE_SIZE 2

#define RBUF_SIZE (10 + TILE_SIZE)

// block size
#define BX (WARP_SIZE * TILE_SIZE * WARP_NUM)
#define BY (SLIDING_LEN)

// shared memory size
#define SX (BX + 10)
#define SY (1)

// convolution scratchpad size
#define CX (BX)
#define CY (BY + 10)


/**
 * @brief Load pixels into shared memory asynchronously with only one warp.
 */
template<int len>
__device__ inline void ld1row_into_shared_async(float * __restrict__ s_row_pixels, const float * __restrict__ d_images, const int CH, const int H, const int W, const int warp_start_x, const int warp_start_y) {
  const int fillzero_y = (warp_start_y < 0 || warp_start_y >= H);
  // const int lane = threadIdx.x & 0x1f;
  const int lane = threadIdx.x;
  for (int i = lane; i < len; i += WARP_SIZE) {
    const int cur_x = warp_start_x + i;
    const int cur_y = warp_start_y;
    const int fillzero = fillzero_y || (cur_x < 0 || cur_x >= W);
    if (fillzero) {
      s_row_pixels[i] = 0.0f;
    } else {
      __pipeline_memcpy_async(s_row_pixels + i, d_images + blockIdx.z * H * W + cur_y * W + cur_x, sizeof(float));
    }
    // __pipeline_memcpy_async(s_row_pixels + i, d_images + blockIdx.z * H * W + cur_y * W + cur_x, sizeof(float), fillzero);
  }
  __pipeline_commit();
}


__device__ __forceinline__ float do_sq(float val) {
  return val * val;
}


template<bool sq>
__device__ __forceinline__ float do_conv_x(const float * __restrict__ pixels, int w) {
  float val = 0.0f;
  // calculate along width/x dimension
  if constexpr (sq) {
    val += G_00 * do_sq(pixels[w +  0]);
    val += G_01 * do_sq(pixels[w +  1]);
    val += G_02 * do_sq(pixels[w +  2]);
    val += G_03 * do_sq(pixels[w +  3]);
    val += G_04 * do_sq(pixels[w +  4]);
    val += G_05 * do_sq(pixels[w +  5]);
    val += G_06 * do_sq(pixels[w +  6]);
    val += G_07 * do_sq(pixels[w +  7]);
    val += G_08 * do_sq(pixels[w +  8]);
    val += G_09 * do_sq(pixels[w +  9]);
    val += G_10 * do_sq(pixels[w + 10]);
  } else {
    val += G_00 * pixels[w +  0];
    val += G_01 * pixels[w +  1];
    val += G_02 * pixels[w +  2];
    val += G_03 * pixels[w +  3];
    val += G_04 * pixels[w +  4];
    val += G_05 * pixels[w +  5];
    val += G_06 * pixels[w +  6];
    val += G_07 * pixels[w +  7];
    val += G_08 * pixels[w +  8];
    val += G_09 * pixels[w +  9];
    val += G_10 * pixels[w + 10];
  }
  return val;
}


/**
 * @brief Convolution along y dimension. Data flows within registers.
 * 
 * @param vin a input data
 */
__device__ __forceinline__ float do_conv_y_flow(const float &vin, float &v00, float &v01, float &v02, float &v03, float &v04, float &v05, float &v06, float &v07, float &v08, float &v09) {
  const float val = v09 + G_10 * vin;
  v09 = v08 + G_09 * vin;
  v08 = v07 + G_08 * vin;
  v07 = v06 + G_07 * vin;
  v06 = v05 + G_06 * vin;
  v05 = v04 + G_05 * vin;
  v04 = v03 + G_04 * vin;
  v03 = v02 + G_03 * vin;
  v02 = v01 + G_02 * vin;
  v01 = v00 + G_01 * vin;
  v00 =       G_00 * vin;
  return val;
}
template <int sliding_idx>
__device__ __forceinline__ void do_conv_y_flow_prefill(const float &vin, float &v00, float &v01, float &v02, float &v03, float &v04, float &v05, float &v06, float &v07, float &v08, float &v09) {
  if constexpr (sliding_idx > 8) { v09 = v08 + G_09 * vin; }
  if constexpr (sliding_idx > 7) { v08 = v07 + G_08 * vin; }
  if constexpr (sliding_idx > 6) { v07 = v06 + G_07 * vin; }
  if constexpr (sliding_idx > 5) { v06 = v05 + G_06 * vin; }
  if constexpr (sliding_idx > 4) { v05 = v04 + G_05 * vin; }
  if constexpr (sliding_idx > 3) { v04 = v03 + G_04 * vin; }
  if constexpr (sliding_idx > 2) { v03 = v02 + G_03 * vin; }
  if constexpr (sliding_idx > 1) { v02 = v01 + G_02 * vin; }
  if constexpr (sliding_idx > 0) { v01 = v00 + G_01 * vin; }
                                   v00 =       G_00 * vin;
}
template <int sliding_idx>
__device__ __forceinline__ float do_conv_y_flow_empty(const float &vin, float &v00, float &v01, float &v02, float &v03, float &v04, float &v05, float &v06, float &v07, float &v08, float &v09) {
  const float val = v09 + G_10 * vin; // If only execute this statment, sliding_idx should be SLIDING_LEN - 1;
  if constexpr (sliding_idx < SLIDING_LEN- 1) { v09 = v08 + G_09 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 2) { v08 = v07 + G_08 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 3) { v07 = v06 + G_07 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 4) { v06 = v05 + G_06 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 5) { v05 = v04 + G_05 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 6) { v04 = v03 + G_04 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 7) { v03 = v02 + G_03 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 8) { v02 = v01 + G_02 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN- 9) { v01 = v00 + G_01 * vin; }
  if constexpr (sliding_idx < SLIDING_LEN-10) { v00 =       G_00 * vin; }
  return val;
}


template<int len>
__device__ inline void mv_data_1d(float * __restrict__ dest, const float * __restrict__ src, int offset_x) {
  // mv_data_2d<len, 0, 0>(dest, src, 0, dest_x, 0, src_x);
  #pragma unroll
  for (int i = 0; i < len; ++i) {
    dest[i] = src[offset_x + i];
  }
}


template<int len>
__device__ inline void mv_mul_data_1d(float * __restrict__ dest, const float * __restrict__ src, int offset_x) {
  #pragma unroll
  for (int i = 0; i < len; ++i) {
    dest[i] *= src[offset_x + i];
  }
}


template<bool isLoad>
__device__ inline void stage_0_load_data_x_conv(
  float &rbuf1_0, float &rbuf1_1,
  float &rbuf1_sq_0, float &rbuf1_sq_1,
  float &rbuf2_0, float &rbuf2_1,
  float &rbuf2_sq_0, float &rbuf2_sq_1,
  float &rbuf3_0, float &rbuf3_1,
  float * __restrict__ rbuf0,
  float * __restrict__ rbuf0_mul,
  float * __restrict__ sbuf1,
  float * __restrict__ sbuf2,
  float const * __restrict__ img1,
  float const * __restrict__ img2,
  int CH, int H, int W,
  int block_data_x, int block_data_y, int thread_local_data_x, int i
) {
  // load data of img1
  __pipeline_wait_prior(1);
  mv_data_1d<RBUF_SIZE>(rbuf0, sbuf1, thread_local_data_x);
  mv_data_1d<RBUF_SIZE>(rbuf0_mul, sbuf1, thread_local_data_x);
  if constexpr (isLoad) {
    ld1row_into_shared_async<SX>(sbuf1, img1, CH, H, W, block_data_x-5, block_data_y-5+i+1); // load i+1 data
  }
  // do convolution of rbuf1
  rbuf1_0 = do_conv_x<false>(rbuf0, 0);
  rbuf1_1 = do_conv_x<false>(rbuf0, 1);
  // do convolution of rbuf1_sq
  rbuf1_sq_0 = do_conv_x<true>(rbuf0, 0);
  rbuf1_sq_1 = do_conv_x<true>(rbuf0, 1);

  // load data of img2
  __pipeline_wait_prior(1);
  mv_data_1d<RBUF_SIZE>(rbuf0, sbuf2, thread_local_data_x);
  mv_mul_data_1d<RBUF_SIZE>(rbuf0_mul, sbuf2, thread_local_data_x);
  if constexpr (isLoad) {
    ld1row_into_shared_async<SX>(sbuf2, img2, CH, H, W, block_data_x-5, block_data_y-5+i+1);  // load i+1 data
  }
  // do convolution of rbuf2
  rbuf2_0 = do_conv_x<false>(rbuf0, 0);
  rbuf2_1 = do_conv_x<false>(rbuf0, 1);
  // do convolution of rbuf2_sq
  rbuf2_sq_0 = do_conv_x<true>(rbuf0, 0);
  rbuf2_sq_1 = do_conv_x<true>(rbuf0, 1);
  // do convolution of rbuf3
  rbuf3_0 = do_conv_x<false>(rbuf0_mul, 0);
  rbuf3_1 = do_conv_x<false>(rbuf0_mul, 1);
}


/**
 * @brief Do convolution along y dimension. Data flows within registers.
 * 
 * @tparam state pipeline state, 0: prefill, 1: normal, 2: empty
 * @tparam sliding_idx sliding index, used when stage == 0 or 2
 */
template <int state, int sliding_idx = 0>
__device__ inline void stage_1_y_conv_flow(
  float &rbuf1_0_in, float &rbuf1_1_in, float &rbuf1_sq_0_in, float &rbuf1_sq_1_in, float &rbuf2_0_in, float &rbuf2_1_in, float &rbuf2_sq_0_in, float &rbuf2_sq_1_in, float &rbuf3_0_in, float &rbuf3_1_in,
  float &rbuf1_0_00, float &rbuf1_1_00, float &rbuf1_sq_0_00, float &rbuf1_sq_1_00, float &rbuf2_0_00, float &rbuf2_1_00, float &rbuf2_sq_0_00, float &rbuf2_sq_1_00, float &rbuf3_0_00, float &rbuf3_1_00,
  float &rbuf1_0_01, float &rbuf1_1_01, float &rbuf1_sq_0_01, float &rbuf1_sq_1_01, float &rbuf2_0_01, float &rbuf2_1_01, float &rbuf2_sq_0_01, float &rbuf2_sq_1_01, float &rbuf3_0_01, float &rbuf3_1_01,
  float &rbuf1_0_02, float &rbuf1_1_02, float &rbuf1_sq_0_02, float &rbuf1_sq_1_02, float &rbuf2_0_02, float &rbuf2_1_02, float &rbuf2_sq_0_02, float &rbuf2_sq_1_02, float &rbuf3_0_02, float &rbuf3_1_02,
  float &rbuf1_0_03, float &rbuf1_1_03, float &rbuf1_sq_0_03, float &rbuf1_sq_1_03, float &rbuf2_0_03, float &rbuf2_1_03, float &rbuf2_sq_0_03, float &rbuf2_sq_1_03, float &rbuf3_0_03, float &rbuf3_1_03,
  float &rbuf1_0_04, float &rbuf1_1_04, float &rbuf1_sq_0_04, float &rbuf1_sq_1_04, float &rbuf2_0_04, float &rbuf2_1_04, float &rbuf2_sq_0_04, float &rbuf2_sq_1_04, float &rbuf3_0_04, float &rbuf3_1_04,
  float &rbuf1_0_05, float &rbuf1_1_05, float &rbuf1_sq_0_05, float &rbuf1_sq_1_05, float &rbuf2_0_05, float &rbuf2_1_05, float &rbuf2_sq_0_05, float &rbuf2_sq_1_05, float &rbuf3_0_05, float &rbuf3_1_05,
  float &rbuf1_0_06, float &rbuf1_1_06, float &rbuf1_sq_0_06, float &rbuf1_sq_1_06, float &rbuf2_0_06, float &rbuf2_1_06, float &rbuf2_sq_0_06, float &rbuf2_sq_1_06, float &rbuf3_0_06, float &rbuf3_1_06,
  float &rbuf1_0_07, float &rbuf1_1_07, float &rbuf1_sq_0_07, float &rbuf1_sq_1_07, float &rbuf2_0_07, float &rbuf2_1_07, float &rbuf2_sq_0_07, float &rbuf2_sq_1_07, float &rbuf3_0_07, float &rbuf3_1_07,
  float &rbuf1_0_08, float &rbuf1_1_08, float &rbuf1_sq_0_08, float &rbuf1_sq_1_08, float &rbuf2_0_08, float &rbuf2_1_08, float &rbuf2_sq_0_08, float &rbuf2_sq_1_08, float &rbuf3_0_08, float &rbuf3_1_08,
  float &rbuf1_0_09, float &rbuf1_1_09, float &rbuf1_sq_0_09, float &rbuf1_sq_1_09, float &rbuf2_0_09, float &rbuf2_1_09, float &rbuf2_sq_0_09, float &rbuf2_sq_1_09, float &rbuf3_0_09, float &rbuf3_1_09,
  float * __restrict__ mu1_arr = nullptr,
  float * __restrict__ sigma1_sq_arr = nullptr,
  float * __restrict__ mu2_arr = nullptr,
  float * __restrict__ sigma2_sq_arr = nullptr,
  float * __restrict__ sigma12_arr = nullptr
) {
  if constexpr (state == 0) {
    do_conv_y_flow_prefill<sliding_idx>(rbuf1_0_in, rbuf1_0_00, rbuf1_0_01, rbuf1_0_02, rbuf1_0_03, rbuf1_0_04, rbuf1_0_05, rbuf1_0_06, rbuf1_0_07, rbuf1_0_08, rbuf1_0_09);
    do_conv_y_flow_prefill<sliding_idx>(rbuf1_1_in, rbuf1_1_00, rbuf1_1_01, rbuf1_1_02, rbuf1_1_03, rbuf1_1_04, rbuf1_1_05, rbuf1_1_06, rbuf1_1_07, rbuf1_1_08, rbuf1_1_09);

    do_conv_y_flow_prefill<sliding_idx>(rbuf1_sq_0_in, rbuf1_sq_0_00, rbuf1_sq_0_01, rbuf1_sq_0_02, rbuf1_sq_0_03, rbuf1_sq_0_04, rbuf1_sq_0_05, rbuf1_sq_0_06, rbuf1_sq_0_07, rbuf1_sq_0_08, rbuf1_sq_0_09);
    do_conv_y_flow_prefill<sliding_idx>(rbuf1_sq_1_in, rbuf1_sq_1_00, rbuf1_sq_1_01, rbuf1_sq_1_02, rbuf1_sq_1_03, rbuf1_sq_1_04, rbuf1_sq_1_05, rbuf1_sq_1_06, rbuf1_sq_1_07, rbuf1_sq_1_08, rbuf1_sq_1_09);

    do_conv_y_flow_prefill<sliding_idx>(rbuf2_0_in, rbuf2_0_00, rbuf2_0_01, rbuf2_0_02, rbuf2_0_03, rbuf2_0_04, rbuf2_0_05, rbuf2_0_06, rbuf2_0_07, rbuf2_0_08, rbuf2_0_09);
    do_conv_y_flow_prefill<sliding_idx>(rbuf2_1_in, rbuf2_1_00, rbuf2_1_01, rbuf2_1_02, rbuf2_1_03, rbuf2_1_04, rbuf2_1_05, rbuf2_1_06, rbuf2_1_07, rbuf2_1_08, rbuf2_1_09);

    do_conv_y_flow_prefill<sliding_idx>(rbuf2_sq_0_in, rbuf2_sq_0_00, rbuf2_sq_0_01, rbuf2_sq_0_02, rbuf2_sq_0_03, rbuf2_sq_0_04, rbuf2_sq_0_05, rbuf2_sq_0_06, rbuf2_sq_0_07, rbuf2_sq_0_08, rbuf2_sq_0_09);
    do_conv_y_flow_prefill<sliding_idx>(rbuf2_sq_1_in, rbuf2_sq_1_00, rbuf2_sq_1_01, rbuf2_sq_1_02, rbuf2_sq_1_03, rbuf2_sq_1_04, rbuf2_sq_1_05, rbuf2_sq_1_06, rbuf2_sq_1_07, rbuf2_sq_1_08, rbuf2_sq_1_09);

    do_conv_y_flow_prefill<sliding_idx>(rbuf3_0_in, rbuf3_0_00, rbuf3_0_01, rbuf3_0_02, rbuf3_0_03, rbuf3_0_04, rbuf3_0_05, rbuf3_0_06, rbuf3_0_07, rbuf3_0_08, rbuf3_0_09);
    do_conv_y_flow_prefill<sliding_idx>(rbuf3_1_in, rbuf3_1_00, rbuf3_1_01, rbuf3_1_02, rbuf3_1_03, rbuf3_1_04, rbuf3_1_05, rbuf3_1_06, rbuf3_1_07, rbuf3_1_08, rbuf3_1_09);
  }
  if constexpr (state == 1) {
    // do convolution of mu1arr
    mu1_arr[0] = do_conv_y_flow(rbuf1_0_in, rbuf1_0_00, rbuf1_0_01, rbuf1_0_02, rbuf1_0_03, rbuf1_0_04, rbuf1_0_05, rbuf1_0_06, rbuf1_0_07, rbuf1_0_08, rbuf1_0_09);
    mu1_arr[1] = do_conv_y_flow(rbuf1_1_in, rbuf1_1_00, rbuf1_1_01, rbuf1_1_02, rbuf1_1_03, rbuf1_1_04, rbuf1_1_05, rbuf1_1_06, rbuf1_1_07, rbuf1_1_08, rbuf1_1_09);
    // do convolution of sigma1_sq_arr
    sigma1_sq_arr[0] = do_conv_y_flow(rbuf1_sq_0_in, rbuf1_sq_0_00, rbuf1_sq_0_01, rbuf1_sq_0_02, rbuf1_sq_0_03, rbuf1_sq_0_04, rbuf1_sq_0_05, rbuf1_sq_0_06, rbuf1_sq_0_07, rbuf1_sq_0_08, rbuf1_sq_0_09) - do_sq(mu1_arr[0]);
    sigma1_sq_arr[1] = do_conv_y_flow(rbuf1_sq_1_in, rbuf1_sq_1_00, rbuf1_sq_1_01, rbuf1_sq_1_02, rbuf1_sq_1_03, rbuf1_sq_1_04, rbuf1_sq_1_05, rbuf1_sq_1_06, rbuf1_sq_1_07, rbuf1_sq_1_08, rbuf1_sq_1_09) - do_sq(mu1_arr[1]);
    // do convolution of mu2_arr
    mu2_arr[0] = do_conv_y_flow(rbuf2_0_in, rbuf2_0_00, rbuf2_0_01, rbuf2_0_02, rbuf2_0_03, rbuf2_0_04, rbuf2_0_05, rbuf2_0_06, rbuf2_0_07, rbuf2_0_08, rbuf2_0_09);
    mu2_arr[1] = do_conv_y_flow(rbuf2_1_in, rbuf2_1_00, rbuf2_1_01, rbuf2_1_02, rbuf2_1_03, rbuf2_1_04, rbuf2_1_05, rbuf2_1_06, rbuf2_1_07, rbuf2_1_08, rbuf2_1_09);
    // do convolution of sigma2_sq_arr
    sigma2_sq_arr[0] = do_conv_y_flow(rbuf2_sq_0_in, rbuf2_sq_0_00, rbuf2_sq_0_01, rbuf2_sq_0_02, rbuf2_sq_0_03, rbuf2_sq_0_04, rbuf2_sq_0_05, rbuf2_sq_0_06, rbuf2_sq_0_07, rbuf2_sq_0_08, rbuf2_sq_0_09) - do_sq(mu2_arr[0]);
    sigma2_sq_arr[1] = do_conv_y_flow(rbuf2_sq_1_in, rbuf2_sq_1_00, rbuf2_sq_1_01, rbuf2_sq_1_02, rbuf2_sq_1_03, rbuf2_sq_1_04, rbuf2_sq_1_05, rbuf2_sq_1_06, rbuf2_sq_1_07, rbuf2_sq_1_08, rbuf2_sq_1_09) - do_sq(mu2_arr[1]);
    // do convolution of sigma12_arr
    sigma12_arr[0] = do_conv_y_flow(rbuf3_0_in, rbuf3_0_00, rbuf3_0_01, rbuf3_0_02, rbuf3_0_03, rbuf3_0_04, rbuf3_0_05, rbuf3_0_06, rbuf3_0_07, rbuf3_0_08, rbuf3_0_09) - mu1_arr[0] * mu2_arr[0];
    sigma12_arr[1] = do_conv_y_flow(rbuf3_1_in, rbuf3_1_00, rbuf3_1_01, rbuf3_1_02, rbuf3_1_03, rbuf3_1_04, rbuf3_1_05, rbuf3_1_06, rbuf3_1_07, rbuf3_1_08, rbuf3_1_09) - mu1_arr[1] * mu2_arr[1];
  }
  if constexpr (state == 2) {
    // do convolution of mu1arr
    mu1_arr[0] = do_conv_y_flow_empty<sliding_idx>(rbuf1_0_in, rbuf1_0_00, rbuf1_0_01, rbuf1_0_02, rbuf1_0_03, rbuf1_0_04, rbuf1_0_05, rbuf1_0_06, rbuf1_0_07, rbuf1_0_08, rbuf1_0_09);
    mu1_arr[1] = do_conv_y_flow_empty<sliding_idx>(rbuf1_1_in, rbuf1_1_00, rbuf1_1_01, rbuf1_1_02, rbuf1_1_03, rbuf1_1_04, rbuf1_1_05, rbuf1_1_06, rbuf1_1_07, rbuf1_1_08, rbuf1_1_09);
    // do convolution of sigma1_sq_arr
    sigma1_sq_arr[0] = do_conv_y_flow_empty<sliding_idx>(rbuf1_sq_0_in, rbuf1_sq_0_00, rbuf1_sq_0_01, rbuf1_sq_0_02, rbuf1_sq_0_03, rbuf1_sq_0_04, rbuf1_sq_0_05, rbuf1_sq_0_06, rbuf1_sq_0_07, rbuf1_sq_0_08, rbuf1_sq_0_09) - do_sq(mu1_arr[0]);
    sigma1_sq_arr[1] = do_conv_y_flow_empty<sliding_idx>(rbuf1_sq_1_in, rbuf1_sq_1_00, rbuf1_sq_1_01, rbuf1_sq_1_02, rbuf1_sq_1_03, rbuf1_sq_1_04, rbuf1_sq_1_05, rbuf1_sq_1_06, rbuf1_sq_1_07, rbuf1_sq_1_08, rbuf1_sq_1_09) - do_sq(mu1_arr[1]);
    // do convolution of mu2_arr
    mu2_arr[0] = do_conv_y_flow_empty<sliding_idx>(rbuf2_0_in, rbuf2_0_00, rbuf2_0_01, rbuf2_0_02, rbuf2_0_03, rbuf2_0_04, rbuf2_0_05, rbuf2_0_06, rbuf2_0_07, rbuf2_0_08, rbuf2_0_09);
    mu2_arr[1] = do_conv_y_flow_empty<sliding_idx>(rbuf2_1_in, rbuf2_1_00, rbuf2_1_01, rbuf2_1_02, rbuf2_1_03, rbuf2_1_04, rbuf2_1_05, rbuf2_1_06, rbuf2_1_07, rbuf2_1_08, rbuf2_1_09);
    // do convolution of sigma2_sq_arr
    sigma2_sq_arr[0] = do_conv_y_flow_empty<sliding_idx>(rbuf2_sq_0_in, rbuf2_sq_0_00, rbuf2_sq_0_01, rbuf2_sq_0_02, rbuf2_sq_0_03, rbuf2_sq_0_04, rbuf2_sq_0_05, rbuf2_sq_0_06, rbuf2_sq_0_07, rbuf2_sq_0_08, rbuf2_sq_0_09) - do_sq(mu2_arr[0]);
    sigma2_sq_arr[1] = do_conv_y_flow_empty<sliding_idx>(rbuf2_sq_1_in, rbuf2_sq_1_00, rbuf2_sq_1_01, rbuf2_sq_1_02, rbuf2_sq_1_03, rbuf2_sq_1_04, rbuf2_sq_1_05, rbuf2_sq_1_06, rbuf2_sq_1_07, rbuf2_sq_1_08, rbuf2_sq_1_09) - do_sq(mu2_arr[1]);
    // do convolution of sigma12_arr
    sigma12_arr[0] = do_conv_y_flow_empty<sliding_idx>(rbuf3_0_in, rbuf3_0_00, rbuf3_0_01, rbuf3_0_02, rbuf3_0_03, rbuf3_0_04, rbuf3_0_05, rbuf3_0_06, rbuf3_0_07, rbuf3_0_08, rbuf3_0_09) - mu1_arr[0] * mu2_arr[0];
    sigma12_arr[1] = do_conv_y_flow_empty<sliding_idx>(rbuf3_1_in, rbuf3_1_00, rbuf3_1_01, rbuf3_1_02, rbuf3_1_03, rbuf3_1_04, rbuf3_1_05, rbuf3_1_06, rbuf3_1_07, rbuf3_1_08, rbuf3_1_09) - mu1_arr[1] * mu2_arr[1];
  }
}


__device__ inline void stage_2_compute_ssim_store(
  const float * __restrict__ mu1_arr, const float * __restrict__ sigma1_sq_arr, const float * __restrict__ mu2_arr, const float * __restrict__ sigma2_sq_arr, const float * __restrict__ sigma12_arr,
  float * __restrict__ ssim_map, float * __restrict__ dm_dmu1, float * __restrict__ dm_dsigma1_sq, float * __restrict__ dm_dsigma12,
  const float C1, const float C2, const int channel, const int H, const int W, const int block_data_x, const int block_data_y, const int thread_local_data_x, const int sliding_idx
) {
  #pragma unroll
  for (int ii = 0; ii < TILE_SIZE; ii++) {
    const int cur_x = block_data_x + thread_local_data_x + ii;
    const int cur_y = block_data_y + sliding_idx;
    const float mu1 = mu1_arr[ii];
    const float mu2 = mu2_arr[ii];
    const float sigma1_sq = sigma1_sq_arr[ii];
    const float sigma2_sq = sigma2_sq_arr[ii];
    const float sigma12 = sigma12_arr[ii];

    const float C = (2.0f * mu1 * mu2 + C1);
    const float D = (2.0f * sigma12 + C2);
    const float A = (mu1 * mu1 + mu2 * mu2 + C1);
    const float B = (sigma1_sq + sigma2_sq + C2);
    const float m = (C * D) / (A * B);

    const int global_idx = channel * H * W + cur_y * W + cur_x;
    ssim_map[global_idx] = m;
    if (dm_dmu1) {
      dm_dmu1[global_idx] = (
        (mu2 * 2.0f * D) / (A * B)
        -(mu2 * 2.0f * C) / (A * B)
        -(mu1 * 2.0f * C * D) / ( A * A * B)
        +(mu1 * 2.0f * C * D) / (A * B * B)
      );
      if (cur_x < W && cur_y < H) {
        dm_dsigma1_sq[global_idx] = ((-C * D) / (A * B * B));
        dm_dsigma12[global_idx] = ((2 * C) / (A * B));
      }
    }
  }
}


#define CONST_PARAMETERS rbuf0, rbuf0_mul, sbuf1, sbuf2, img1, img2, CH, H, W, block_data_x, block_data_y, thread_local_data_x
#define REGS_00 rbuf1_0_00, rbuf1_1_00, rbuf1_sq_0_00, rbuf1_sq_1_00, rbuf2_0_00, rbuf2_1_00, rbuf2_sq_0_00, rbuf2_sq_1_00, rbuf3_0_00, rbuf3_1_00
#define REGS_01 rbuf1_0_01, rbuf1_1_01, rbuf1_sq_0_01, rbuf1_sq_1_01, rbuf2_0_01, rbuf2_1_01, rbuf2_sq_0_01, rbuf2_sq_1_01, rbuf3_0_01, rbuf3_1_01
#define REGS_02 rbuf1_0_02, rbuf1_1_02, rbuf1_sq_0_02, rbuf1_sq_1_02, rbuf2_0_02, rbuf2_1_02, rbuf2_sq_0_02, rbuf2_sq_1_02, rbuf3_0_02, rbuf3_1_02
#define REGS_03 rbuf1_0_03, rbuf1_1_03, rbuf1_sq_0_03, rbuf1_sq_1_03, rbuf2_0_03, rbuf2_1_03, rbuf2_sq_0_03, rbuf2_sq_1_03, rbuf3_0_03, rbuf3_1_03
#define REGS_04 rbuf1_0_04, rbuf1_1_04, rbuf1_sq_0_04, rbuf1_sq_1_04, rbuf2_0_04, rbuf2_1_04, rbuf2_sq_0_04, rbuf2_sq_1_04, rbuf3_0_04, rbuf3_1_04
#define REGS_05 rbuf1_0_05, rbuf1_1_05, rbuf1_sq_0_05, rbuf1_sq_1_05, rbuf2_0_05, rbuf2_1_05, rbuf2_sq_0_05, rbuf2_sq_1_05, rbuf3_0_05, rbuf3_1_05
#define REGS_06 rbuf1_0_06, rbuf1_1_06, rbuf1_sq_0_06, rbuf1_sq_1_06, rbuf2_0_06, rbuf2_1_06, rbuf2_sq_0_06, rbuf2_sq_1_06, rbuf3_0_06, rbuf3_1_06
#define REGS_07 rbuf1_0_07, rbuf1_1_07, rbuf1_sq_0_07, rbuf1_sq_1_07, rbuf2_0_07, rbuf2_1_07, rbuf2_sq_0_07, rbuf2_sq_1_07, rbuf3_0_07, rbuf3_1_07
#define REGS_08 rbuf1_0_08, rbuf1_1_08, rbuf1_sq_0_08, rbuf1_sq_1_08, rbuf2_0_08, rbuf2_1_08, rbuf2_sq_0_08, rbuf2_sq_1_08, rbuf3_0_08, rbuf3_1_08
#define REGS_09 rbuf1_0_09, rbuf1_1_09, rbuf1_sq_0_09, rbuf1_sq_1_09, rbuf2_0_09, rbuf2_1_09, rbuf2_sq_0_09, rbuf2_sq_1_09, rbuf3_0_09, rbuf3_1_09
#define REGS_in rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in

__global__ void fusedssimCUDA(
  int H,
  int W,
  int CH,
  float C1,
  float C2,
  float* img1,
  float* img2,
  float* ssim_map,
  float* dm_dmu1 = nullptr,
  float* dm_dsigma1_sq = nullptr,
  float* dm_dsigma12 = nullptr
) {
  assert(BY >= SLIDING_STEPS*2);

  const int block_data_x = blockIdx.x * BX;
  const int block_data_y = blockIdx.y * BY;
  const int thread_local_data_x = threadIdx.x*TILE_SIZE;
  const int channel = blockIdx.z;

  __shared__ float sbuf1[SX];
  __shared__ float sbuf2[SX];
  float rbuf0[RBUF_SIZE];
  float rbuf0_mul[RBUF_SIZE];

  // float rbuf1[TILE_SIZE*WINDOW_SIZE];
  // float rbuf1_sq[TILE_SIZE*WINDOW_SIZE];
  // float rbuf2[TILE_SIZE*WINDOW_SIZE];
  // float rbuf2_sq[TILE_SIZE*WINDOW_SIZE];
  // float rbuf3[TILE_SIZE*WINDOW_SIZE];
  float rbuf1_0_00, rbuf1_0_01, rbuf1_0_02, rbuf1_0_03, rbuf1_0_04, rbuf1_0_05, rbuf1_0_06, rbuf1_0_07, rbuf1_0_08, rbuf1_0_09, rbuf1_0_in;
  float rbuf1_1_00, rbuf1_1_01, rbuf1_1_02, rbuf1_1_03, rbuf1_1_04, rbuf1_1_05, rbuf1_1_06, rbuf1_1_07, rbuf1_1_08, rbuf1_1_09, rbuf1_1_in;
  float rbuf1_sq_0_00, rbuf1_sq_0_01, rbuf1_sq_0_02, rbuf1_sq_0_03, rbuf1_sq_0_04, rbuf1_sq_0_05, rbuf1_sq_0_06, rbuf1_sq_0_07, rbuf1_sq_0_08, rbuf1_sq_0_09, rbuf1_sq_0_in;
  float rbuf1_sq_1_00, rbuf1_sq_1_01, rbuf1_sq_1_02, rbuf1_sq_1_03, rbuf1_sq_1_04, rbuf1_sq_1_05, rbuf1_sq_1_06, rbuf1_sq_1_07, rbuf1_sq_1_08, rbuf1_sq_1_09, rbuf1_sq_1_in;
  float rbuf2_0_00, rbuf2_0_01, rbuf2_0_02, rbuf2_0_03, rbuf2_0_04, rbuf2_0_05, rbuf2_0_06, rbuf2_0_07, rbuf2_0_08, rbuf2_0_09, rbuf2_0_in;
  float rbuf2_1_00, rbuf2_1_01, rbuf2_1_02, rbuf2_1_03, rbuf2_1_04, rbuf2_1_05, rbuf2_1_06, rbuf2_1_07, rbuf2_1_08, rbuf2_1_09, rbuf2_1_in;
  float rbuf2_sq_0_00, rbuf2_sq_0_01, rbuf2_sq_0_02, rbuf2_sq_0_03, rbuf2_sq_0_04, rbuf2_sq_0_05, rbuf2_sq_0_06, rbuf2_sq_0_07, rbuf2_sq_0_08, rbuf2_sq_0_09, rbuf2_sq_0_in;
  float rbuf2_sq_1_00, rbuf2_sq_1_01, rbuf2_sq_1_02, rbuf2_sq_1_03, rbuf2_sq_1_04, rbuf2_sq_1_05, rbuf2_sq_1_06, rbuf2_sq_1_07, rbuf2_sq_1_08, rbuf2_sq_1_09, rbuf2_sq_1_in;
  float rbuf3_0_00, rbuf3_0_01, rbuf3_0_02, rbuf3_0_03, rbuf3_0_04, rbuf3_0_05, rbuf3_0_06, rbuf3_0_07, rbuf3_0_08, rbuf3_0_09, rbuf3_0_in;
  float rbuf3_1_00, rbuf3_1_01, rbuf3_1_02, rbuf3_1_03, rbuf3_1_04, rbuf3_1_05, rbuf3_1_06, rbuf3_1_07, rbuf3_1_08, rbuf3_1_09, rbuf3_1_in;
  
  float mu1_arr[TILE_SIZE];
  float sigma1_sq_arr[TILE_SIZE];
  float mu2_arr[TILE_SIZE];
  float sigma2_sq_arr[TILE_SIZE];
  float sigma12_arr[TILE_SIZE];
  
  // fulfill the pipeline
  ld1row_into_shared_async<SX>(sbuf1, img1, CH, H, W, block_data_x-5, block_data_y-5);
  ld1row_into_shared_async<SX>(sbuf2, img2, CH, H, W, block_data_x-5, block_data_y-5);

  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 0
  );
  stage_1_y_conv_flow<0, 0>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 1
  );
  stage_1_y_conv_flow<0, 1>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 2
  );
  stage_1_y_conv_flow<0, 2>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 3
  );
  stage_1_y_conv_flow<0, 3>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 4
  );
  stage_1_y_conv_flow<0, 4>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 5
  );
  stage_1_y_conv_flow<0, 5>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 6
  );
  stage_1_y_conv_flow<0, 6>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 7
  );
  stage_1_y_conv_flow<0, 7>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 8
  );
  stage_1_y_conv_flow<0, 8>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  stage_0_load_data_x_conv<true>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, 9
  );
  stage_1_y_conv_flow<0, 9>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09);
  
  // Sliding the window, wuhu!
  // constexpr int REST_TIMES = (WINDOW_SIZE-1); 
  constexpr int REST_TIMES = 1; 
  constexpr int LOOP_TIMES = SLIDING_TIMES - REST_TIMES;
  int sliding_idx = 0;
  for (; sliding_idx < LOOP_TIMES; sliding_idx++) {
    stage_0_load_data_x_conv<true>(
      rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
      CONST_PARAMETERS, (10+sliding_idx)
    );
    stage_1_y_conv_flow<1>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09, mu1_arr, sigma1_sq_arr, mu2_arr, sigma2_sq_arr, sigma12_arr);
    stage_2_compute_ssim_store(mu1_arr, sigma1_sq_arr, mu2_arr, sigma2_sq_arr, sigma12_arr, 
      ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, C1, C2, channel, H, W, block_data_x, block_data_y, thread_local_data_x,
      sliding_idx);
  }

  // Rest
  stage_0_load_data_x_conv<false>(
    rbuf1_0_in, rbuf1_1_in, rbuf1_sq_0_in, rbuf1_sq_1_in, rbuf2_0_in, rbuf2_1_in, rbuf2_sq_0_in, rbuf2_sq_1_in, rbuf3_0_in, rbuf3_1_in, 
    CONST_PARAMETERS, (10+sliding_idx)
  );
  stage_1_y_conv_flow<2, LOOP_TIMES>(REGS_in, REGS_00, REGS_01, REGS_02, REGS_03, REGS_04, REGS_05, REGS_06, REGS_07, REGS_08, REGS_09, mu1_arr, sigma1_sq_arr, mu2_arr, sigma2_sq_arr, sigma12_arr);
  stage_2_compute_ssim_store(mu1_arr, sigma1_sq_arr, mu2_arr, sigma2_sq_arr, sigma12_arr, 
    ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, C1, C2, channel, H, W, block_data_x, block_data_y, thread_local_data_x,
    sliding_idx);
}

#undef CONST_PARAMETERS
#undef REGS_00
#undef REGS_01
#undef REGS_02
#undef REGS_03
#undef REGS_04
#undef REGS_05
#undef REGS_06
#undef REGS_07
#undef REGS_08
#undef REGS_09
#undef REGS_10

__global__ void fusedssim_backwardCUDA(
  int H,
  int W,
  int CH,
  float C1,
  float C2,
  float* img1,
  float* img2,
  float *dL_dmap,
  float *dL_dimg1,
  float* dm_dmu1 = nullptr,
  float* dm_dsigma1_sq = nullptr,
  float* dm_dsigma12 = nullptr
)
{
  // auto block = cg::this_thread_block();
  // const int pix_y = block.group_index().y * BY + block.thread_index().y;
  // const int pix_x = block.group_index().x * BX + block.thread_index().x;
  // const int pix_id = pix_y * W + pix_x;
  // const int num_pix = H * W;
  // const int batch = block.group_index().z;

  // // shared memory that will be used to load pixels temporarily
  // __shared__ float buf1[SY][SX];
  // __shared__ float buf2[SY][SX];
  // __shared__ float buf3[CY][CX];

  // for (int i = 0; i < CH; ++i) {
  //   float dL_dpix = 0.0f;
  //   float tmp = 0.0f;
  //   float pix1 = get_pix_value(img1, batch, i, pix_y, pix_x, CH, H, W);
  //   float pix2 = get_pix_value(img2, batch, i, pix_y, pix_x, CH, H, W);
  //   load_into_shared(buf1, dL_dmap, CH, H, W, i);

  //   // gradient from mu1
  //   load_into_shared(buf2, dm_dmu1, CH, H, W, i);
  //   block.sync();
  //   multiply_shared_mem(buf2, buf1);
  //   block.sync();
  //   flush_conv_scratch(buf3);
  //   block.sync();
  //   do_separable_conv_x(buf2, buf3, H, W);
  //   block.sync();
  //   tmp = do_separable_conv_y(buf3, H, W);
  //   block.sync();
  //   dL_dpix += tmp;

  //   // gradient from sigma1_sq
  //   load_into_shared(buf2, dm_dsigma1_sq, CH, H, W, i);
  //   block.sync();
  //   multiply_shared_mem(buf2, buf1);
  //   block.sync();
  //   flush_conv_scratch(buf3);
  //   block.sync();
  //   do_separable_conv_x(buf2, buf3, H, W);
  //   block.sync();
  //   tmp = pix1 * 2.0f * do_separable_conv_y(buf3, H, W);
  //   block.sync();
  //   dL_dpix += tmp;

  //   // gradient from sigma12
  //   load_into_shared(buf2, dm_dsigma12, CH, H, W, i);
  //   block.sync();
  //   multiply_shared_mem(buf2, buf1);
  //   block.sync();
  //   flush_conv_scratch(buf3);
  //   block.sync();
  //   do_separable_conv_x(buf2, buf3, H, W);
  //   block.sync();
  //   tmp = pix2 * do_separable_conv_y(buf3, H, W);
  //   block.sync();
  //   dL_dpix += tmp;

  //   if (pix_x < W && pix_y < H) {
  //     const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
  //     dL_dimg1[global_idx] = dL_dpix;
  //   }
  // }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim_opt(
  float C1,
  float C2,
  torch::Tensor &img1,
  torch::Tensor &img2,
  bool train
)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);
  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, CH);
  // dim3 block(BX, BY, 1);
  dim3 block(WARP_SIZE);
  // printf("B: %d, CH: %d, H: %d, W: %d\n", B, CH, H, W);
  // printf("grid: %d %d %d\n", grid.x, grid.y, grid.z);
  // printf("block: %d %d %d\n", block.x, block.y, block.z);

  torch::Tensor target = torch::zeros_like(img1).contiguous();
  torch::Tensor dm_dmu1 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  torch::Tensor dm_dsigma1_sq = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  torch::Tensor dm_dsigma12 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  fusedssimCUDA<<<grid,block>>>(
    H,
    W,
    CH,
    C1,
    C2,
    img1.contiguous().data<float>(),
    img2.contiguous().data<float>(),
    target.contiguous().data<float>(),
    dm_dmu1.contiguous().data<float>(),
    dm_dsigma1_sq.contiguous().data<float>(),
    dm_dsigma12.contiguous().data<float>()
  );

  return std::make_tuple(target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

torch::Tensor
fusedssim_opt_backward(
  float C1,
  float C2,
  torch::Tensor &img1,
  torch::Tensor &img2,
  torch::Tensor &dL_dmap,
  torch::Tensor &dm_dmu1,
  torch::Tensor &dm_dsigma1_sq,
  torch::Tensor &dm_dsigma12
)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);

  torch::Tensor dL_dimg1 = torch::zeros_like(img1).contiguous();

  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
  dim3 block(BX, BY, 1);
  fusedssim_backwardCUDA<<<grid,block>>>(
    H,
    W,
    CH,
    C1,
    C2,
    img1.contiguous().data<float>(),
    img2.contiguous().data<float>(),
    dL_dmap.contiguous().data<float>(),
    dL_dimg1.contiguous().data<float>(),
    dm_dmu1.contiguous().data<float>(),
    dm_dsigma1_sq.contiguous().data<float>(),
    dm_dsigma12.contiguous().data<float>()
  );

  return dL_dimg1;
}
