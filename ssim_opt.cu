#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

#define DEBUG_CON (block.group_index().x == 1 && block.group_index().y == 1 && block.group_index().z == 0 && block.thread_index().x == 0 && block.thread_index().y == 0)

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
#define WARP_NUM 8
// #define TILE_SIZE 4
// #define TILE_SIZE 3
#define TILE_SIZE 2

#define RBUF_SIZE (10 + TILE_SIZE)

// block size
#define BX (WARP_SIZE * TILE_SIZE)
#define BY (WARP_NUM)

// shared memory size
#define SX (BX + 10)
#define SY (BY + 10)

// convolution scratchpad size
#define CX (BX)
#define CY (BY + 10)


/**
 * Get pixel value from image
 * 
 * @param img Image tensor
 * @param b Batch index
 * @param c Channel index
 * @param y Y coordinate
 * @param x X coordinate
 * @param CH Number of channels
 * @param H Height of image
 * @param W Width of image
 */
__device__ __forceinline__ float get_pix_value(const float* img, const int b, const int c, const int y, const int x, const int CH, const int H, const int W) {
  if (x >= W || y >= H || x < 0 || y < 0) {
    return 0.0f;
  } else {
    return img[b * CH * H * W + c * H * W + y * W + x];
  }
}

// TODO: Use float4 for loading 4 pixels at once
// TODO: Use async copy for loading pixels
__device__ void load_into_shared(float * __restrict__ pixels, const float * __restrict__ inp, const int CH, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int batch = 0;
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY * SX;
  const int thread_num = WARP_NUM * WARP_SIZE;
  const int num_blocks = (cnt + thread_num - 1) / (thread_num);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (thread_num) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX;
      int local_x = tid % SX;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, batch, i, y - 5, x - 5, CH, H, W);
      pixels[local_y*SX + local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem(float * __restrict__ pix1, const float * __restrict__ pix2) {
  auto block = cg::this_thread_block();
  const int cnt = SY * SX;
  const int thread_num = WARP_NUM * WARP_SIZE;
  const int num_blocks = (cnt + thread_num - 1) / (thread_num);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (thread_num) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX;
      int local_x = tid % SX;
      float one = pix1[local_y*SX + local_x];
      float two = pix2[local_y*SX + local_x];
      pix1[local_y*SX + local_x] = one * two;
    }
  }
}

__device__ void
flush_conv_scratch(float buf[CY][CX]) {
  auto block = cg::this_thread_block();
  const int cnt = CY * CX;
  const int thread_num = WARP_NUM * WARP_SIZE;
  const int num_blocks = (cnt + thread_num - 1) / (thread_num);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (thread_num) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX;
      const int local_x = tid % CX;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ __forceinline__ float do_sq(float val) {
  return val * val;
}

template<int width, bool do_width, bool sq>
__device__ __forceinline__ float do_conv(const float * __restrict__ pixels, int h, int w) {
  float val = 0.0f;
  // calculate along width/x dimension
  if constexpr (do_width) {
    if constexpr (sq) {
      val += G_00 * do_sq(pixels[h*width + w - 5]);
      val += G_01 * do_sq(pixels[h*width + w - 4]);
      val += G_02 * do_sq(pixels[h*width + w - 3]);
      val += G_03 * do_sq(pixels[h*width + w - 2]);
      val += G_04 * do_sq(pixels[h*width + w - 1]);
      val += G_05 * do_sq(pixels[h*width + w    ]);
      val += G_06 * do_sq(pixels[h*width + w + 1]);
      val += G_07 * do_sq(pixels[h*width + w + 2]);
      val += G_08 * do_sq(pixels[h*width + w + 3]);
      val += G_09 * do_sq(pixels[h*width + w + 4]);
      val += G_10 * do_sq(pixels[h*width + w + 5]);
    } else {
      val += G_00 * pixels[h*width + w - 5];
      val += G_01 * pixels[h*width + w - 4];
      val += G_02 * pixels[h*width + w - 3];
      val += G_03 * pixels[h*width + w - 2];
      val += G_04 * pixels[h*width + w - 1];
      val += G_05 * pixels[h*width + w    ];
      val += G_06 * pixels[h*width + w + 1];
      val += G_07 * pixels[h*width + w + 2];
      val += G_08 * pixels[h*width + w + 3];
      val += G_09 * pixels[h*width + w + 4];
      val += G_10 * pixels[h*width + w + 5];
    }
  } else {
    if constexpr (sq) {
      val += G_00 * do_sq(pixels[(h - 5)*width + w]);
      val += G_01 * do_sq(pixels[(h - 4)*width + w]);
      val += G_02 * do_sq(pixels[(h - 3)*width + w]);
      val += G_03 * do_sq(pixels[(h - 2)*width + w]);
      val += G_04 * do_sq(pixels[(h - 1)*width + w]);
      val += G_05 * do_sq(pixels[(h    )*width + w]);
      val += G_06 * do_sq(pixels[(h + 1)*width + w]);
      val += G_07 * do_sq(pixels[(h + 2)*width + w]);
      val += G_08 * do_sq(pixels[(h + 3)*width + w]);
      val += G_09 * do_sq(pixels[(h + 4)*width + w]);
      val += G_10 * do_sq(pixels[(h + 5)*width + w]);
    } else {
      val += G_00 * pixels[(h - 5)*width + w];
      val += G_01 * pixels[(h - 4)*width + w];
      val += G_02 * pixels[(h - 3)*width + w];
      val += G_03 * pixels[(h - 2)*width + w];
      val += G_04 * pixels[(h - 1)*width + w];
      val += G_05 * pixels[(h    )*width + w];
      val += G_06 * pixels[(h + 1)*width + w];
      val += G_07 * pixels[(h + 2)*width + w];
      val += G_08 * pixels[(h + 3)*width + w];
      val += G_09 * pixels[(h + 4)*width + w];
      val += G_10 * pixels[(h + 5)*width + w];
    }
  }
  return val;
}

template<int len, int width>
__device__ inline void load_into_reg(float * __restrict__ buf, const float * __restrict__ pixels, int h, int w) {
  #pragma unroll
  for (int i = 0; i < len; ++i) {
    buf[i] = pixels[h*width + w + i];
  }
}

template<bool sq>
__device__ inline void do_separable_conv_x(const float * __restrict__ pixels, float * __restrict__ opt, float * __restrict__ buf) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x * TILE_SIZE + 5;

  do {
    load_into_reg<RBUF_SIZE, SX>(buf, pixels, local_y, local_x - 5);

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      opt[local_y*CX + (local_x-5) + i] = do_conv<0, true, sq>(buf, 0, i + 5);
    }
    local_y += BY;
  } while (local_y < SY);
}

template<bool sq>
__device__ inline void do_separable_conv_y(const float * __restrict__ pixels, float * __restrict__ output) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 5;
  int local_x = block.thread_index().x * TILE_SIZE;

  #pragma unroll
  for (int i = 0; i < TILE_SIZE; ++i) {
    output[i] = do_conv<CX, false, sq>(pixels, local_y, local_x + i);
  }
}

template<int len>
__device__ __forceinline__ void do_elementwise_mul_sub(float * out, const float * f1, const float * f2) {
  #pragma unroll
  for (int i = 0; i < len; ++i) {
    out[i] -= f1[i] * f2[i];
  }
}

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
)
{
  auto block = cg::this_thread_block();
  const int pix_y = block.group_index().y * BY + block.thread_index().y;
  const int pix_x = block.group_index().x * BX + block.thread_index().x * TILE_SIZE;
  const int pix_id = pix_y * W + pix_x;
  const int num_pix = H * W;
  const int i = block.group_index().z;
  const int batch = 0;

  // shared memory that will be used to load pixels temporarily
  __shared__ float sbuf1[SY*SX];
  __shared__ float sbuf2[SY*SX];
  __shared__ float sbuf3[CY*CX];
  float rbuf1[RBUF_SIZE];

  float mu1_arr[TILE_SIZE];
  float sigma1_sq_arr[TILE_SIZE];
  float mu2_arr[TILE_SIZE];
  float sigma2_sq_arr[TILE_SIZE];
  float sigma12_arr[TILE_SIZE];

  // load into shared
  load_into_shared(sbuf1, img1, CH, H, W, i);
  block.sync();

  load_into_shared(sbuf2, img2, CH, H, W, i);

  // calculate mu1
  do_separable_conv_x<false>(sbuf1, sbuf3, rbuf1);
  block.sync();
  do_separable_conv_y<false>(sbuf3, mu1_arr);
  block.sync();

  // calculate sigma1_sq
  do_separable_conv_x<true>(sbuf1, sbuf3, rbuf1);
  block.sync();
  do_separable_conv_y<false>(sbuf3, sigma1_sq_arr);
  // block.sync();
  do_elementwise_mul_sub<TILE_SIZE>(sigma1_sq_arr, mu1_arr, mu1_arr);

  // calculate mu2
  // block.sync();
  do_separable_conv_x<false>(sbuf2, sbuf3, rbuf1);
  block.sync();
  do_separable_conv_y<false>(sbuf3, mu2_arr);
  // block.sync();

  // calculate sigma2_sq
  do_separable_conv_x<true>(sbuf2, sbuf3, rbuf1);
  block.sync();
  do_separable_conv_y<false>(sbuf3, sigma2_sq_arr);
  do_elementwise_mul_sub<TILE_SIZE>(sigma2_sq_arr, mu2_arr, mu2_arr);
  // block.sync();

  // calculate sigma12
  multiply_shared_mem(sbuf1, sbuf2);
  block.sync();
  do_separable_conv_x<false>(sbuf1, sbuf3, rbuf1);
  block.sync();
  do_separable_conv_y<false>(sbuf3, sigma12_arr);
  do_elementwise_mul_sub<TILE_SIZE>(sigma12_arr, mu1_arr, mu2_arr);
  block.sync();

  // calculate SSIM
  #pragma unroll
  for (int ii = 0; ii < TILE_SIZE; ++ii) {
    const float mu1 = mu1_arr[ii];
    const float mu2 = mu2_arr[ii];
    const float sigma1_sq = sigma1_sq_arr[ii];
    const float sigma2_sq = sigma2_sq_arr[ii];
    const float sigma12 = sigma12_arr[ii];

    float mu1_sq = mu1 * mu1;
    float mu2_sq = mu2 * mu2;
    float mu1_mu2 = mu1 * mu2;
    float C = (2.0f * mu1_mu2 + C1);
    float D = (2.0f * sigma12 + C2);
    float A = (mu1_sq + mu2_sq + C1);
    float B = (sigma1_sq + sigma2_sq + C2);
    float m = (C * D) / (A * B);
    if (pix_x + ii < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id + ii;
      ssim_map[global_idx] = m;

      if (dm_dmu1) {
        dm_dmu1[global_idx] = (
          (mu2 * 2.0f * D) / (A * B)
          -(mu2 * 2.0f * C) / (A * B)
          -(mu1 * 2.0f * C * D) / ( A * A * B)
          +(mu1 * 2.0f * C * D) / (A * B * B)
        );
        dm_dsigma1_sq[global_idx] = ((-C * D) / (A * B * B));
        dm_dsigma12[global_idx] = ((2 * C) / (A * B));
      }
    }
  }
}

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
  dim3 block(WARP_SIZE, WARP_NUM, 1);

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
