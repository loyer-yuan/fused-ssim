## V1: Remove Redundancy

**Roughly Remove Redundant Operation** 删除冗余的赋值和同步操作

Before Opt.:
```cpp
// load into shared
load_into_shared(buf1, img1, CH, H, W, i);
block.sync();

// calculate mu1
flush_conv_scratch(buf3);
block.sync();
do_separable_conv_x(buf1, buf3, H, W);
block.sync();
float mu1 = do_separable_conv_y(buf3, H, W);
block.sync();

// calculate sigma1_sq
flush_conv_scratch(buf3);
block.sync();
do_separable_conv_x(buf1, buf3, H, W, true);
block.sync();
float sigma1_sq = do_separable_conv_y(buf3, H, W) - mu1 * mu1;
block.sync();

// calculate mu2
load_into_shared(buf2, img2, CH, H, W, i);
block.sync();
flush_conv_scratch(buf3);
block.sync();
do_separable_conv_x(buf2, buf3, H, W);
block.sync();
float mu2 = do_separable_conv_y(buf3, H, W);
block.sync();

// calculate sigma2_sq
flush_conv_scratch(buf3);
block.sync();
do_separable_conv_x(buf2, buf3, H, W, true);
block.sync();
float sigma2_sq = do_separable_conv_y(buf3, H, W) - mu2 * mu2;
block.sync();

// calculate sigma12
multiply_shared_mem(buf1, buf2);
block.sync();
flush_conv_scratch(buf3);
block.sync();
do_separable_conv_x(buf1, buf3, H, W);
block.sync();
float sigma12 = do_separable_conv_y(buf3, H, W) - mu1 * mu2;
block.sync();
```

After Opt.:
```cpp
// load into shared
load_into_shared(buf1, img1, CH, H, W, i);
block.sync();

// calculate mu1
do_separable_conv_x(buf1, buf3, H, W);
block.sync();
float mu1 = do_separable_conv_y(buf3, H, W);
block.sync();

// calculate sigma1_sq
do_separable_conv_x(buf1, buf3, H, W, true);
block.sync();
float sigma1_sq = do_separable_conv_y(buf3, H, W) - mu1 * mu1;

// calculate mu2
load_into_shared(buf2, img2, CH, H, W, i);
block.sync();
do_separable_conv_x(buf2, buf3, H, W);
block.sync();
float mu2 = do_separable_conv_y(buf3, H, W);
block.sync();

// calculate sigma2_sq
do_separable_conv_x(buf2, buf3, H, W, true);
block.sync();
float sigma2_sq = do_separable_conv_y(buf3, H, W) - mu2 * mu2;
block.sync();

// calculate sigma12
multiply_shared_mem(buf1, buf2);
block.sync();
do_separable_conv_x(buf1, buf3, H, W);
block.sync();
float sigma12 = do_separable_conv_y(buf3, H, W) - mu1 * mu2;
block.sync();
```

## V2: Tiling

在X维度上进行Tiling

- 性能下降了13%，虽然warp内的各种stall有所改善，但IPC下降了，指令数增加了，特别是FFMA指令也增加很多，总时间也增加了。分析原因，有可能是因为分块的形状改变（拉长），导致Halo区增大，总体的冗余计算增加，增加了额外的计算。所以，增加指令并行度的带来的收益较小，需要结合Y方向的滑窗去做联合优化。