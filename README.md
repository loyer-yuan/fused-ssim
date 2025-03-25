## Performance Recording

Picture Size: [1, 3, 2160, 3840]

Platform: A100-80GB

| Version | Duration [ms] | Speedup | Time Decrese Percentage |
| :-----: | :-----------: | :-----: | :---------------------: |
|  Naive  |     2.40      |    0    |           0%            |
|   V1    |     2.26      |  1.062  |         -5.87%          |
|   V2    |     2.56      |  0.938  |         +6.58%          |
|  V3-0   |     2.71      |  0.886  |         +12.79%         |
|  V3-1   |     2.35      |  1.021  |         -2.32%          |
|   V3    |     2.01      |  1.194  |         -16.47%         |
|   V4    |     1.00      |   2.4   |         -58.26%         |



## V1: Remove Redundancy

- **Roughly Remove Redundant Operation** 删除冗余的赋值和同步操作
- Fix the bug of accessing out-of-boundary elements in `do_separable_conv_x` and `do_separable_conv_y` function

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

- 性能相比于V1下降了13%。
- 虽然warp内的各种stall有所改善，但IPC下降了，指令数增加了，特别是FFMA指令也增加很多，总时间也增加了。分析原因，有可能是因为分块的形状改变（拉长），导致Halo区增大，总体的冗余计算增加，增加了额外的计算。所以，增加指令并行度的带来的收益较小，需要结合Y方向的滑窗去做联合优化。

卷积运算公式

One block calculate (BX, BY) datas: 
$$
\frac{3 \cdot 5 \cdot (BX \cdot (BY+10) \cdot 11+BX \cdot BY \cdot 11)}{32} \rightarrow \\ 
\frac{165BX \cdot (BY+5)}{16}(one\ block\ conv.\ FFMA\ inst.)
$$
Block number:
$$
numBlock=\left \lceil \frac{X}{BX} \right \rceil \cdot \left \lceil \frac{Y}{BY} \right \rceil
$$
All calculations:
$$
cals = \left \lceil \frac{X}{BX} \right \rceil \cdot \left \lceil \frac{Y}{BY} \right \rceil \cdot \frac{165BX \cdot (BY+5)}{16}
$$
由上面的式子不难看出，在当前的计算模式下，BY越小，冗余计算越多。

## V3-0: Soft Pipeline but local memory

在Y维度上进行pipeline

- 性能相比于V2继续下降了5.82%。
- 虽然在Y轴上扩大了计算的幅度，按照V2的公式，减少了冗余计算，但因为展开了很多代码和增加了关键函数的计算量，导致代码体积膨胀，超过了指令的cache。因为no instruction的每个warp平均停顿时间甚至超过了V2上因为等待数据导致的停顿时间。
- 同时，因为在取数时写了一些分支，导致生成的代码中有很多分支跳转，使得no instruction的情况更加严重。

## V3-1: Soft Pipeline No local memory

相比于V3-0，编程上不使用数组的方式，手动编写寄存器的复用和流水。

- 性能相比于V3-0提升了13.39%，相比于V2提升了8.35%，相比于V1还是慢了3.78%。
- 通过手动控制软流水，warp stall的情况大幅度改善。stall no instruction的情况下降了55%，来到了3.94 cycles per instruction ，这是最主要的stall的原因。其他stall的原因也减少了很多。这也提升了SM Througput。
- 通过手写软流水，也消除了local memory，让DRAM Throughput回到了正常的水平。

## V3: Soft Pipeline (final version)

相比于V3-1，主要改变的是取数时候的条件判断。该条件判断是用来防止非法的内存访问。但使用__pipeline_memcpy_async中fillzero的参数，使得这条指令在访问非法内存的时候，填充0到shared memory中，但内存的访问还是非法。

- 性能相比于V3-1提升了14.49%，相比于V1提升了11.25%。
- 通过简单的减少分支，能够弥补指令数量过多带来的问题，warp stall的情况继续改善，整体warp cycles per issued instruction相比于V3-1提升了11.36%，主要stall原因还是stall no instruction，但也相比于V3-1减少了20.65%，来到了3.13。因此，带来的好处是SM Throughput, Memory Throughput, L1 Cache Throughput, DRAM Throughput，差不多都提升了15～16%。

## V4-1: Advanced Soft Pipeline but unsafe memory access

相比于V3，主要改变了在y维上计算的方式，每个寄存器存一个部分和，每个部分和做计算后会存到下一个寄存器中，这样让数据在寄存器间流动起来，算法具体可见代码。这样能够避免在循环内展开太多计算，避免了指令膨胀，使得核心循环能够在instruction cache内完成。

- 性能相比于v3，提升了50.03%。
- 虽然还是一样的软流水的思路，但是换了一种实现的方式，规避了v3软流水实现方式的缺点，拿到了真正的收益。这里也有一部分原因是因为NVIDIA SASS指令的FFMA 是有包含4个操作数，这样既能做乘加融合，也能加结果返回到不同的寄存器，实现了move指令的功能，隐式地节省了很大的move指令开销。

## V4: Advanced and Safe Soft Pipeline

相比于V4-1，在请求global memory访存的时候多了条件判断，而不是用fill zero的方式去做，这样开销会增大，但是不会出现illegal memory access。

- 性能相比于V4-1下降，时间多增加了5.56%。
- 增加了条件分支。

----
原来的README.md更新为README-old.md
