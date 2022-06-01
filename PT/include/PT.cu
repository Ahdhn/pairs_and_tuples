#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

template <typename... PartitionsT>
__device__ inline static void pair_example(
    std::pair<PartitionsT, int>... partitions)
{
    for (auto& p : {partitions...}) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("\n p.first= %d", p.first);
            printf("\n p.second= %d", p.second);
        }
    }
}


template <typename... PartitionsT>
__device__ inline static void tuple_example(
    std::tuple<PartitionsT, int>... partitions)
{
    for (auto& p : {partitions...}) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("\n p<0>= %d", std::get<0>(p));
            printf("\n p<1>= %d", std::get<1>(p));
        }
    }
}

__global__ void exec_kernel()
{
    int i = 1;
    int j = 2;

    pair_example(std::make_pair(i, j), std::make_pair(j, i));

    tuple_example(std::make_tuple(i, j), std::make_tuple(j, i));
}

TEST(Test, exe)
{
    exec_kernel<<<1, 1>>>();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
