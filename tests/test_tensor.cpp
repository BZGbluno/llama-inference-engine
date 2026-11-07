#include <gtest/gtest.h>
#include "tensor/tensor.h"

TEST(Tensor, StoresShape) {
    Tensor t({2, 3});
    // 2u is the unsigned 2
    EXPECT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 3u);
}

TEST(Tensor, Numel) {
    EXPECT_EQ(Tensor({2, 3}).numel(), 6u);
    EXPECT_EQ(Tensor({4, 32, 128}).numel(), 16384u);
}

TEST(Tensor, LoadingData) {
    Tensor t({2,3});
    std::vector<float> ans = {0, 1, 2, 3, 4, 5};

    for (size_t i = 0; i < t.numel(); i++) {
        t.data()[i] = i;
    }

    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_EQ(t.data()[i], ans[i]);
    }
}
