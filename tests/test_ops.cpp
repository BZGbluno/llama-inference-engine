#include <gtest/gtest.h>
#include "tensor/ops.h"
#include <cmath>
#include <limits>

// helper: copy a list of values into a tensor
// static makes the fill function definition only part of this file to 
// avoid multiple definition error with linker
static void fill(Tensor& t, const std::vector<float>& values) {
    for (size_t i = 0; i < t.numel(); i++) {
        t.data()[i] = values[i];
    }
}

TEST(Ops, Add) {
    Tensor a({2, 3}), b({2, 3}), out({2, 3});
    fill(a, {1, 2, 3, 4, 5, 6});
    fill(b, {10, 20, 30, 40, 50, 60});

    add(a, b, out);

    std::vector<float> ans = {11, 22, 33, 44, 55, 66};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_EQ(out.data()[i], ans[i]);
    }
}

TEST(Ops, Scale) {
    Tensor a({2, 3}), b({2, 3}), out({2, 3});
    fill(a, {1, 2, 3, 4, 5, 6});
    fill(b, {2, 2, 2, 0, -1, 0.5f});

    scale(a, b, out);

    std::vector<float> ans = {2, 4, 6, 0, -5, 3};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_EQ(out.data()[i], ans[i]);
    }
}

TEST(Ops, AddShapeMismatchThrows) {
    Tensor a({2, 3}), b({3, 2}), out({2, 3});
    EXPECT_THROW(add(a, b, out), std::invalid_argument);
}

TEST(Ops, ScaleShapeMismatchThrows) {
    Tensor a({2, 3}), b({2, 3}), out({6});
    EXPECT_THROW(scale(a, b, out), std::invalid_argument);
}

TEST(Ops, Matmul2D) {
    // [2, 3] @ [3, 2]
    Tensor a({2, 3}), b({3, 2}), out({2, 2});
    fill(a, {1, 2, 3, 4, 5, 6});
    fill(b, {1, 0, 0, 1, 1, 1});

    matmul(a, b, out);

    std::vector<float> ans = {4, 5, 10, 11};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_EQ(out.data()[i], ans[i]);
    }
}

TEST(Ops, MatmulBatched) {
    // 3 separate 2x2 @ 2x2, each A[b] pairs with its own B[b]
    Tensor a({3, 2, 2}), b({3, 2, 2}), out({3, 2, 2});
    fill(a, {1, 2, 3, 4,   5, 0, 2, 7,   9, 1, 0, 3});
    fill(b, {1, 0, 0, 1,   2, 1, 1, 2,   1, 1, 1, 1});

    matmul(a, b, out);

    std::vector<float> ans = {1, 2, 3, 4,   10, 5, 11, 16,   10, 10, 3, 3};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_EQ(out.data()[i], ans[i]);
    }
}

TEST(Ops, MatmulSharedB) {
    // [2, 2, 2] @ [2, 2], same W reused for both batches
    Tensor a({2, 2, 2}), w({2, 2}), out({2, 2, 2});
    fill(a, {1, 2, 3, 4,   5, 6, 7, 8});
    fill(w, {1, 0, 0, 2});

    matmul(a, w, out);

    std::vector<float> ans = {1, 4, 3, 8,   5, 12, 7, 16};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_EQ(out.data()[i], ans[i]);
    }
}

TEST(Ops, MatmulInnerDimMismatchThrows) {
    Tensor a({2, 3}), b({4, 2}), out({2, 2});
    EXPECT_THROW(matmul(a, b, out), std::invalid_argument);
}

TEST(Ops, MatmulOutShapeMismatchThrows) {
    // only one out dim wrong, the old `and` check let this through
    Tensor a({2, 3}), b({3, 2}), out({2, 5});
    EXPECT_THROW(matmul(a, b, out), std::invalid_argument);
}

TEST(Ops, MatmulBatchDimMismatchThrows) {
    Tensor a({3, 2, 4}), b({5, 4, 5}), out({3, 2, 5});
    EXPECT_THROW(matmul(a, b, out), std::invalid_argument);
}

TEST(Ops, RMSNorm1D) {
    // [1, 2, 3, 4]: mean of squares = 7.5, rms = sqrt(7.5) ~ 2.7386
    Tensor x({4}), w({4}), out({4});
    fill(x, {1, 2, 3, 4});
    fill(w, {1, 1, 1, 1});

    rmsnorm(x, w, out, 1e-5f);

    std::vector<float> ans = {0.365148f, 0.730296f, 1.095444f, 1.460593f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], ans[i], 1e-4);
    }
}

TEST(Ops, RMSNormWeight) {
    // same normalization as above, then each position scaled by w
    Tensor x({4}), w({4}), out({4});
    fill(x, {1, 2, 3, 4});
    fill(w, {1, 1, 2, 0.5f});

    rmsnorm(x, w, out, 1e-5f);

    std::vector<float> ans = {0.365148f, 0.730296f, 2.190889f, 0.730296f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], ans[i], 1e-4);
    }
}

TEST(Ops, RMSNormRowsIndependent) {
    // 2 tokens, each row gets its own rms and the same w is shared
    // row 0 rms ~ 3.54, row 1 rms ~ 0.1
    Tensor x({2, 2}), w({2}), out({2, 2});
    fill(x, {3, 4,   0.1f, 0.1f});
    fill(w, {1, 1});

    rmsnorm(x, w, out, 1e-5f);

    std::vector<float> ans = {0.848528f, 1.131371f,   0.9995f, 0.9995f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], ans[i], 1e-4);
    }
}

TEST(Ops, RMSNormWeightSizeMismatchThrows) {
    Tensor x({2, 4}), w({3}), out({2, 4});
    EXPECT_THROW(rmsnorm(x, w, out, 1e-5f), std::invalid_argument);
}

TEST(Ops, RMSNormOutShapeMismatchThrows) {
    Tensor x({2, 4}), w({4}), out({2, 3});
    EXPECT_THROW(rmsnorm(x, w, out, 1e-5f), std::invalid_argument);
}

TEST(Ops, Softmax) {
    // exp: 2.718, 7.389, 20.086  sum = 30.193
    Tensor x({3}), out({3});
    fill(x, {1, 2, 3});

    softmax(x, out);

    std::vector<float> ans = {0.090031f, 0.244728f, 0.665241f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], ans[i], 1e-5);
    }
}

TEST(Ops, SoftmaxLargeValuesNoOverflow) {
    // exp(1000) overflows a float, subtracting the max keeps it finite
    // only differences matter so this matches softmax([1, 2, 3])
    Tensor x({3}), out({3});
    fill(x, {1000, 1001, 1002});

    softmax(x, out);

    std::vector<float> ans = {0.090031f, 0.244728f, 0.665241f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_FALSE(std::isnan(out.data()[i]));
        EXPECT_NEAR(out.data()[i], ans[i], 1e-5);
    }
}

TEST(Ops, SoftmaxAllEqual) {
    // equal inputs split the probability evenly
    Tensor x({4}), out({4});
    fill(x, {5, 5, 5, 5});

    softmax(x, out);

    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], 0.25f, 1e-6);
    }
}

TEST(Ops, SoftmaxRowsIndependent) {
    // each row is its own distribution and sums to 1
    Tensor x({2, 3}), out({2, 3});
    fill(x, {1, 2, 3,   0, 0, 0});

    softmax(x, out);

    std::vector<float> ans = {0.090031f, 0.244728f, 0.665241f,   0.333333f, 0.333333f, 0.333333f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], ans[i], 1e-5);
    }

    for (size_t r = 0; r < 2; r++) {
        float sum = 0.0f;
        for (size_t i = 0; i < 3; i++) {
            sum += out.data()[r * 3 + i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-6);
    }
}

TEST(Ops, SoftmaxMaskedValueIsZero) {
    // causal mask sets future positions to -inf, exp(-inf) = 0
    // the rest is softmax([1, 3])
    const float negInf = -std::numeric_limits<float>::infinity();
    Tensor x({3}), out({3});
    fill(x, {1, negInf, 3});

    softmax(x, out);

    EXPECT_NEAR(out.data()[0], 0.119203f, 1e-5);
    EXPECT_EQ(out.data()[1], 0.0f);
    EXPECT_NEAR(out.data()[2], 0.880797f, 1e-5);
}

TEST(Ops, SoftmaxShapeMismatchThrows) {
    Tensor x({2, 3}), out({3, 2});
    EXPECT_THROW(softmax(x, out), std::invalid_argument);
}

TEST(Ops, SiLU) {
    // silu(x) = x / (1 + e^-x)
    // negatives pushed toward 0, silu(0) = 0, positives mostly pass through
    Tensor x({5}), out({5});
    fill(x, {-2, -1, 0, 1, 2});

    SiLU(x, out);

    std::vector<float> ans = {-0.238406f, -0.268941f, 0.0f, 0.731059f, 1.761594f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], ans[i], 1e-5);
    }
}

TEST(Ops, SiLUExtremesNoNaN) {
    // x = -100: e^100 overflows to inf, -100 / inf = 0
    // x = 100:  e^-100 ~ 0, 100 / 1 = 100
    Tensor x({2}), out({2});
    fill(x, {-100, 100});

    SiLU(x, out);

    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_FALSE(std::isnan(out.data()[i]));
    }
    EXPECT_NEAR(out.data()[0], 0.0f, 1e-5);
    EXPECT_NEAR(out.data()[1], 100.0f, 1e-4);
}

TEST(Ops, SiLUAnyShape) {
    // element-wise so a [2, 3] tensor gives the same value per position
    Tensor x({2, 3}), out({2, 3});
    fill(x, {-2, -1, 0,   1, 2, -0.5f});

    SiLU(x, out);

    std::vector<float> ans = {-0.238406f, -0.268941f, 0.0f,   0.731059f, 1.761594f, -0.188770f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_NEAR(out.data()[i], ans[i], 1e-5);
    }
}

TEST(Ops, SiLUShapeMismatchThrows) {
    Tensor x({2, 3}), out({6});
    EXPECT_THROW(SiLU(x, out), std::invalid_argument);
}

TEST(Ops, Embedding) {
    // table is [vocab=4, D=3], each id picks out its row
    Tensor table({4, 3}), out({3, 3});
    fill(table, {0.1f, 0.2f, 0.3f,
                 1.0f, 1.1f, 1.2f,
                 2.0f, 2.1f, 2.2f,
                 3.0f, 3.1f, 3.2f});

    embedding({2, 0, 2}, table, out);

    // rows 2, 0, 2 copied exactly so EXPECT_EQ is fine (no arithmetic)
    std::vector<float> ans = {2.0f, 2.1f, 2.2f,
                              0.1f, 0.2f, 0.3f,
                              2.0f, 2.1f, 2.2f};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_EQ(out.data()[i], ans[i]);
    }
}

TEST(Ops, EmbeddingBatched) {
    // 2 prompts of 2 tokens, ids passed back to back, out is [batch, seq, D]
    Tensor table({3, 2}), out({2, 2, 2});
    fill(table, {0, 1,   10, 11,   20, 21});

    embedding({1, 2,   0, 1}, table, out);

    std::vector<float> ans = {10, 11,  20, 21,    0, 1,  10, 11};
    for (size_t i = 0; i < out.numel(); i++) {
        EXPECT_EQ(out.data()[i], ans[i]);
    }
}

TEST(Ops, EmbeddingIdTooBigThrows) {
    // vocab = 4 so valid ids are 0..3, id 4 would read one row past the table
    Tensor table({4, 3}), out({1, 3});
    EXPECT_THROW(embedding({4}, table, out), std::out_of_range);
}

TEST(Ops, EmbeddingNegativeIdThrows) {
    Tensor table({4, 3}), out({1, 3});
    EXPECT_THROW(embedding({-1}, table, out), std::out_of_range);
}

TEST(Ops, EmbeddingWrongDThrows) {
    // out rows are length 2 but table rows are length 3
    Tensor table({4, 3}), out({2, 2});
    EXPECT_THROW(embedding({0, 1}, table, out), std::invalid_argument);
}

TEST(Ops, EmbeddingWrongRowCountThrows) {
    // 3 ids but out only has room for 2 rows
    Tensor table({4, 3}), out({2, 3});
    EXPECT_THROW(embedding({0, 1, 2}, table, out), std::invalid_argument);
}

TEST(Ops, EmbeddingTableNot2DThrows) {
    Tensor table({12}), out({1, 3});
    EXPECT_THROW(embedding({0}, table, out), std::invalid_argument);
}
