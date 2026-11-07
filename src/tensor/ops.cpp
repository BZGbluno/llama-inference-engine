#include "tensor/ops.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

void add(const Tensor& a, const Tensor& b, Tensor& out){

    if (a.shape() != b.shape() || a.shape() != out.shape())
        throw std::invalid_argument("add: shape mismatch");

    const float* dataA = a.data();
    const float* dataB = b.data();
    float * dataOut  = out.data();

    for(size_t i = 0; i < a.numel(); i++) {
        dataOut[i] = dataA[i] + dataB[i];
    }
}

void scale(const Tensor& a, const Tensor&b, Tensor& out) {
    if (a.shape() != b.shape() || a.shape() != out.shape())
        throw std::invalid_argument("scale: shape mismatch");

    const float* dataA = a.data();
    const float* dataB = b.data();
    float * dataOut  = out.data();

    for(size_t i = 0; i < a.numel(); i++) {
        dataOut[i] = dataA[i] * dataB[i];
    }
}


void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    const auto& as = a.shape();
    const auto& bs = b.shape();
    const auto& os = out.shape();

    if (as.size() < 2 || bs.size() < 2 || as.size() != os.size())
        throw std::invalid_argument("matmul: rank mismatch");

    // b is either fully batched like a, or a single 2D matrix shared by all
    bool sharedB = (bs.size() == 2);
    if (!sharedB && bs.size() != as.size())
        throw std::invalid_argument("matmul: rank mismatch");

    size_t r = as.size();
    size_t M = as[r - 2];
    size_t K = as[r - 1];
    size_t N = bs[bs.size() - 1];

    // inner dims must match and out must be [..., M, N]
    if (bs[bs.size() - 2] != K || os[r - 2] != M || os[r - 1] != N)
        throw std::invalid_argument("matmul: shape mismatch");

    size_t batch = 1;
    for (size_t d = 0; d < r - 2; d++) {
        if (os[d] != as[d] || (!sharedB && bs[d] != as[d]))
            throw std::invalid_argument("matmul: batch dim mismatch");
        batch *= as[d];
    }

    for (size_t bi = 0; bi < batch; bi++) {

        // jump to the bi-th matrix in each buffer, each skip is one
        const float* A = a.data() + bi * M * K;
        const float* B = b.data() + (sharedB ? 0 : bi * K * N);
        float* C = out.data() + bi * M * N;

        std::fill(C, C + M * N, 0.0f);

        // start of the 2D Mat Mul
        // inner loop walks B and C row by row so memory access is contiguous
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {

                // A[i][k]
                float aik = A[i * K + k];

                for (size_t j = 0; j < N; j++) {
                    // C[i][j] += A[i][k] * B[k][j]
                    C[i * N + j] += aik * B[k * N + j];
                }
            }
        }
    }
}

// root mean square of one row of length D
static float rms(const float* x, size_t D, float eps) {
    float sumSq = 0.0f;

    for(size_t i = 0; i < D; i++) {
        sumSq += x[i] * x[i];
    }

    return std::sqrt(sumSq / D + eps);
}

// [seq, D]          a stack of token vectors, one row per token
// [batch, seq, D]   a stack of prompts, each a stack of token vectors
void rmsnorm(const Tensor& x, const Tensor& w, Tensor& out, float eps) {
    const auto& xs = x.shape();

    if (xs.empty() || w.shape().size() != 1 || w.shape()[0] != xs.back() || out.shape() != xs)
        throw std::invalid_argument("rmsnorm: shape mismatch");

    size_t D = xs.back();
    size_t rows = x.numel() / D;

    const float * wD = w.data();

    for(size_t r = 0; r < rows; r++) {

        // jump to row r, same idea as bi * M * K in matmul
        const float * xD = x.data() + r * D;
        float * o = out.data() + r * D;

        float rms_val = rms(xD, D, eps);

        for(size_t i = 0; i < D; i++) {
            o[i] = xD[i] / rms_val * wD[i];
        }
    }
}

void softmax(const Tensor& x, Tensor& out) {
    const auto& xs = x.shape();

    if (xs.empty() || out.shape() != xs)
        throw std::invalid_argument("softmax: shape mismatch");


    size_t D = xs.back();
    size_t rows = x.numel() / D;


    for(size_t r = 0; r < rows; r++) {

        const float* xD = x.data() + r * D;
        float * o = out.data() + r * D;
        float maxVal = xD[0];

        for(size_t i = 1; i < D; i++) {
            maxVal = std::max(maxVal, xD[i]);
        } 
        
        float sum = 0.0f;

        for(size_t i = 0; i < D; i++) {
            o[i] = std::exp(xD[i] - maxVal);
            sum += o[i];
        }

        for(size_t i = 0; i < D; i++) {
            o[i] /= sum;
        }
        
    }
}


void SiLU(const Tensor& x, Tensor& out) {
    
    if (out.shape() != x.shape()) {
        throw std::invalid_argument("silu: shape mismatch");
    }

    const float * xD =  x.data();
    float * o = out.data();

    for(size_t i = 0; i < x.numel(); i++) {
        
        o[i] = xD[i] / (1.0f + std::exp(-xD[i]));
    }
}

void embedding(const std::vector<int>& ids, const Tensor& table, Tensor& out) {
    const auto& ts = table.shape();

    if (ts.size() != 2) {
        throw std::invalid_argument("embedding: table must be [vocab, D]");
    }

    size_t rows = ts[0];
    size_t D = ts[1];

    if(out.shape().empty() || out.shape().back() != D || out.numel() != ids.size() * D) {
        throw std::invalid_argument("embedding: shape mismatch");
    }

    for(size_t i = 0; i < ids.size(); i++) {

        if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= rows) {
            throw std::out_of_range("embedding: token id out of range");
        }

        float * o = out.data() + i * D;
        const float * tableD = table.data() + ids[i] * D;

        for(size_t pos = 0; pos < D; pos++) {
            o[pos] = tableD[pos];
        }
        
    }
}