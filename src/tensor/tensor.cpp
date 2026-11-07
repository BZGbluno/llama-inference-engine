#include "tensor/tensor.h"

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape)) {
        data_ = new float[numel()];
    }

Tensor::~Tensor() {
    delete[] data_;
}
size_t Tensor::numel() const {
    size_t n = 1;
    for(size_t d: shape_) {
        n *= d;
    }
    return n;
}






