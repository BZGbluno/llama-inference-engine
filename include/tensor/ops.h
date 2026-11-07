
#pragma once
#include "tensor/tensor.h"
#include <vector>

void add(const Tensor& a, const Tensor&b, Tensor& out);
void scale(const Tensor& a, const Tensor&b, Tensor& out);
void matmul(const Tensor& a, const Tensor&b, Tensor& out);
void rmsnorm(const Tensor& x, const Tensor& w, Tensor& out, float eps);
void softmax(const Tensor& x, Tensor& out);
void SiLU(const Tensor& x, Tensor& out);
void embedding(const std::vector<int>& ids, const Tensor& table, Tensor& out);