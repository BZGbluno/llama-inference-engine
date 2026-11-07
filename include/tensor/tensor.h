#pragma once

#include <vector>
class Tensor {
    public:
        Tensor(std::vector<size_t> shape);
        
        // avoids copy construction
        Tensor(const Tensor&) = delete;

        // avoids copy assignement like
        Tensor& operator=(const Tensor&) = delete;

        // destrutor
        ~Tensor();

        // const because you cant update the # of elements
        size_t numel() const;

        // & because we want reference of shape
        const std::vector<size_t>& shape() const {return shape_;}

        // exposes the pointer and let the caller write to it
        float * data() {return data_;}

        const float * data() const {return data_;}

    private:
        std::vector<size_t> shape_;
        // nullptr is the value a pointer holds when it points to nothing.
        float * data_ = nullptr;

};