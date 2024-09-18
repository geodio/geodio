#include <memory>
#include <../tensors/Tensor.h>

#ifndef GEODIO_OPERATION_H
#define GEODIO_OPERATION_H

namespace dio {


template<typename T>
using ForwardFunc = std::function<std::shared_ptr<Tensor<T>>(
    const std::vector<std::shared_ptr<Tensor<T>>>& inputs)>;

template<typename T>
using BackwardFunc = std::function<std::vector<std::shared_ptr<Tensor<T>>>(
    const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
    const std::shared_ptr<Tensor<T>>& upstream_gradient,
    const std::shared_ptr<Tensor<T>>& forward_output)>;

template<typename T>
struct Operation {
    ForwardFunc<T> forward;
    BackwardFunc<T> backward;
};

} // namespace dio


#endif //GEODIO_OPERATION_H
