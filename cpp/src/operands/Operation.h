
#include <memory>
#include <../tensors/Tensor.h>

#ifndef GEODIO_OPERATION_H
#define GEODIO_OPERATION_H

namespace dio {
    typedef std::function<std::shared_ptr<Tensor<float>>(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs)> ForwardFunc;

    typedef std::function<std::vector<std::shared_ptr<Tensor<float>>>(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        const std::shared_ptr<Tensor<float>>& upstream_gradient)> BackwardFunc;

    struct Operation {
        ForwardFunc forward{};
        BackwardFunc backward{};
    };
} // namespace dio


#endif //GEODIO_OPERATION_H
