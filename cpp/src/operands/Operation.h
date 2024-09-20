#include <memory>
#include <../tensors/Tensor.h>
#include "AnyTensor.h"

#ifndef GEODIO_OPERATION_H
#define GEODIO_OPERATION_H

namespace dio {

using tensor = tensor_ptr;

using ForwardFunc = std::function<tensor(
    const std::vector<tensor>& inputs)>;

using BackwardFunc = std::function<std::vector<tensor>(
    const std::vector<tensor>& inputs,
    const tensor& upstream_gradient,
    const tensor& forward_output)>;

struct Operation {
    ForwardFunc forward;
    BackwardFunc backward;
};

} // namespace dio


#endif //GEODIO_OPERATION_H
