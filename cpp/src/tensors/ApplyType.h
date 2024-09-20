#ifndef GEODIO_APPLYTYPE_H
#define GEODIO_APPLYTYPE_H

#include "Tensor.h"
#include <functional>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>

#endif //GEODIO_APPLYTYPE_H
namespace dio {
    enum class ApplyType {
    Add,
    Product,
    Divide,
    Subtract,
    Matmul,
    Custom
    };
}