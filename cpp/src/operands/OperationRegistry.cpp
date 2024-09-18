#include "OperationRegistry.h"

namespace dio {
template<typename T>
OperationRegistry<T>& OperationRegistry<T>::get_instance() {
    static OperationRegistry<T> instance;
    return instance;
}

template<typename T>
void OperationRegistry<T>::register_operation(OperandType type, const Operation<T>& operation) {
    operations_[type] = operation;
}

template<typename T>
Operation<T> OperationRegistry<T>::get_operation(OperandType type) const {
    auto it = operations_.find(type);
    if (it != operations_.end()) {
        return it->second;
    } else {
        throw std::runtime_error(
            std::string("Operation not registered for OperandType in OperationRegistry<") +
            typeid(T).name() + ">");
    }
}

template class OperationRegistry<int>;
template class OperationRegistry<float>;
template class OperationRegistry<double>;
} // namespace dio
