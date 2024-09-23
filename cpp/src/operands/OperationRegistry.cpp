#include "OperationRegistry.h"

namespace dio {
OperationRegistry& OperationRegistry::get_instance() {
    static OperationRegistry instance;
    return instance;
}

void OperationRegistry::register_operation(OperandType type, const Operation& operation) {
    operations_[type] = operation;
}

Operation OperationRegistry::get_operation(OperandType type) const {
    auto it = operations_.find(type);
    if (it != operations_.end()) {
        return it->second;
    } else {
        throw std::runtime_error(
            std::string("Operation not registered for OperandType in OperationRegistry"));
    }
}
} // namespace dio
