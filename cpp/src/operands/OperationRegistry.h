#include "OperandType.h"
#include "Operation.h"
#ifndef GEODIO_OPERATIONREGISTRY_H
#define GEODIO_OPERATIONREGISTRY_H

namespace dio {

class OperationRegistry {
public:
    static OperationRegistry& get_instance();

    void register_operation(OperandType type, const Operation& operation);
    [[nodiscard]] Operation get_operation(OperandType type) const;

private:
    OperationRegistry() = default;
    std::unordered_map<OperandType, Operation> operations_{};
};
} // namespace dio

#endif //GEODIO_OPERATIONREGISTRY_H
