#include "CPUBackend.h"
#include "Backend.h"
#include <memory>

#ifndef GEODIO_BACKENDMANAGER_H
#define GEODIO_BACKENDMANAGER_H

namespace dio{
    template<typename T>
    class BackendManager {
    public:
        static std::shared_ptr<Backend<T>> get_backend() {
            static std::shared_ptr<Backend<T>> instance = std::make_shared<CPUBackend<T>>();
            return instance;
        }
    };
} // namespace dio




#endif //GEODIO_BACKENDMANAGER_H
