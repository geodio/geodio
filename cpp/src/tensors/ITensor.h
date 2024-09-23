//
// Created by zwartengaten on 9/20/24.
//

#ifndef GEODIO_ITENSOR_H
#define GEODIO_ITENSOR_H

#include <typeinfo>

namespace dio {

    class ITensor {
    public:
        virtual ~ITensor() = default;
        [[nodiscard]] virtual const std::type_info& type_info() const = 0;

    };

} // dio

#endif //GEODIO_ITENSOR_H
