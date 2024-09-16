//
// Created by zwartengaten on 9/16/24.
//

#ifndef GEOPY_TENSORGPU_H
#define GEOPY_TENSORGPU_H

#include "Tensor.h"

namespace dio {

    template<typename T>
    class TensorGPU : public Tensor<T> {
    public:
        TensorGPU(std::initializer_list<T> list);
        ~TensorGPU();

        void to_gpu() override;
        void from_gpu() override;

    private:
        T* gpu_data_;  // Pointer to GPU memory
    };

} // dio

#endif //GEOPY_TENSORGPU_H
