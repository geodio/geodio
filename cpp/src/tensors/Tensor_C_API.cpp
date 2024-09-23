//#include "Tensor_C_API.h"
//#include "Tensor.h"
//
//extern "C" {
//
//struct TensorHandle {
//    void* a_tens;
//    char type_code;  // 'f' for float, 'd' for double, 'i' for int
//};
//
//TensorHandle* tensor_create_float(const float* data, const size_t* shape, size_t ndim, size_t size) {
//    auto* tensor = new dio::Tensor<float>(std::vector<float>(data, data + size), std::vector<size_t>(shape, shape + ndim));
//    auto* handle = new TensorHandle;
//    handle->a_tens = tensor;
//    handle->type_code = 'f';
//    return handle;
//}
//
//void tensor_destroy(TensorHandle* handle) {
//    if (handle->type_code == 'f') {
//        delete static_cast<dio::Tensor<float>*>(handle->a_tens);
//    }
//    // Handle other types...
//    delete handle;
//}
//
//TensorHandle* tensor_add(const TensorHandle* a, const TensorHandle* b) {
//    if (a->type_code == 'f' && b->type_code == 'f') {
//        auto tensor_a = static_cast<dio::Tensor<float>*>(a->a_tens);
//        auto tensor_b = static_cast<dio::Tensor<float>*>(b->a_tens);
//        auto result = new dio::Tensor<float>(*tensor_a + *tensor_b);
//        auto* handle = new TensorHandle;
//        handle->a_tens = result;
//        handle->type_code = 'f';
//        return handle;
//    }
//    // Handle other types and combinations...
//    return nullptr;
//}
//
//// Implement other functions...
//
//}
