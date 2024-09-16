//#include "Tensor_C_API.h"
//#include "Tensor.h"
//
//extern "C" {
//
//struct TensorHandle {
//    void* tensor_ptr;
//    char type_code;  // 'f' for float, 'd' for double, 'i' for int
//};
//
//TensorHandle* tensor_create_float(const float* data, const size_t* shape, size_t ndim, size_t size) {
//    auto* tensor = new dio::Tensor<float>(std::vector<float>(data, data + size), std::vector<size_t>(shape, shape + ndim));
//    auto* handle = new TensorHandle;
//    handle->tensor_ptr = tensor;
//    handle->type_code = 'f';
//    return handle;
//}
//
//void tensor_destroy(TensorHandle* handle) {
//    if (handle->type_code == 'f') {
//        delete static_cast<dio::Tensor<float>*>(handle->tensor_ptr);
//    }
//    // Handle other types...
//    delete handle;
//}
//
//TensorHandle* tensor_add(const TensorHandle* a, const TensorHandle* b) {
//    if (a->type_code == 'f' && b->type_code == 'f') {
//        auto tensor_a = static_cast<dio::Tensor<float>*>(a->tensor_ptr);
//        auto tensor_b = static_cast<dio::Tensor<float>*>(b->tensor_ptr);
//        auto result = new dio::Tensor<float>(*tensor_a + *tensor_b);
//        auto* handle = new TensorHandle;
//        handle->tensor_ptr = result;
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
