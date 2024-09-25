// AnyTensor.h

#ifndef GEODIO_ANYTENSOR_H
#define GEODIO_ANYTENSOR_H

#include <typeinfo>
#include <typeindex>
#include <stdexcept>
#include <functional>
#include <memory>
#include "Tensor.h"
#include "ApplyType.h"
#include "ITensor.h"

namespace dio {

    static const std::type_index TYPE_FLOAT = typeid(float);

    static const std::type_index TYPE_DOUBLE = typeid(double);

    static const std::type_index TYPE_INT = typeid(int);

    class AnyTensor {
public:
    AnyTensor() = default;

    // Constructor that takes ownership of the tensor pointer
    template<typename T>
    explicit AnyTensor(std::shared_ptr<Tensor<T>> tensor)
        : tensor_ptr_(std::move(tensor)), type_index_(typeid(T)) {}

    // Constructor from vector and shape
    template<typename T>
    explicit AnyTensor(const std::vector<T> &data, const std::vector<size_t> &shape)
    :tensor_ptr_(std::make_shared<dio::Tensor<T>>(data, shape)), type_index_(typeid(T)) {}

    // Constructor for scalar tensors
    template<typename T>
    explicit AnyTensor(const T &value):tensor_ptr_(std::make_shared<dio::Tensor<T>>(value)), type_index_(typeid(T)) {}

    // Copy constructor
    AnyTensor(const AnyTensor& other)
        : tensor_ptr_(other.tensor_ptr_), type_index_(other.type_index_) {}

    // Move constructor
    AnyTensor(AnyTensor&& other) noexcept
        : tensor_ptr_(std::move(other.tensor_ptr_)), type_index_(other.type_index_) {}

    // Assignment operator
    AnyTensor& operator=(const AnyTensor& other) {
        if (this != &other) {
            tensor_ptr_ = other.tensor_ptr_;
        }
        return *this;
    }

    // Move assignment operator
    AnyTensor& operator=(AnyTensor&& other) noexcept {
        if (this != &other) {
            tensor_ptr_ = std::move(other.tensor_ptr_);
        }
        return *this;
    }

    // Destructor
    ~AnyTensor() = default;

    // Get type info
    [[nodiscard]] const std::type_info& type() const {
        if (!tensor_ptr_) {
            throw std::runtime_error("AnyTensor is empty");
        }
        return tensor_ptr_->type_info();
    }

    // Check if empty
    [[nodiscard]] bool empty() const {
        return !tensor_ptr_;
    }

    // Retrieve the tensor as its original type
   template<typename T>
    [[nodiscard]] Tensor<T>& get() const {
        if (!tensor_ptr_) {
            throw std::runtime_error("Error: Attempted to access an empty AnyTensor. The tensor is uninitialized.");
        }
        if (type_index_ != typeid(T)) {
            throw std::runtime_error("Error: Type mismatch in AnyTensor. Expected type '" +
                std::string(typeid(T).name()) + "', but actual stored type is '" +
                std::string(type_index_.name()) + "'.");
        }
        auto casted_ptr = std::static_pointer_cast<Tensor<T>>(tensor_ptr_);
        if (!casted_ptr) {
            throw std::bad_cast();
        }
        return *casted_ptr;
    }

    // Check if the stored type is T
    template<typename T>
    [[nodiscard]] bool is() const {
        if (!tensor_ptr_) {
            return false;
        }
        return type_index_ == typeid(T);
    }

    // Get the stored type index
    [[nodiscard]] std::type_index type_index() const {
        return type_index_;
    }

    // Apply binary operation with another AnyTensor
    [[nodiscard]] AnyTensor apply(const AnyTensor& other, ApplyType apply_type,
                                  void* custom_op = nullptr) const;

    // Methods for basic operations
    [[nodiscard]] AnyTensor add(const AnyTensor& other) const {
        return apply(other, ApplyType::Add);
    }

    [[nodiscard]] AnyTensor subtract(const AnyTensor& other) const {
        return apply(other, ApplyType::Subtract);
    }

    [[nodiscard]] AnyTensor multiply(const AnyTensor& other) const {
        return apply(other, ApplyType::Product);
    }

    [[nodiscard]] AnyTensor divide(const AnyTensor& other) const {
        return apply(other, ApplyType::Divide);
    }

    template<typename T>
    [[nodiscard]] AnyTensor add(T scalar) const;

    template<typename T>
    [[nodiscard]] AnyTensor subtract(T scalar) const;

    template<typename T>
    [[nodiscard]] AnyTensor multiply(T scalar) const;

    template<typename T>
    [[nodiscard]] AnyTensor divide(T scalar) const;

    [[nodiscard]] AnyTensor matmul(const AnyTensor& other) const {
        return apply(other, ApplyType::Matmul);
    }

    [[nodiscard]] AnyTensor transpose(const std::vector<size_t>& axis={0}) const;

    [[nodiscard]] AnyTensor sum(const std::vector<size_t>& axis={0}) const;

    [[nodiscard]] const std::vector<size_t>& shape() const;

    // Apply a custom unary operation
    template<typename Func>
    AnyTensor apply_unary(Func custom_op) const;

    // Apply a custom binary operation
    template<typename Func>
    [[nodiscard]] AnyTensor apply_custom(const AnyTensor& other, Func custom_op) const;

    [[nodiscard]] AnyTensor slice(const std::vector<Slice> &slices) const;

    // Operators
    AnyTensor operator+(const AnyTensor& other) const;

    AnyTensor operator-(const AnyTensor& other) const;

    AnyTensor operator*(const AnyTensor& other) const;

    AnyTensor operator/(const AnyTensor& other) const;

private:
    std::shared_ptr<ITensor> tensor_ptr_;
    std::type_index type_index_ = TYPE_FLOAT;
};

using a_tens = AnyTensor;

template<typename T>
[[maybe_unused]] a_tens make_tensor_ptr(T value);

template<typename T>
[[maybe_unused]] a_tens make_tensor_ptr(const std::vector<T>& data, const std::vector<size_t>& shape);

// Enumeration for tensor types
enum class TensorType {
    Float,
    Double,
    Int,
    Unknown
};



// Helper function to map type_info to TensorType
inline TensorType getTensorType(const std::type_info& type_info) {
    std::type_index t_i = {type_info};
    if (t_i == TYPE_FLOAT) {
        return TensorType::Float;
    } else if (t_i == TYPE_DOUBLE) {
        return TensorType::Double;
    } else if (t_i == TYPE_INT) {
        return TensorType::Int;
    } else {
        return TensorType::Unknown;
    }
}

inline TensorType getTensorType(const std::type_index& t_i) {
    if (t_i == TYPE_FLOAT) {
        return TensorType::Float;
    } else if (t_i == TYPE_DOUBLE) {
        return TensorType::Double;
    } else if (t_i == TYPE_INT) {
        return TensorType::Int;
    } else {
        return TensorType::Unknown;
    }
}

// Function pointer type for apply functions
using ApplyFunctionPtr = AnyTensor (*)(const AnyTensor&, const AnyTensor&, ApplyType,
                                       void* custom_op_void_ptr);

// Function declarations for all combinations
AnyTensor apply_tensors_float_float(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                    void* custom_op_void_ptr);
AnyTensor apply_tensors_float_double(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                     void* custom_op_void_ptr);
AnyTensor apply_tensors_float_int(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                  void* custom_op_void_ptr);
AnyTensor apply_tensors_double_float(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                     void* custom_op_void_ptr);
AnyTensor apply_tensors_double_double(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                      void* custom_op_void_ptr);
AnyTensor apply_tensors_double_int(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                   void* custom_op_void_ptr);
AnyTensor apply_tensors_int_float(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                  void* custom_op_void_ptr);
AnyTensor apply_tensors_int_double(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                   void* custom_op_void_ptr);
AnyTensor apply_tensors_int_int(const AnyTensor& a, const AnyTensor& b, ApplyType apply_type,
                                void* custom_op_void_ptr);

// Initialize function table
static const int numTypes = 3;  // Float, Double, Int

static ApplyFunctionPtr applyFunctionTable[numTypes][numTypes] = {
    // type1 == Float
    {apply_tensors_float_float, apply_tensors_float_double, apply_tensors_float_int},
    // type1 == Double
    {apply_tensors_double_float, apply_tensors_double_double, apply_tensors_double_int},
    // type1 == Int
    {apply_tensors_int_float, apply_tensors_int_double, apply_tensors_int_int}};

// Implementations of apply functions
template<typename T1, typename T2>
AnyTensor apply_tensors_impl(
    const AnyTensor& a, const AnyTensor& b, ApplyType apply_type, void* custom_op_void_ptr = nullptr) {
    const Tensor<T1>& tensor1 = a.get<T1>();
    const Tensor<T2>& tensor2 = b.get<T2>();

    using ResultType = typename std::common_type<T1, T2>::type;

    Tensor<ResultType> result;

    if (apply_type == ApplyType::Add) {
        result = tensor1.template elementwise_binary_operation<T2, ResultType>(tensor2, std::plus<>());
    } else if (apply_type == ApplyType::Divide) {
        result = tensor1.template elementwise_binary_operation<T2, ResultType>(tensor2, std::divides<>());
    } else if (apply_type == ApplyType::Product) {
        result = tensor1.template elementwise_binary_operation<T2, ResultType>(tensor2, std::multiplies<>());
    } else if (apply_type == ApplyType::Subtract) {
        result = tensor1.template elementwise_binary_operation<T2, ResultType>(tensor2, std::minus<>());
    } else if (apply_type == ApplyType::Matmul) {
        result = tensor1.matmul(tensor2);
    } else {
        // Use the custom binary operation if provided
        if (!custom_op_void_ptr) {
            throw std::invalid_argument("Custom operation is null");
        }
        // Cast the void* back to the appropriate function pointer type
        using FuncType = std::function<ResultType(T1, T2)>;
        auto* custom_op = static_cast<FuncType*>(custom_op_void_ptr);

        result = tensor1.template elementwise_binary_operation<T2, ResultType>(tensor2, *custom_op);
    }

    return AnyTensor(std::make_shared<Tensor<ResultType>>(std::move(result)));
}

// Define the functions
#define DEFINE_APPLY_TENSORS_FUNC(T1, T2)                                                 \
    inline AnyTensor apply_tensors_##T1##_##T2(const AnyTensor& a, const AnyTensor& b,    \
                                               ApplyType apply_type,                      \
                                               void* custom_op_void_ptr = nullptr) {      \
        return apply_tensors_impl<T1, T2>(a, b, apply_type, custom_op_void_ptr);          \
    }

DEFINE_APPLY_TENSORS_FUNC(float, float)
DEFINE_APPLY_TENSORS_FUNC(float, double)
DEFINE_APPLY_TENSORS_FUNC(float, int)
DEFINE_APPLY_TENSORS_FUNC(double, float)
DEFINE_APPLY_TENSORS_FUNC(double, double)
DEFINE_APPLY_TENSORS_FUNC(double, int)
DEFINE_APPLY_TENSORS_FUNC(int, float)
DEFINE_APPLY_TENSORS_FUNC(int, double)
DEFINE_APPLY_TENSORS_FUNC(int, int)

// Implementation of AnyTensor::apply
inline AnyTensor AnyTensor::apply(const AnyTensor& other, ApplyType apply_type,
                                  void* custom_op_void_ptr) const {
    if (!tensor_ptr_ || !other.tensor_ptr_) {
        throw std::runtime_error("Cannot apply empty AnyTensors");
    }

    TensorType type1 = getTensorType(this->type_index());
    TensorType type2 = getTensorType(other.type_index());

    if (type1 == TensorType::Unknown || type2 == TensorType::Unknown) {
        throw std::runtime_error("Unsupported tensor type in application");
    }

    ApplyFunctionPtr func = applyFunctionTable[static_cast<int>(type1)][static_cast<int>(type2)];

    if (!func) {
        throw std::runtime_error("Tensor application not implemented for these types");
    }

    return func(*this, other, apply_type, custom_op_void_ptr);
}

// Implementation of apply_custom
template<typename Func>
inline AnyTensor AnyTensor::apply_custom(const AnyTensor& other, Func custom_op) const {
    if (!tensor_ptr_ || !other.tensor_ptr_) {
        throw std::runtime_error("Cannot apply custom operation on empty AnyTensors");
    }

    // Retrieve TensorTypes
    TensorType type1 = getTensorType(this->type());
    TensorType type2 = getTensorType(other.type());

    if (type1 == TensorType::Unknown || type2 == TensorType::Unknown) {
        throw std::runtime_error("Unsupported tensor type in custom application");
    }

    // Use if-else chain to handle all combinations
    #define HANDLE_CUSTOM_APPLY(T1Enum, T2Enum, T1Type, T2Type, ResultType) \
        if (type1 == TensorType::T1Enum && type2 == TensorType::T2Enum) { \
            const Tensor<T1Type>& tensor1 = get<T1Type>(); \
            const Tensor<T2Type>& tensor2 = other.get<T2Type>(); \
            \
            std::function<ResultType(T1Type, T2Type)> op = custom_op; \
            \
            Tensor<ResultType> result = tensor1.template elementwise_binary_operation<T2Type, ResultType>(tensor2, op); \
            \
            return AnyTensor(std::make_shared<Tensor<ResultType>>(std::move(result))); \
        }

    // Handle all type combinations
    HANDLE_CUSTOM_APPLY(Float, Float, float, float, float)
    HANDLE_CUSTOM_APPLY(Float, Double, float, double, double)
    HANDLE_CUSTOM_APPLY(Float, Int, float, int, float)
    HANDLE_CUSTOM_APPLY(Double, Float, double, float, double)
    HANDLE_CUSTOM_APPLY(Double, Double, double, double, double)
    HANDLE_CUSTOM_APPLY(Double, Int, double, int, double)
    HANDLE_CUSTOM_APPLY(Int, Float, int, float, float)
    HANDLE_CUSTOM_APPLY(Int, Double, int, double, double)
    HANDLE_CUSTOM_APPLY(Int, Int, int, int, int)

    #undef HANDLE_CUSTOM_APPLY

    throw std::runtime_error("Unsupported tensor type combination for custom operation");
}

// Implementation of apply_unary
template<typename Func>
inline AnyTensor AnyTensor::apply_unary(Func custom_op) const {
    if (!tensor_ptr_) {
        throw std::runtime_error("Cannot apply custom operation on empty AnyTensor");
    }

    // Retrieve TensorType
    TensorType type1 = getTensorType(this->type());

    if (type1 == TensorType::Unknown) {
        throw std::runtime_error("Unsupported tensor type in custom unary operation");
    }

    // Use if-else chain to handle all combinations
    #define HANDLE_UNARY_APPLY(T1Enum, T1Type, ResultType) \
        if (type1 == TensorType::T1Enum) { \
            const Tensor<T1Type>& tensor1 = get<T1Type>(); \
            std::function<ResultType(T1Type)> op = custom_op; \
            \
            Tensor<ResultType> result = tensor1.apply_elementwise_function(op); \
            \
            return AnyTensor(std::make_shared<Tensor<ResultType>>(std::move(result))); \
        }

    // Handle all type combinations
    HANDLE_UNARY_APPLY(Float, float, float)
    HANDLE_UNARY_APPLY(Double, double, double)
    HANDLE_UNARY_APPLY(Int, int, int)

    #undef HANDLE_UNARY_APPLY

    throw std::runtime_error("Unsupported tensor type for custom unary operation");
}

template<typename T>
[[maybe_unused]] a_tens make_tensor_ptr(T value) {
    return dio::AnyTensor(std::make_shared<dio::Tensor<T>>(value));
}


template<typename T>
[[maybe_unused]] a_tens make_tensor_ptr(const std::vector<T>& data, const std::vector<size_t>& shape) {
    return dio::AnyTensor(std::make_shared<dio::Tensor<T>>(data, shape));
}

template<typename T>
AnyTensor AnyTensor::add(const T scalar) const {
    auto other = AnyTensor(scalar);
    return add(other);
}

template<typename T>
AnyTensor AnyTensor::subtract(const T scalar) const {
    auto other = AnyTensor(scalar);
    return subtract(other);
}

template<typename T>
AnyTensor AnyTensor::multiply(const T scalar) const {
    auto other = AnyTensor(scalar);
    return multiply(other);
}

template<typename T>
AnyTensor AnyTensor::divide(const T scalar) const {
    auto other = AnyTensor(scalar);
    return subtract(other);
}



}  // namespace dio

#endif  // GEODIO_ANYTENSOR_H
