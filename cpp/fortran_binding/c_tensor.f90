! In your Fortran code
use iso_c_binding

type, bind(C) :: TensorHandle
    type(C_PTR) :: tensor_ptr
    character(C_CHAR) :: type_code
end type TensorHandle

interface
    function tensor_create_float(data, shape, ndim) bind(C, name="tensor_create_float")
        import :: C_PTR, C_FLOAT, C_SIZE_T
        type(C_PTR) :: tensor_create_float
        real(C_FLOAT), intent(in) :: data(*)
        integer(C_SIZE_T), intent(in) :: shape(*)
        integer(C_SIZE_T), value :: ndim
    end function tensor_create_float

    subroutine tensor_destroy(handle) bind(C, name="tensor_destroy")
        import :: C_PTR
        type(C_PTR), value :: handle
    end subroutine tensor_destroy

    function tensor_add(a, b) bind(C, name="tensor_add")
        import :: C_PTR
        type(C_PTR) :: tensor_add
        type(C_PTR), value :: a, b
    end function tensor_add

    ! Other functions...
end interface

! Usage in Fortran code
