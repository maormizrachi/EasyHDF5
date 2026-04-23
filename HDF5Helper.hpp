#ifndef HDF5HELPER_HPP
#define HDF5HELPER_HPP

#include <iostream>
#include <hdf5.h>
#include <vector>
#include <array>
#include <string>
#include <type_traits>
#include <stdexcept>

namespace HDF5Utils
{
    struct HID {
        hid_t id = H5I_INVALID_HID;
        HID() = default;
        explicit HID(hid_t h) : id(h) {}
        ~HID() { if (id >= 0) H5Idec_ref(id); }
        HID(const HID&) = delete;
        HID& operator=(const HID&) = delete;
        HID(HID&& o) noexcept : id(o.id) { o.id = H5I_INVALID_HID; }
        HID& operator=(HID&& o) noexcept {
            if (id >= 0) H5Idec_ref(id);
            id = o.id; o.id = H5I_INVALID_HID; return *this;
        }
        operator hid_t() const { return id; }
        hid_t release() { hid_t h = id; id = H5I_INVALID_HID; return h; }
    };

    template<typename T>
    struct IsVector : std::false_type {};
    template<typename U>
    struct IsVector<std::vector<U>> : std::true_type {};

    template<typename T>
    struct IsArray : std::false_type {};
    template<typename U, size_t N>
    struct IsArray<std::array<U, N>> : std::true_type {};

    template<typename T>
    struct IsContainer : std::bool_constant<IsVector<T>::value || IsArray<T>::value> {};

    template<typename T>
    struct ContainsVector : std::false_type {};
    template<typename U>
    struct ContainsVector<std::vector<U>> : std::true_type {};
    template<typename U, size_t N>
    struct ContainsVector<std::array<U, N>> : ContainsVector<U> {};

    template<typename T>
    struct InnerType { using type = T; };
    template<typename U>
    struct InnerType<std::vector<U>> { using type = typename InnerType<U>::type; };
    template<typename U, size_t N>
    struct InnerType<std::array<U, N>> { using type = typename InnerType<U>::type; };

    template<typename T>
    struct Rank { static constexpr int value = 1; };
    template<typename U>
    struct Rank<std::vector<U>> { static constexpr int value = 1 + Rank<U>::value; };
    template<typename U, size_t N>
    struct Rank<std::array<U, N>> { static constexpr int value = 1 + Rank<U>::value; };

    template<typename T, typename = void>
    struct HasCompType : std::false_type {};
    template<typename T>
    struct HasCompType<T, std::void_t<decltype(T::CreateHDF5CompType())>> : std::true_type {};

    /** Returns hid_t for a compound type. Returned ID is typically static; caller must NOT close it. */
    template<typename T>
    struct CompTypeCreator
    {
        static hid_t get() { return T::CreateHDF5CompType(); }
    };

    /** Maps C++ scalar types to HDF5 predefined type IDs. Returned IDs are predefined constants; caller must NOT close. */
    template<typename T>
    struct HDF5Type
    {
        static_assert(sizeof(T) == 0, "HDF5Type: unsupported scalar type. Add specialization for double, float, int, long, size_t, unsigned long long.");
    };
    template<> struct HDF5Type<double> { static hid_t value() { return H5T_NATIVE_DOUBLE; } };
    template<> struct HDF5Type<float> { static hid_t value() { return H5T_NATIVE_FLOAT; } };
    template<> struct HDF5Type<int> { static hid_t value() { return H5T_NATIVE_INT; } };
    template<> struct HDF5Type<long> { static hid_t value() { return H5T_NATIVE_LONG; } };
    template<> struct HDF5Type<size_t> { static hid_t value() { return H5T_NATIVE_ULLONG; } };
    template<> struct HDF5Type<unsigned long long> { static hid_t value() { return H5T_NATIVE_ULLONG; } };
    template<> struct HDF5Type<std::string> {
        static hid_t value() {
            static hid_t t = [](){
                hid_t s = H5Tcopy(H5T_C_S1);
                H5Tset_size(s, H5T_VARIABLE);
                return s;
            }();
            return t;
        }
    };

    std::vector<std::string> splitPath(const std::string &path);

    /** Opens (and optionally creates) groups along a path. Caller must close the returned hid_t with H5Gclose. */
    hid_t openGroupPath(hid_t loc_id, const std::string &groupPath, bool create = false);

    std::pair<std::string, std::string> splitPathAndName(const std::string &path);

    template<typename Container>
    void ContainerResize(Container &c, size_t n)
    {
        if constexpr(IsVector<Container>::value)
        {
            c.resize(n);
        }
        else
        {
            if(c.size() != n)
            {
                throw std::runtime_error("HDF5Utils: array size mismatch: expected " +
                    std::to_string(c.size()) + " but got " + std::to_string(n));
            }
        }
    }
}

#endif // HDF5HELPER_HPP
