#ifndef HDF5HELPER_HPP
#define HDF5HELPER_HPP

#include <iostream>
#include <H5Cpp.h>
#include <vector>
#include <array>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace HDF5Utils
{
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

    // True if T or any nested container inside T is a std::vector.
    // Used to determine if a type hierarchy can represent jagged (VLEN) data.
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

    /** True if T supports HDF5 compound type (has CreateHDF5CompType or CompTypeCreator<T> is specialized). */
    template<typename T, typename = void>
    struct HasCompType : std::false_type {};
    template<typename T>
    struct HasCompType<T, std::void_t<decltype(T::CreateHDF5CompType())>> : std::true_type {};

    /** Returns H5::CompType for T. Use CreateHDF5CompType() by default; specialize for custom names (e.g. CreateParticleType). */
    template<typename T>
    struct CompTypeCreator
    {
        static H5::CompType get() { return T::CreateHDF5CompType(); }
    };

    template<typename T>
    struct HDF5Type
    {
        static_assert(sizeof(T) == 0, "HDF5Type: unsupported scalar type. Add specialization for double, float, int, long, size_t, unsigned long long.");
    };
    template<> struct HDF5Type<double> { static const H5::PredType& value() { return H5::PredType::NATIVE_DOUBLE; } };
    template<> struct HDF5Type<float> { static const H5::PredType& value() { return H5::PredType::NATIVE_FLOAT; } };
    template<> struct HDF5Type<int> { static const H5::PredType& value() { return H5::PredType::NATIVE_INT; } };
    template<> struct HDF5Type<long> { static const H5::PredType& value() { return H5::PredType::NATIVE_LONG; } };
    template<> struct HDF5Type<size_t> { static const H5::PredType& value() { return H5::PredType::NATIVE_ULLONG; } };
    template<> struct HDF5Type<unsigned long long> { static const H5::PredType& value() { return H5::PredType::NATIVE_ULLONG; } };

    std::vector<std::string> splitPath(const std::string &path);

    H5::Group openGroupPath(H5::H5File &file, const std::string &groupPath, bool create = false);

    const H5::Group openGroupPath(const H5::H5File &file, const std::string &groupPath);

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