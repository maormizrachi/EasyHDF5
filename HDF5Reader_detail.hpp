#ifndef HDF5READER_DETAIL_HPP
#define HDF5READER_DETAIL_HPP

#include <H5Cpp.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include "HDF5Helper.hpp"

namespace HDF5Reader_detail
{
    template<typename T>
    void ReadScalarData(H5::DataSet &dataset, T &data)
    {
        const H5::DataType &mem_type = HDF5Utils::HDF5Type<T>::value();
        dataset.read(&data, mem_type);
    }

    template<typename Scalar, typename Vec>
    void ReadRectangularDataUnflatten(Scalar *flat, const hsize_t *dims, int ndims, Vec &out)
    {
        using T = typename Vec::value_type;
        if constexpr(not HDF5Utils::IsContainer<T>::value)
        {
            HDF5Utils::ContainerResize(out, dims[0]);
            std::copy(flat, flat + dims[0], out.begin());
        }
        else
        {
            HDF5Utils::ContainerResize(out, dims[0]);
            size_t stride = 1;
            for(int i = 1; i < ndims; ++i)
            {
                stride *= dims[i];
            }
            for(hsize_t i = 0; i < dims[0]; ++i)
            {
                ReadRectangularDataUnflatten<Scalar>(flat + i * stride, dims + 1, ndims - 1, out[i]);
            }
        }
    }

    template<typename Container>
    void ReadRectangularData(H5::DataSet &dataset, Container &data, const hsize_t *dims, int ndims)
    {
        using T = typename Container::value_type;
        if constexpr(HDF5Utils::IsContainer<T>::value)
        {
            using Scalar = typename HDF5Utils::InnerType<Container>::type;
            size_t total = 1;
            for(int i = 0; i < ndims; ++i)
            {
                total *= dims[i];
            }
            std::vector<Scalar> flat(total);
            const H5::DataType &mem_type = HDF5Utils::HDF5Type<Scalar>::value();
            dataset.read(flat.data(), mem_type);

            HDF5Utils::ContainerResize(data, dims[0]);
            size_t stride = 1;
            for (int i = 1; i < ndims; ++i) stride *= dims[i];
            for (hsize_t i = 0; i < dims[0]; ++i)
            {
                ReadRectangularDataUnflatten<Scalar>(
                    flat.data() + i * stride, dims + 1, ndims - 1, data[i]);
            }
        }
        else
        {
            size_t total = 1;
            for(int i = 0; i < ndims; ++i)
            {
                total *= dims[i];
            }
            HDF5Utils::ContainerResize(data, total);
            const H5::DataType &mem_type = HDF5Utils::HDF5Type<T>::value();
            if(not data.empty())
            {
                dataset.read(data.data(), mem_type);
            }
        }
    }

    // Generic VLEN recursion: Container can be any mix of vector/array at each level.
    // Leaf case: Container holds scalars (e.g., vector<int> or array<int, N>).
    // Recursive case: Container holds sub-containers (e.g., array<vector<int>, N>, vector<vector<int>>).
    template<typename Container>
    void ReadJaggedDataNestedVLENImpl(void *ptr, size_t count, Container &out, hid_t inner_tid)
    {
        using T = typename Container::value_type;
        if constexpr(not HDF5Utils::IsContainer<T>::value)
        {
            // Leaf: Container holds scalars — copy raw data
            HDF5Utils::ContainerResize(out, count);
            T *raw = static_cast<T*>(ptr);
            std::copy(raw, raw + count, out.begin());
        }
        else
        {
            // Recursive: Container holds sub-containers
            hvl_t *inner = static_cast<hvl_t*>(ptr);
            HDF5Utils::ContainerResize(out, count);
            for(size_t i = 0; i < count; i++)
            {
                H5T_class_t c = H5Tget_class(inner_tid);
                if(c == H5T_VLEN)
                {
                    hid_t super_tid = H5Tget_super(inner_tid);
                    ReadJaggedDataNestedVLENImpl(inner[i].p, inner[i].len, out[i], super_tid);
                    H5Tclose(super_tid);
                }
                else
                {
                    // Leaf VLEN level: out[i] is a container of scalars
                    using Scalar = typename T::value_type;
                    HDF5Utils::ContainerResize(out[i], inner[i].len);
                    if(inner[i].len > 0)
                    {
                        memcpy(out[i].data(), inner[i].p, inner[i].len * sizeof(Scalar));
                    }
                }
            }
        }
    }

    // Container can be vector<vector<T>> or array<vector<T>, N> etc.
    // Inner elements (data[i]) are always vectors since jagged = variable-length.
    template<typename Container>
    void ReadJaggedDataNestedVLEN(H5::DataSet &dataset, Container &data)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;
        using Scalar = typename HDF5Utils::InnerType<T>::type;
        constexpr int vlen_depth = HDF5Utils::Rank<T>::value;

        std::vector<hid_t> type_chain;
        type_chain.push_back(HDF5Utils::HDF5Type<Scalar>::value().getId());
        for(int i = 0; i < vlen_depth; i++)
        {
            type_chain.push_back(H5Tvlen_create(type_chain.back()));
        }
        hid_t vlen_tid = type_chain.back();

        hsize_t dims[1];
        H5::DataSpace filespace = dataset.getSpace();
        filespace.getSimpleExtentDims(dims);

        std::vector<hvl_t> vhl(dims[0]);
        H5Dread(dataset.getId(), vlen_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, vhl.data());

        HDF5Utils::ContainerResize(data, dims[0]);
        for(hsize_t i = 0; i < dims[0]; i++)
        {
            hid_t inner_tid = type_chain[type_chain.size() - 2];
            ReadJaggedDataNestedVLENImpl(vhl[i].p, vhl[i].len, data[i], inner_tid);
        }

        H5Dvlen_reclaim(vlen_tid, filespace.getId(), H5P_DEFAULT, vhl.data());
        for(size_t i = 1; i < type_chain.size(); i++)
        {
            H5Tclose(type_chain[i]);
        }
    }

    template<typename Container>
    void ReadJaggedDataFromVlen(H5::DataSet &dataset, Container &data);

    // Container can be vector<vector<T>> or array<vector<T>, N>.
    // Inner elements (data[i]) must be std::vector<T> (scalar T).
    template<typename Container>
    void ReadJaggedDataImpl(H5::DataSet &dataset, Container &data, hid_t vlen_tid)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;

        hsize_t dims[1];
        H5::DataSpace filespace = dataset.getSpace();
        filespace.getSimpleExtentDims(dims);

        std::vector<hvl_t> vhl(dims[0]);
        dataset.read(vhl.data(), H5::VarLenType(&HDF5Utils::HDF5Type<T>::value()));

        HDF5Utils::ContainerResize(data, dims[0]);
        for(hsize_t i = 0; i < dims[0]; i++)
        {
            data[i].resize(vhl[i].len);
            if (vhl[i].len > 0)
                memcpy(data[i].data(), vhl[i].p, vhl[i].len * sizeof(T));
        }

        H5Dvlen_reclaim(vlen_tid, filespace.getId(), H5P_DEFAULT, vhl.data());
    }

    template<typename Container>
    void ReadJaggedDataFromVlen(H5::DataSet &dataset, Container &data)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;

        H5T_class_t type_class = dataset.getTypeClass();
        if(type_class != H5T_VLEN) 
        {
            H5::DataSpace filespace = dataset.getSpace();
            int ndims = filespace.getSimpleExtentNdims();
            std::vector<hsize_t> dims(ndims);
            filespace.getSimpleExtentDims(dims.data());
            ReadRectangularData(dataset, data, dims.data(), ndims);
            return;
        }

        H5::VarLenType vlen_type = dataset.getVarLenType();
        H5T_class_t base_class = H5Tget_class(vlen_type.getId());
        if(base_class == H5T_VLEN)
        {
            ReadJaggedDataNestedVLEN(dataset, data);
            return;
        }

        if constexpr(not HDF5Utils::IsContainer<T>::value)
        {
            hid_t vlen_tid = H5Tvlen_create(HDF5Utils::HDF5Type<T>::value().getId());
            ReadJaggedDataImpl(dataset, data, vlen_tid);
            H5Tclose(vlen_tid);
        }
    }

    template<typename Container>
    void ReadContainerData(H5::DataSet &dataset, Container &data)
    {
        using T = typename Container::value_type;
        H5::DataSpace filespace = dataset.getSpace();
        int ndims = filespace.getSimpleExtentNdims();
        if(ndims == 0)
        {
            if constexpr(not HDF5Utils::IsContainer<T>::value)
            {
                T scalar;
                ReadScalarData(dataset, scalar);
                HDF5Utils::ContainerResize(data, 1);
                data[0] = scalar;
                return;
            }
            throw std::runtime_error("HDF5Reader: cannot read scalar dataset into nested container type");
        }

        std::vector<hsize_t> dims(ndims);
        filespace.getSimpleExtentDims(dims.data());

        H5T_class_t type_class = dataset.getTypeClass();

        if constexpr(HDF5Utils::Rank<T>::value >= 2 && HDF5Utils::ContainsVector<T>::value)
        {
            // if the data is >= 2D and contains a vector somewhere (can hold jagged data),
            // check if the HDF5 dataset is VLEN
            if(type_class == H5T_VLEN and dims.size() == 1 /* in jagged data, dims.size() is 1 */)
            {
                ReadJaggedDataFromVlen(dataset, data);
                return;
            }   
        }

        // else, data is rectangular
        ReadRectangularData(dataset, data, dims.data(), ndims);
    }
}

#endif // HDF5READER_DETAIL_HPP