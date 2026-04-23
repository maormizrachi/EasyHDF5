#ifndef HDF5READER_DETAIL_HPP
#define HDF5READER_DETAIL_HPP

#include <hdf5.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include "HDF5Helper.hpp"

namespace HDF5Reader_detail
{
    template<typename T>
    void ReadScalarData(hid_t dataset_id, T &data)
    {
        if constexpr(std::is_same_v<T, std::string>)
        {
            hid_t strType = HDF5Utils::HDF5Type<std::string>::value();
            char *cstr = nullptr;
            H5Dread(dataset_id, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cstr);
            data = std::string(cstr);
            HDF5Utils::HID space(H5Dget_space(dataset_id));
            H5Dvlen_reclaim(strType, space, H5P_DEFAULT, &cstr);
        }
        else
        {
            hid_t mem_type;
            if constexpr(HDF5Utils::HasCompType<T>::value)
                mem_type = HDF5Utils::CompTypeCreator<T>::get();
            else
                mem_type = HDF5Utils::HDF5Type<T>::value();
            H5Dread(dataset_id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
        }
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
    void ReadRectangularData(hid_t dataset_id, Container &data, const hsize_t *dims, int ndims)
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

            if constexpr(std::is_same_v<Scalar, std::string>)
            {
                std::vector<std::string> flat(total);
                if(total > 0)
                {
                    hid_t strType = HDF5Utils::HDF5Type<std::string>::value();
                    std::vector<char*> rdata(total);
                    H5Dread(dataset_id, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata.data());
                    for(size_t i = 0; i < total; i++)
                        flat[i] = std::string(rdata[i]);
                    HDF5Utils::HID space(H5Dget_space(dataset_id));
                    H5Dvlen_reclaim(strType, space, H5P_DEFAULT, rdata.data());
                }

                HDF5Utils::ContainerResize(data, dims[0]);
                size_t stride = 1;
                for(int i = 1; i < ndims; ++i) stride *= dims[i];
                for(hsize_t i = 0; i < dims[0]; ++i)
                {
                    ReadRectangularDataUnflatten<std::string>(flat.data() + i * stride, dims + 1, ndims - 1, data[i]);
                }
            }
            else
            {
                std::vector<Scalar> flat(total);
                hid_t mem_type;
                if constexpr(HDF5Utils::HasCompType<Scalar>::value)
                    mem_type = HDF5Utils::CompTypeCreator<Scalar>::get();
                else
                    mem_type = HDF5Utils::HDF5Type<Scalar>::value();
                H5Dread(dataset_id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());

                HDF5Utils::ContainerResize(data, dims[0]);
                size_t stride = 1;
                for (int i = 1; i < ndims; ++i) stride *= dims[i];
                for (hsize_t i = 0; i < dims[0]; ++i)
                {
                    ReadRectangularDataUnflatten<Scalar>(flat.data() + i * stride, dims + 1, ndims - 1, data[i]);
                }
            }
        }
        else if constexpr(std::is_same_v<T, std::string>)
        {
            size_t total = 1;
            for(int i = 0; i < ndims; ++i)
            {
                total *= dims[i];
            }
            HDF5Utils::ContainerResize(data, total);
            if(total > 0)
            {
                hid_t strType = HDF5Utils::HDF5Type<std::string>::value();
                std::vector<char*> rdata(total);
                H5Dread(dataset_id, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata.data());
                for(size_t i = 0; i < total; i++)
                    data[i] = std::string(rdata[i]);
                HDF5Utils::HID space(H5Dget_space(dataset_id));
                H5Dvlen_reclaim(strType, space, H5P_DEFAULT, rdata.data());
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
            hid_t mem_type;
            if constexpr(HDF5Utils::HasCompType<T>::value)
                mem_type = HDF5Utils::CompTypeCreator<T>::get();
            else
                mem_type = HDF5Utils::HDF5Type<T>::value();
            if(not data.empty())
            {
                H5Dread(dataset_id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
            }
        }
    }

    template<typename Container>
    void ReadJaggedDataNestedVLENImpl(void *ptr, size_t count, Container &out, hid_t inner_tid)
    {
        using T = typename Container::value_type;
        if constexpr(not HDF5Utils::IsContainer<T>::value)
        {
            HDF5Utils::ContainerResize(out, count);
            T *raw = static_cast<T*>(ptr);
            std::copy(raw, raw + count, out.begin());
        }
        else
        {
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

    template<typename Container>
    void ReadJaggedDataNestedVLEN(hid_t dataset_id, Container &data)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;
        using Scalar = typename HDF5Utils::InnerType<T>::type;
        constexpr int vlen_depth = HDF5Utils::Rank<T>::value;

        hid_t base_tid;
        if constexpr(HDF5Utils::HasCompType<Scalar>::value)
        {
            base_tid = HDF5Utils::CompTypeCreator<Scalar>::get();
        }
        else
        {
            base_tid = HDF5Utils::HDF5Type<Scalar>::value();
        }
        std::vector<hid_t> type_chain;
        type_chain.push_back(base_tid);
        for(int i = 0; i < vlen_depth; i++)
        {
            type_chain.push_back(H5Tvlen_create(type_chain.back()));
        }
        hid_t vlen_tid = type_chain.back();

        HDF5Utils::HID filespace(H5Dget_space(dataset_id));
        hsize_t dims[1];
        H5Sget_simple_extent_dims(filespace, dims, nullptr);

        std::vector<hvl_t> vhl(dims[0]);
        H5Dread(dataset_id, vlen_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, vhl.data());

        HDF5Utils::ContainerResize(data, dims[0]);
        for(hsize_t i = 0; i < dims[0]; i++)
        {
            hid_t inner_tid = type_chain[type_chain.size() - 2];
            ReadJaggedDataNestedVLENImpl(vhl[i].p, vhl[i].len, data[i], inner_tid);
        }

        H5Dvlen_reclaim(vlen_tid, filespace, H5P_DEFAULT, vhl.data());
        for(size_t i = 1; i < type_chain.size(); i++)
        {
            H5Tclose(type_chain[i]);
        }
    }

    template<typename Container>
    void ReadJaggedDataImpl(hid_t dataset_id, Container &data, hid_t vlen_tid)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;

        hid_t base_type;
        if constexpr(HDF5Utils::HasCompType<T>::value)
            base_type = HDF5Utils::CompTypeCreator<T>::get();
        else
            base_type = HDF5Utils::HDF5Type<T>::value();

        HDF5Utils::HID filespace(H5Dget_space(dataset_id));
        hsize_t dims[1];
        H5Sget_simple_extent_dims(filespace, dims, nullptr);

        hid_t vlen_type = H5Tvlen_create(base_type);
        std::vector<hvl_t> vhl(dims[0]);
        H5Dread(dataset_id, vlen_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, vhl.data());

        HDF5Utils::ContainerResize(data, dims[0]);
        for(hsize_t i = 0; i < dims[0]; i++)
        {
            data[i].resize(vhl[i].len);
            if (vhl[i].len > 0)
                memcpy(data[i].data(), vhl[i].p, vhl[i].len * sizeof(T));
        }

        H5Dvlen_reclaim(vlen_tid, filespace, H5P_DEFAULT, vhl.data());
        H5Tclose(vlen_type);
    }

    template<typename Container>
    void ReadJaggedDataFromVlen(hid_t dataset_id, Container &data)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;

        HDF5Utils::HID dtype(H5Dget_type(dataset_id));
        const H5T_class_t type_class = H5Tget_class(dtype);
        if(type_class != H5T_VLEN)
        {
            HDF5Utils::HID filespace(H5Dget_space(dataset_id));
            int ndims = H5Sget_simple_extent_ndims(filespace);
            std::vector<hsize_t> dims(ndims);
            H5Sget_simple_extent_dims(filespace, dims.data(), nullptr);
            ReadRectangularData(dataset_id, data, dims.data(), ndims);
            return;
        }

        hid_t base_id = H5Tget_super(dtype);
        const H5T_class_t base_class = H5Tget_class(base_id);
        H5Tclose(base_id);

        if(base_class == H5T_VLEN)
        {
            ReadJaggedDataNestedVLEN(dataset_id, data);
            return;
        }

        if constexpr(not HDF5Utils::IsContainer<T>::value)
        {
            hid_t base_tid;
            if constexpr(HDF5Utils::HasCompType<T>::value)
                base_tid = HDF5Utils::CompTypeCreator<T>::get();
            else
                base_tid = HDF5Utils::HDF5Type<T>::value();
            const hid_t vlen_tid = H5Tvlen_create(base_tid);
            ReadJaggedDataImpl(dataset_id, data, vlen_tid);
            H5Tclose(vlen_tid);
        }
    }

    template<typename Container>
    void ReadContainerData(hid_t dataset_id, Container &data)
    {
        using T = typename Container::value_type;
        HDF5Utils::HID filespace(H5Dget_space(dataset_id));
        int ndims = H5Sget_simple_extent_ndims(filespace);
        if(ndims == 0)
        {
            if constexpr(not HDF5Utils::IsContainer<T>::value)
            {
                T scalar;
                ReadScalarData(dataset_id, scalar);
                HDF5Utils::ContainerResize(data, 1);
                data[0] = scalar;
                return;
            }
            throw std::runtime_error("HDF5Reader: cannot read scalar dataset into nested container type");
        }

        std::vector<hsize_t> dims(ndims);
        H5Sget_simple_extent_dims(filespace, dims.data(), nullptr);

        HDF5Utils::HID dtype(H5Dget_type(dataset_id));
        const H5T_class_t type_class = H5Tget_class(dtype);

        if constexpr(HDF5Utils::Rank<T>::value >= 2 && HDF5Utils::ContainsVector<T>::value)
        {
            if(type_class == H5T_VLEN and dims.size() == 1)
            {
                ReadJaggedDataFromVlen(dataset_id, data);
                return;
            }   
        }

        ReadRectangularData(dataset_id, data, dims.data(), ndims);
    }
}

#endif // HDF5READER_DETAIL_HPP
