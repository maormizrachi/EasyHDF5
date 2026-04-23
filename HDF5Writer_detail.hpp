#ifndef HDF5WRITER_DETAIL_HPP
#define HDF5WRITER_DETAIL_HPP

#include <hdf5.h>
#include <string>
#include <deque>
#include "HDF5Helper.hpp"

namespace HDF5Writer_detail
{
    template<typename Container>
    void CreateVHLsImpl(const Container &data, std::vector<hvl_t> &pointers, std::deque<std::vector<hvl_t>> &storage)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;
        if constexpr(not HDF5Utils::IsContainer<T>::value)
        {
            for (size_t i = 0; i < data.size(); i++)
            {
                hvl_t hvl;
                hvl.p = const_cast<void*>(static_cast<const void*>(data[i].data()));
                hvl.len = data[i].size();
                pointers.push_back(hvl);
            }
        }
        else
        {
            for(size_t i = 0; i < data.size(); i++)
            {
                const size_t idx = storage.size();
                storage.emplace_back();
                CreateVHLsImpl(data[i], storage.back(), storage);
                hvl_t hvl;
                hvl.p = const_cast<void*>(static_cast<const void*>(storage[idx].data()));
                hvl.len = storage[idx].size();
                pointers.push_back(hvl);
            }
        }
    }

    template<typename Container>
    void WriteJaggedDataNestedVLEN(hid_t group_id, const std::string &name, const Container &data,
                                    std::vector<hvl_t> &vhl, std::deque<std::vector<hvl_t>> &storage)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;
        using Scalar = typename HDF5Utils::InnerType<T>::type;
        constexpr int vlen_depth = HDF5Utils::Rank<T>::value;

        std::vector<hid_t> type_chain;
        hid_t base_tid;
        if constexpr(HDF5Utils::HasCompType<Scalar>::value)
        {
            base_tid = HDF5Utils::CompTypeCreator<Scalar>::get();
        }
        else
        {
            base_tid = HDF5Utils::HDF5Type<Scalar>::value();
        }
        type_chain.push_back(base_tid);
        for (int i = 0; i < vlen_depth; i++)
        {
            type_chain.push_back(H5Tvlen_create(type_chain.back()));
        }
        hid_t vlen_tid = type_chain.back();

        hsize_t dims[] = {static_cast<hsize_t>(data.size())};
        hid_t space_id = H5Screate_simple(1, dims, nullptr);
        hid_t dset_id = H5Dcreate2(group_id, name.c_str(), vlen_tid, space_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        H5Dwrite(dset_id, vlen_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, vhl.data());

        H5Dclose(dset_id);
        H5Sclose(space_id);
        for (size_t i = 1; i < type_chain.size(); i++)
        {
            H5Tclose(type_chain[i]);
        }
    }

    template<typename Container>
    void WriteJaggedData(hid_t group_id, const std::string &name, const Container &data)
    {
        using Inner = typename Container::value_type;
        using T = typename Inner::value_type;
        std::vector<hvl_t> vhl;
        std::deque<std::vector<hvl_t>> storage;
        CreateVHLsImpl(data, vhl, storage);

        if constexpr(HDF5Utils::IsContainer<T>::value)
        {
            WriteJaggedDataNestedVLEN(group_id, name, data, vhl, storage);
        }
        else
        {
            hid_t base_tid;
            if constexpr(HDF5Utils::HasCompType<T>::value)
                base_tid = HDF5Utils::CompTypeCreator<T>::get();
            else
                base_tid = HDF5Utils::HDF5Type<T>::value();

            hid_t vlen_tid = H5Tvlen_create(base_tid);
            hsize_t dims[] = {static_cast<hsize_t>(data.size())};
            HDF5Utils::HID space(H5Screate_simple(1, dims, nullptr));
            HDF5Utils::HID dset(H5Dcreate2(group_id, name.c_str(), vlen_tid, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            H5Dwrite(dset, vlen_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, vhl.data());
            H5Tclose(vlen_tid);
        }
    }

    template<typename Container>
    void flattenRectangular(const Container &data, std::vector<typename HDF5Utils::InnerType<Container>::type> &flat)
    {
        using T = typename Container::value_type;
        if constexpr(HDF5Utils::IsContainer<T>::value)
        {
            for (const auto &row : data)
            {
                flattenRectangular(row, flat);
            }
        }
        else
        {
            for (const auto &x : data) {
                flat.push_back(x);
            }
        }
    }

    template<typename Container>
    void WriteRectangularData(hid_t group_id, const std::string &name, const Container &data, const hsize_t *dims, int ndims)
    {
        using T = typename Container::value_type;
        if constexpr(HDF5Utils::IsContainer<T>::value)
        {
            using Scalar = typename HDF5Utils::InnerType<Container>::type;
            std::vector<Scalar> flat;
            flattenRectangular(data, flat);
            WriteRectangularData(group_id, name, flat, dims, ndims);
        }
        else if constexpr(std::is_same_v<T, std::string>)
        {
            hid_t strType = HDF5Utils::HDF5Type<std::string>::value();
            HDF5Utils::HID space(H5Screate_simple(ndims, dims, nullptr));
            HDF5Utils::HID dset(H5Dcreate2(group_id, name.c_str(), strType, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            if(not data.empty())
            {
                std::vector<const char*> cstrs(data.size());
                for(size_t i = 0; i < data.size(); i++)
                    cstrs[i] = data[i].c_str();
                H5Dwrite(dset, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, cstrs.data());
            }
        }
        else
        {
            hid_t mem_type;
            if constexpr(HDF5Utils::HasCompType<T>::value)
            {
                mem_type = HDF5Utils::CompTypeCreator<T>::get();
            }
            else
            {
                mem_type = HDF5Utils::HDF5Type<T>::value();
            }

            hid_t file_type = mem_type;
            HDF5Utils::HID packed_type;
            if constexpr(HDF5Utils::HasCompType<T>::value)
            {
                packed_type = HDF5Utils::HID(H5Tcopy(mem_type));
                H5Tpack(packed_type);
                file_type = packed_type;
            }

            HDF5Utils::HID space(H5Screate_simple(ndims, dims, nullptr));
            HDF5Utils::HID dset(H5Dcreate2(group_id, name.c_str(), file_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            if(not data.empty())
            {
                H5Dwrite(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
            }
            else
            {
                H5Dwrite(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, nullptr);
            }
        }
    }

    template<typename Container>
    bool isRectangular(const Container &data, std::vector<hsize_t> &dims)
    {
        using T = typename Container::value_type;
        dims.clear();
        if constexpr(not HDF5Utils::IsContainer<T>::value) 
        {
            dims.push_back(static_cast<hsize_t>(data.size()));
            return true;
        }
        if(data.empty())
        {
            dims.assign(static_cast<size_t>(HDF5Utils::Rank<T>::value), 1);
            return true;
        }
        dims.push_back(static_cast<hsize_t>(data.size()));
        const size_t first = data[0].size();
        for(size_t i = 1; i < data.size(); ++i)
        {
            if(data[i].size() != first)
            {
                return false;
            }
        }
        if constexpr(HDF5Utils::IsContainer<typename T::value_type>::value)
        {
            std::vector<hsize_t> inner_dims;
            if(not isRectangular(data[0], inner_dims))
            {
                return false;
            }
            dims.insert(dims.end(), inner_dims.begin(), inner_dims.end());
            for(size_t i = 1; i < data.size(); ++i) 
            {
                std::vector<hsize_t> row_dims;
                if(not isRectangular(data[i], row_dims))
                {
                    return false;
                }
                if(row_dims != inner_dims)
                {
                    return false;
                }
            }
        }
        else
        {
            dims.push_back(static_cast<hsize_t>(first));
        }
        return true;
    }

    template<typename Container>
    void WriteContainerData(hid_t group_id, const std::string &name, const Container &data)
    {
        using T = typename Container::value_type;
        if constexpr(HDF5Utils::IsContainer<T>::value) 
        {
            std::vector<hsize_t> dims;
            if (isRectangular(data, dims))
            {
                WriteRectangularData(group_id, name, data, dims.data(), static_cast<int>(dims.size()));
            }
            else 
            {
                WriteJaggedData(group_id, name, data);
            }
        }
        else
        {
            hsize_t dims[] = {static_cast<hsize_t>(data.size())};
            WriteRectangularData(group_id, name, data, dims, 1);
        }
    }

    template<typename T>
    void WriteScalarData(hid_t group_id, const std::string &name, const T &data)
    {
        if constexpr(std::is_same_v<T, std::string>)
        {
            hid_t strType = HDF5Utils::HDF5Type<std::string>::value();
            HDF5Utils::HID space(H5Screate(H5S_SCALAR));
            HDF5Utils::HID dset(H5Dcreate2(group_id, name.c_str(), strType, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            const char *cstr = data.c_str();
            H5Dwrite(dset, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cstr);
        }
        else
        {
            HDF5Utils::HID space(H5Screate(H5S_SCALAR));
            hid_t mem_type;
            if constexpr(HDF5Utils::HasCompType<T>::value)
                mem_type = HDF5Utils::CompTypeCreator<T>::get();
            else
                mem_type = HDF5Utils::HDF5Type<T>::value();
            HDF5Utils::HID dset(H5Dcreate2(group_id, name.c_str(), mem_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            H5Dwrite(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
        }
    }
}

#endif // HDF5WRITER_DETAIL_HPP
