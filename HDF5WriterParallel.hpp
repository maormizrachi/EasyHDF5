#ifndef HDF5WRITERPARALLEL_HPP
#define HDF5WRITERPARALLEL_HPP

#include <hdf5.h>

#ifdef H5_HAVE_PARALLEL
#include <mpi.h>
#include <string>
#include "HDF5Helper.hpp"
#include "HDF5Writer_detail.hpp"

class HDF5WriterParallel
{
public:
    HDF5WriterParallel(const std::string &filename, MPI_Comm comm, bool truncate = true);
    ~HDF5WriterParallel();
    void Close();

    template<typename T>
    void WriteElement(const std::string &path, const T &data);

    template<typename Container>
    void WriteSlice(const std::string &path, const Container &data, size_t count);

    /** Writes to path as-is (no rank prefix). Caller ensures no overlapping writes. */
    template<typename T>
    void WriteGlobal(const std::string &path, const T &data);

    int GetRank() const { return rank_; }
    int GetSize() const { return size_; }
    hid_t GetFileId() const { return file_id_; }
    const std::string& GetPrefix() const { return prefix_; }

private:
    hid_t file_id_ = H5I_INVALID_HID;
    int rank_ = 0;
    int size_ = 1;
    std::string prefix_;
    bool closed_ = false;
};

template<typename T>
void HDF5WriterParallel::WriteElement(const std::string &path, const T &data)
{
    std::string fullPath = prefix_ + "/" + path;
    auto [groupPath, name] = HDF5Utils::splitPathAndName(fullPath);
    HDF5Utils::HID group(HDF5Utils::openGroupPath(file_id_, groupPath, true));

    if constexpr(HDF5Utils::IsContainer<T>::value)
    {
        HDF5Writer_detail::WriteContainerData(group, name, data);
    }
    else
    {
        HDF5Writer_detail::WriteScalarData(group, name, data);
    }
}

template<typename Container>
void HDF5WriterParallel::WriteSlice(const std::string &path, const Container &data, size_t count)
{
    using T = typename Container::value_type;
    static_assert(!HDF5Utils::IsContainer<T>::value,
                  "WriteSlice only supports flat containers");

    std::string fullPath = prefix_ + "/" + path;
    auto [groupPath, name] = HDF5Utils::splitPathAndName(fullPath);
    HDF5Utils::HID group(HDF5Utils::openGroupPath(file_id_, groupPath, true));

    hsize_t dims[] = {static_cast<hsize_t>(count)};
    HDF5Utils::HID dataspace(H5Screate_simple(1, dims, nullptr));

    if constexpr(std::is_same_v<T, std::string>)
    {
        hid_t strType = HDF5Utils::HDF5Type<std::string>::value();
        HDF5Utils::HID dataset(H5Dcreate2(group, name.c_str(), strType, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if(count > 0)
        {
            std::vector<const char*> cstrs(count);
            for (size_t i = 0; i < count; i++)
                cstrs[i] = data[i].c_str();
            H5Dwrite(dataset, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, cstrs.data());
        }
    }
    else
    {
        hid_t mem_type;
        if constexpr(HDF5Utils::HasCompType<T>::value)
            mem_type = HDF5Utils::CompTypeCreator<T>::get();
        else
            mem_type = HDF5Utils::HDF5Type<T>::value();

        hid_t file_type = mem_type;
        HDF5Utils::HID packed_type;
        if constexpr(HDF5Utils::HasCompType<T>::value)
        {
            packed_type = HDF5Utils::HID(H5Tcopy(mem_type));
            H5Tpack(packed_type);
            file_type = packed_type;
        }

        HDF5Utils::HID dataset(H5Dcreate2(group, name.c_str(), file_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if(count > 0)
        {
            H5Dwrite(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        }
    }
}

template<typename T>
void HDF5WriterParallel::WriteGlobal(const std::string &path, const T &data)
{
    auto [groupPath, name] = HDF5Utils::splitPathAndName(path);
    HDF5Utils::HID group(HDF5Utils::openGroupPath(file_id_, groupPath, true));

    if constexpr(HDF5Utils::IsContainer<T>::value)
    {
        HDF5Writer_detail::WriteContainerData(group, name, data);
    }
    else
    {
        HDF5Writer_detail::WriteScalarData(group, name, data);
    }
}

#endif // H5_HAVE_PARALLEL

#endif // HDF5WRITERPARALLEL_HPP
