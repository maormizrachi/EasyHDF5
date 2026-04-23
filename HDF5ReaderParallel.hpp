#ifndef HDF5READERPARALLEL_HPP
#define HDF5READERPARALLEL_HPP

#include <hdf5.h>

#ifdef H5_HAVE_PARALLEL
#include <mpi.h>
#include <string>
#include <vector>
#include "HDF5Helper.hpp"
#include "HDF5Reader_detail.hpp"

class HDF5ReaderParallel
{
public:
    HDF5ReaderParallel(const std::string &filename, MPI_Comm comm);
    ~HDF5ReaderParallel();
    void Close();

    /** Reads from /rankN/path (auto-prefixed). */
    template<typename T>
    void ReadElement(const std::string &path, T &data) const;

    /** Reads from path as-is (no prefix). */
    template<typename T>
    void ReadGlobal(const std::string &path, T &data) const;

    bool Exists(const std::string &path) const;
    bool ExistsGlobal(const std::string &path) const;

    std::vector<std::string> ReadGroupNames(const std::string &path) const;

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

    void readElementImpl(const std::string &fullPath, auto &data) const;
};

template<typename T>
void HDF5ReaderParallel::ReadElement(const std::string &path, T &data) const
{
    std::string fullPath = prefix_ + "/" + path;
    auto [groupPath, name] = HDF5Utils::splitPathAndName(fullPath);

    HDF5Utils::HID group(HDF5Utils::openGroupPath(file_id_, groupPath));
    htri_t exists = H5Lexists(group, name.c_str(), H5P_DEFAULT);
    if(exists <= 0)
    {
        throw std::runtime_error("HDF5ReaderParallel: dataset does not exist: " + fullPath);
    }
    HDF5Utils::HID dataset(H5Dopen2(group, name.c_str(), H5P_DEFAULT));

    if constexpr(HDF5Utils::IsContainer<T>::value)
    {
        HDF5Reader_detail::ReadContainerData(dataset, data);
    }
    else
    {
        HDF5Reader_detail::ReadScalarData(dataset, data);
    }
}

template<typename T>
void HDF5ReaderParallel::ReadGlobal(const std::string &path, T &data) const
{
    auto [groupPath, name] = HDF5Utils::splitPathAndName(path);

    HDF5Utils::HID group(HDF5Utils::openGroupPath(file_id_, groupPath));
    htri_t exists = H5Lexists(group, name.c_str(), H5P_DEFAULT);
    if(exists <= 0)
    {
        throw std::runtime_error("HDF5ReaderParallel: dataset does not exist: " + path);
    }
    HDF5Utils::HID dataset(H5Dopen2(group, name.c_str(), H5P_DEFAULT));

    if constexpr(HDF5Utils::IsContainer<T>::value)
    {
        HDF5Reader_detail::ReadContainerData(dataset, data);
    }
    else
    {
        HDF5Reader_detail::ReadScalarData(dataset, data);
    }
}

#endif // H5_HAVE_PARALLEL

#endif // HDF5READERPARALLEL_HPP
