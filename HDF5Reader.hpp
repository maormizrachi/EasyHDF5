#ifndef HDF5READER_HPP
#define HDF5READER_HPP

#include <hdf5.h>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include "HDF5Helper.hpp"
#include "HDF5Reader_detail.hpp"

class HDF5Reader
{
public:
    HDF5Reader();

    HDF5Reader(const std::string &filename);

    /** Accepts an already-opened file handle. Caller retains ownership (will NOT be closed by this class). */
    explicit HDF5Reader(hid_t file_id);

    ~HDF5Reader();

    void Load(const std::string &filename);

    std::vector<std::string> ReadGroupNames(const std::string &path) const;

    bool Exists(const std::string &path) const;

    template<typename T>
    void ReadElement(const std::string &path, T &data) const;

    hid_t GetFileId() const { return file_; }

private:
    hid_t file_ = H5I_INVALID_HID;
    bool loaded_ = false;
    bool owns_file_ = true;
};

template<typename T>
void HDF5Reader::ReadElement(const std::string &path, T &data) const
{
    if(not loaded_)
    {
        throw std::runtime_error("HDF5Reader: Load() must be called before ReadElement()");
    }

    auto [groupPath, name] = HDF5Utils::splitPathAndName(path);

    HDF5Utils::HID group(HDF5Utils::openGroupPath(file_, groupPath));
    htri_t exists = H5Lexists(group, name.c_str(), H5P_DEFAULT);
    if(exists <= 0)
    {
        throw std::runtime_error("HDF5Reader: dataset does not exist: " + path + " in group " + groupPath);
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

#endif // HDF5READER_HPP
