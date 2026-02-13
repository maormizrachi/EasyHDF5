#ifndef HDF5READER_HPP
#define HDF5READER_HPP

#include <H5Cpp.h>
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

    /**
    Loads the file `filename` and prepares it for reading.
    */
    void Load(const std::string &filename);

    /**
        Reads the names of the groups at `path`.
    */
    std::vector<std::string> ReadGroupNames(const std::string &path) const;

    /**
    Checks if the element at `path` exists.
    */
    bool Exists(const std::string &path) const;

    /**
    Reads the element at `path` into `data`
    */
    template<typename T>
    void ReadElement(const std::string &path, T &data) const;

private:
    H5::H5File file_;
    bool loaded_ = false;
};

template<typename T>
void HDF5Reader::ReadElement(const std::string &path, T &data) const
{
    if(not loaded_)
    {
        throw std::runtime_error("HDF5Reader: Load() must be called before ReadElement()");
    }

    auto [groupPath, name] = HDF5Utils::splitPathAndName(path);

    const H5::Group group = HDF5Utils::openGroupPath(file_, groupPath);
    if(not group.exists(name))
    {
        throw std::runtime_error("HDF5Reader: dataset does not exist: " + path + " in group " + groupPath);
    }
    const H5::DataSet dataset = group.openDataSet(name);

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
