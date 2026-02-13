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

    /**
    Loads the file `filename` and prepares it for reading.
    */
    void Load(const std::string &filename);

    /**
    Reads the element at `path` into `data`
    */
    template<typename T>
    void ReadElement(const std::string &path, T &data);

private:
    H5::H5File file_;
    bool loaded_ = false;
};

template<typename T>
void HDF5Reader::ReadElement(const std::string &path, T &data)
{
    if(not loaded_)
    {
        throw std::runtime_error("HDF5Reader: Load() must be called before ReadElement()");
    }

    std::string groupPath;
    std::string name;
    if(path.find("/") == std::string::npos)
    {
        name = path;
        groupPath = "";
    }
    else
    {
        name = path.substr(path.find_last_of("/") + 1);
        groupPath = path.substr(0, path.find_last_of("/"));
    }

    H5::Group group = HDF5Utils::openGroupPath(file_, groupPath);
    H5::DataSet dataset = group.openDataSet(name);

    if constexpr(HDF5Utils::IsContainer<T>::value)
    {
        HDF5Reader_detail::ReadContainerData(dataset, data);
    }
    else
    {
        HDF5Reader_detail::ReadScalarData(dataset, data);
    }

    dataset.close();
    group.close();
}

#endif // HDF5READER_HPP
