#include "HDF5Reader.hpp"

HDF5Reader::HDF5Reader()
{}

HDF5Reader::HDF5Reader(const std::string &filename)
{
    this->Load(filename);
}

void HDF5Reader::Load(const std::string &filename)
{
    file_ = H5::H5File(filename, H5F_ACC_RDONLY);
    this->loaded_ = true;
}

std::vector<std::string> HDF5Reader::ReadGroupNames(const std::string &path) const
{
    if(not loaded_)
    {
        throw std::runtime_error("HDF5Reader: Load() must be called before ReadGroupNames()");
    }
    H5::Group group = HDF5Utils::openGroupPath(this->file_, path);
    std::vector<std::string> names;
    for(hsize_t n = 0; n < group.getNumObjs(); ++n)
    {
        const H5std_string name = group.getObjnameByIdx(n);
        names.push_back(name);
    }
    group.close();
    return names;
}

bool HDF5Reader::Exists(const std::string &path) const
{
    if(not loaded_)
    {
        throw std::runtime_error("HDF5Reader: Load() must be called before Exists()");
    }

    return H5Lexists(this->file_.getId(), path.c_str(), H5P_DEFAULT) > 0;
}