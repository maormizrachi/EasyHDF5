#include "HDF5Reader.hpp"

namespace HDF5Reader_detail
{}

HDF5Reader::HDF5Reader()
{}

void HDF5Reader::Load(const std::string &filename)
{
    file_ = H5::H5File(filename, H5F_ACC_RDONLY);
    this->loaded_ = true;
}
