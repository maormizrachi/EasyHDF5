#include "HDF5Reader.hpp"

HDF5Reader::HDF5Reader()
{}

HDF5Reader::HDF5Reader(const std::string &filename)
{
    this->Load(filename);
}

HDF5Reader::HDF5Reader(hid_t file_id)
    : file_(file_id), loaded_(true), owns_file_(false)
{
}

HDF5Reader::~HDF5Reader()
{
    if(owns_file_ && loaded_ && file_ >= 0)
    {
        H5Fclose(file_);
    }
}

void HDF5Reader::Load(const std::string &filename)
{
    if(owns_file_ && loaded_ && file_ >= 0)
    {
        H5Fclose(file_);
    }
    file_ = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    loaded_ = true;
    owns_file_ = true;
}

static herr_t collectGroupNames(hid_t /*group*/, const char *name, const H5L_info_t * /*info*/, void *op_data)
{
    auto *names = static_cast<std::vector<std::string>*>(op_data);
    names->push_back(std::string(name));
    return 0;
}

std::vector<std::string> HDF5Reader::ReadGroupNames(const std::string &path) const
{
    if(not loaded_)
    {
        throw std::runtime_error("HDF5Reader: Load() must be called before ReadGroupNames()");
    }
    HDF5Utils::HID group(HDF5Utils::openGroupPath(this->file_, path));
    std::vector<std::string> names;
    hsize_t idx = 0;
    H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, &idx, collectGroupNames, &names);
    return names;
}

bool HDF5Reader::Exists(const std::string &path) const
{
    if(not loaded_)
    {
        throw std::runtime_error("HDF5Reader: Load() must be called before Exists()");
    }

    H5E_auto2_t old_func;
    void *old_client_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    htri_t result = H5Lexists(this->file_, path.c_str(), H5P_DEFAULT);

    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);

    return result > 0;
}
