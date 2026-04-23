#include "HDF5Writer.hpp"

HDF5Writer::HDF5Writer(const std::string &filename, bool truncate)
    : owns_file_(true)
{
    this->file_ = truncate ? H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
                           : H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
}

HDF5Writer::HDF5Writer(hid_t file_id)
    : owns_file_(false), file_(file_id)
{
}

void HDF5Writer::Dump(void)
{
    for(const Element &element : data)
    {
        HDF5Utils::HID group(HDF5Utils::openGroupPath(this->file_, element.groupPath, true));
        element.write(group);
    }

    if(owns_file_)
    {
        H5Fclose(this->file_);
        this->file_ = H5I_INVALID_HID;
        closed_ = true;
    }
}

void HDF5Writer::AddExternalLink(const std::string &externalFile, const std::string &targetPath, const std::string &linkPath)
{
    auto [groupPath, linkName] = HDF5Utils::splitPathAndName(linkPath);

    HDF5Utils::HID group(HDF5Utils::openGroupPath(this->file_, groupPath, true));
    H5Lcreate_external(externalFile.c_str(),
                        targetPath.c_str(),
                        group,
                        linkName.c_str(),
                        H5P_DEFAULT, H5P_DEFAULT);
}

HDF5Writer::~HDF5Writer()
{
    this->Close();
}

void HDF5Writer::Close(void)
{
    if(not closed_ && owns_file_ && file_ >= 0)
    {
        H5Fclose(this->file_);
        this->file_ = H5I_INVALID_HID;
        closed_ = true;
    }
}
