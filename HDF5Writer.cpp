#include "HDF5Writer.hpp"

HDF5Writer::HDF5Writer(const std::string &filename, bool truncate)
{
    this->file_ = H5::H5File(filename, truncate ? H5F_ACC_TRUNC : H5F_ACC_RDWR);
}

void HDF5Writer::Dump(void)
{
    for(const Element &element : data)
    {
        H5::Group group = HDF5Utils::openGroupPath(this->file_, element.groupPath, true);
        element.write(group);
        group.close();
    }

    this->file_.close();
}

void HDF5Writer::AddExternalLink(const std::string &externalFile, const std::string &targetPath, const std::string &linkPath)
{
    // Create parent groups for the link location
    auto [groupPath, linkName] = HDF5Utils::splitPathAndName(linkPath);

    H5::Group group = HDF5Utils::openGroupPath(this->file_, groupPath, true);
    H5Lcreate_external(externalFile.c_str(),  // the other .h5 file
                        targetPath.c_str(),    // path inside that file (e.g. "/dataset")
                        group.getId(),         // where the link lives in THIS file
                        linkName.c_str(),      // name of the link
                        H5P_DEFAULT, H5P_DEFAULT);
}

HDF5Writer::~HDF5Writer()
{
    this->Close();
}

void HDF5Writer::Close(void)
{
    if(not closed)
    {
        this->file_.close();
        closed = true;
    }
}