#include "HDF5Writer.hpp"

HDF5Writer::HDF5Writer(const std::string &filename)
{
    this->file_ = H5::H5File(filename, H5F_ACC_TRUNC);
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
    std::string groupPath, linkName;
    if(linkPath.find("/") == std::string::npos)
    {
        groupPath = "";
        linkName = linkPath;
    }
    else
    {
        linkName = linkPath.substr(linkPath.find_last_of("/") + 1);
        groupPath = linkPath.substr(0, linkPath.find_last_of("/"));
    }

    H5::Group group = HDF5Utils::openGroupPath(this->file_, groupPath, true);
    H5Lcreate_external(externalFile.c_str(),  // the other .h5 file
                        targetPath.c_str(),    // path inside that file (e.g. "/dataset")
                        group.getId(),         // where the link lives in THIS file
                        linkName.c_str(),      // name of the link
                        H5P_DEFAULT, H5P_DEFAULT);
    group.close();
}