#include "HDF5Helper.hpp"

namespace HDF5Utils
{
    std::vector<std::string> splitPath(const std::string &path)
    {
        std::vector<std::string> parts;
        if(path.empty())
        {
            return parts;
        }
        size_t start = 0;
        size_t end = path.find("/");
        while(end != std::string::npos)
        {
            std::string part = path.substr(start, end - start);
            if(not part.empty())
            {
                parts.push_back(part);
            }
            start = end + 1;
            end = path.find("/", start);
        }
        std::string part = path.substr(start);
        if(not part.empty())
        {
            parts.push_back(part);
        }
        return parts;
    }
    
    hid_t openGroupPath(hid_t loc_id, const std::string &groupPath, bool create)
    {
        if(groupPath.empty())
        {
            return H5Gopen2(loc_id, "/", H5P_DEFAULT);
        }
    
        std::vector<std::string> parts = splitPath(groupPath);
        hid_t current = H5Gopen2(loc_id, "/", H5P_DEFAULT);
        for(const std::string &name : parts)
        {
            if(name.empty())
            {
                continue;
            }
            htri_t exists = H5Lexists(current, name.c_str(), H5P_DEFAULT);
            if(exists <= 0)
            {
                if(create)
                {
                    hid_t created = H5Gcreate2(current, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    H5Gclose(created);
                }
                else
                {
                    H5Gclose(current);
                    throw std::runtime_error("HDF5Reader: group does not exist: " + name);
                }
            }
            hid_t next = H5Gopen2(current, name.c_str(), H5P_DEFAULT);
            H5Gclose(current);
            current = next;
        }
        return current;
    }

    std::pair<std::string, std::string> splitPathAndName(const std::string &path)
    {
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
        return std::make_pair(groupPath, name);
    }

}
