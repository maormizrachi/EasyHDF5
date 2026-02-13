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
    
    H5::Group openGroupPath(H5::H5File &file, const std::string &groupPath, bool create)
    {
        if(groupPath.empty())
        {
            return file.openGroup("/");
        }
    
        std::vector<std::string> parts = splitPath(groupPath);
        H5::Group group = file.openGroup("/");
        for(const std::string &name : parts)
        {
            if(name.empty())
            {
                continue;
            }
            bool groupExists = group.exists(name);
            if(not groupExists)
            {
                if(create)
                {
                    group.createGroup(name);
                }
                else
                {
                    throw std::runtime_error("HDF5Reader: group does not exist: " + name);
                }
            }
            H5::Group next = group.openGroup(name);
            group.close();
            group = next;
        }
        return group;
    }
}
