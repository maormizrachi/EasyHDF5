#ifndef HDF5WRITER_HPP
#define HDF5WRITER_HPP

#include <H5Cpp.h>
#include <string>
#include <functional>
#include <set>
#include <any>
#include "HDF5Writer_detail.hpp"

class HDF5Writer
{
public:
    HDF5Writer(const std::string &filename, bool truncate = true);

    ~HDF5Writer();

    /**
    Closes the writer and the file.
    */
    void Close(void);

    /**
    Dumps the stored data to an HDF5 file.
    */
    void Dump(void);
    
    /**
    Writes an element at path `path`.
    */
    template<typename T>
    void WriteElement(const std::string &path, const T &data){this->AddElement(path, data, true);};

    /**
    Adds an element to the writer. `data` MUST be accessible in `Dump()`.
    */
    template<typename T>
    void AddElement(const std::string &path, const T &data, bool write = false);

    /**
    Adds a HDF5 external link to `targetPath` in file `externalFile`, saved in `linkPath` in the current file.
    */
    void AddExternalLink(const std::string &externalFile, const std::string &targetPath, const std::string &linkPath);

private:
    struct Element
    {
        std::string fullpath;
        std::string groupPath;
        std::string name;
        std::function<void(H5::Group&)> write;
        std::any data;

        bool operator<(const Element& other) const
        {
            return fullpath < other.fullpath;
        }

        bool operator==(const Element& other) const
        {
            return fullpath == other.fullpath;
        }

        bool operator!=(const Element& other) const
        {
            return not this->operator==(other);
        }
    };

    bool closed = false;
    H5::H5File file_;
    std::set<Element> data;
};

template<typename T>
void HDF5Writer::AddElement(const std::string &path, const T &data, bool write)
{
    Element element;
    
    element.fullpath = path;
    // datasetName comes right after the last "/"

    std::tie(element.groupPath, element.name) = HDF5Utils::splitPathAndName(path);

    element.data = std::make_any<const T*>(&data);

    element.write = [element](H5::Group &group)
    {
        const T &data = *std::any_cast<const T*>(element.data);
        if constexpr(HDF5Utils::IsContainer<T>::value)
        {
            HDF5Writer_detail::WriteContainerData(group, element.name, data);
        }
        else
        {
            HDF5Writer_detail::WriteScalarData(group, element.name, data);
        }
    };

    if(write)
    {
        H5::Group group = HDF5Utils::openGroupPath(this->file_, element.groupPath, true);
        element.write(group);
        group.close();
    }
    else
    {
        this->data.insert(element);
    }
}


#endif // HDF5WRITER_HPP
