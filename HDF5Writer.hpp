#ifndef HDF5WRITER_HPP
#define HDF5WRITER_HPP

#include <hdf5.h>
#include <string>
#include <functional>
#include <set>
#include <any>
#include "HDF5Writer_detail.hpp"

class HDF5Writer
{
public:
    HDF5Writer(const std::string &filename, bool truncate = true);

    /** Accepts an already-opened file handle. Caller retains ownership of the handle (will NOT be closed by this class). */
    explicit HDF5Writer(hid_t file_id);

    ~HDF5Writer();

    void Close(void);

    void Dump(void);
    
    template<typename T>
    void WriteElement(const std::string &path, const T &data){this->AddElement(path, data, true);};

    template<typename Container>
    void WriteSlice(const std::string &path, const Container &data, size_t count);

    template<typename T>
    void AddElement(const std::string &path, const T &data, bool write = false);

    void AddExternalLink(const std::string &externalFile, const std::string &targetPath, const std::string &linkPath);

private:
    struct Element
    {
        std::string fullpath;
        std::string groupPath;
        std::string name;
        std::function<void(hid_t)> write;
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

    bool closed_ = false;
    bool owns_file_ = true;
    hid_t file_;
    std::set<Element> data;
};

template<typename T>
void HDF5Writer::AddElement(const std::string &path, const T &data, bool write)
{
    Element element;
    
    element.fullpath = path;

    std::tie(element.groupPath, element.name) = HDF5Utils::splitPathAndName(path);

    element.data = std::make_any<const T*>(&data);

    element.write = [element](hid_t group_id)
    {
        const T &data = *std::any_cast<const T*>(element.data);
        if constexpr(HDF5Utils::IsContainer<T>::value)
        {
            HDF5Writer_detail::WriteContainerData(group_id, element.name, data);
        }
        else
        {
            HDF5Writer_detail::WriteScalarData(group_id, element.name, data);
        }
    };

    if(write)
    {
        HDF5Utils::HID group(HDF5Utils::openGroupPath(this->file_, element.groupPath, true));
        element.write(group);
    }
    else
    {
        this->data.insert(element);
    }
}

template<typename Container>
void HDF5Writer::WriteSlice(const std::string &path, const Container &data, size_t count)
{
    using T = typename Container::value_type;
    static_assert(!HDF5Utils::IsContainer<T>::value,
                  "WriteSlice only supports flat containers (no nested vectors)");

    auto [groupPath, name] = HDF5Utils::splitPathAndName(path);
    HDF5Utils::HID group(HDF5Utils::openGroupPath(this->file_, groupPath, true));

    hsize_t dims[] = {static_cast<hsize_t>(count)};
    HDF5Utils::HID dataspace(H5Screate_simple(1, dims, nullptr));

    if constexpr(std::is_same_v<T, std::string>)
    {
        hid_t strType = HDF5Utils::HDF5Type<std::string>::value();
        HDF5Utils::HID dataset(H5Dcreate2(group, name.c_str(), strType, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if(count > 0)
        {
            std::vector<const char*> cstrs(count);
            for (size_t i = 0; i < count; i++)
                cstrs[i] = data[i].c_str();
            H5Dwrite(dataset, strType, H5S_ALL, H5S_ALL, H5P_DEFAULT, cstrs.data());
        }
    }
    else
    {
        hid_t mem_type;
        if constexpr(HDF5Utils::HasCompType<T>::value)
        {
            mem_type = HDF5Utils::CompTypeCreator<T>::get();
        }
        else
        {
            mem_type = HDF5Utils::HDF5Type<T>::value();
        }

        hid_t file_type = mem_type;
        HDF5Utils::HID packed_type;
        if constexpr(HDF5Utils::HasCompType<T>::value)
        {
            packed_type = HDF5Utils::HID(H5Tcopy(mem_type));
            H5Tpack(packed_type);
            file_type = packed_type;
        }

        HDF5Utils::HID dataset(H5Dcreate2(group, name.c_str(), file_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if(count > 0)
        {
            H5Dwrite(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        }
    }
}

#endif // HDF5WRITER_HPP
