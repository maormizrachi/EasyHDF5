#include "HDF5ReaderParallel.hpp"

#ifdef H5_HAVE_PARALLEL

HDF5ReaderParallel::HDF5ReaderParallel(const std::string &filename, MPI_Comm comm)
{
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    prefix_ = "/rank" + std::to_string(rank_);

    HDF5Utils::HID fapl(H5Pcreate(H5P_FILE_ACCESS));
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);

    file_id_ = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl);
}

HDF5ReaderParallel::~HDF5ReaderParallel()
{
    Close();
}

void HDF5ReaderParallel::Close()
{
    if(!closed_ && file_id_ >= 0)
    {
        H5Fclose(file_id_);
        file_id_ = H5I_INVALID_HID;
        closed_ = true;
    }
}

static herr_t collectGroupNamesCallback(hid_t /*group*/, const char *name, const H5L_info_t * /*info*/, void *op_data)
{
    auto *names = static_cast<std::vector<std::string>*>(op_data);
    names->push_back(std::string(name));
    return 0;
}

bool HDF5ReaderParallel::Exists(const std::string &path) const
{
    std::string fullPath = prefix_ + "/" + path;

    H5E_auto2_t old_func;
    void *old_client_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    htri_t result = H5Lexists(file_id_, fullPath.c_str(), H5P_DEFAULT);

    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    return result > 0;
}

bool HDF5ReaderParallel::ExistsGlobal(const std::string &path) const
{
    H5E_auto2_t old_func;
    void *old_client_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    htri_t result = H5Lexists(file_id_, path.c_str(), H5P_DEFAULT);

    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    return result > 0;
}

std::vector<std::string> HDF5ReaderParallel::ReadGroupNames(const std::string &path) const
{
    std::string fullPath = prefix_ + "/" + path;
    HDF5Utils::HID group(HDF5Utils::openGroupPath(file_id_, fullPath));
    std::vector<std::string> names;
    hsize_t idx = 0;
    H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, &idx, collectGroupNamesCallback, &names);
    return names;
}

#endif // H5_HAVE_PARALLEL
