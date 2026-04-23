#include "HDF5WriterParallel.hpp"

#ifdef H5_HAVE_PARALLEL

HDF5WriterParallel::HDF5WriterParallel(const std::string &filename, MPI_Comm comm, bool truncate)
{
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    prefix_ = "/rank" + std::to_string(rank_);

    HDF5Utils::HID fapl(H5Pcreate(H5P_FILE_ACCESS));
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);

    if(truncate)
    {
        file_id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    }
    else
    {
        file_id_ = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl);
    }
}

HDF5WriterParallel::~HDF5WriterParallel()
{
    Close();
}

void HDF5WriterParallel::Close()
{
    if(!closed_ && file_id_ >= 0)
    {
        H5Fclose(file_id_);
        file_id_ = H5I_INVALID_HID;
        closed_ = true;
    }
}

#endif // H5_HAVE_PARALLEL
