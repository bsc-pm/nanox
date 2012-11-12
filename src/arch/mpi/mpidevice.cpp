/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "mpi.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "mpidd.hpp"
#include "mpiprocessor.hpp"
#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"
#include "mpidevice.hpp"


//TODO: check for errors in communications
using namespace nanos;


MPI_Datatype MPIDevice::cacheStruct=false;

MPIDevice::MPIDevice(const char *n) : Device(n) {
    if (cacheStruct != NULL) {
        MPI_Datatype typelist[4] = {MPI_SHORT, MPI_UNSIGNED_LONG_LONG,MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
        int blocklen[4] = {1, 1, 1, 1};
        MPI_Aint disp[4] = {offsetof(cacheOrder, opId), offsetof(cacheOrder, hostAddr), offsetof(cacheOrder, devAddr), offsetof(cacheOrder, size)};
        MPI_Type_create_struct(3, blocklen, disp, typelist, &cacheStruct);
        MPI_Type_commit(&cacheStruct);
    }
}

/*! \brief MPIDevice copy constructor
 */
MPIDevice::MPIDevice(const MPIDevice &arch) : Device(arch) {
    if (cacheStruct != NULL) {
        MPI_Datatype typelist[4] = {MPI_SHORT, MPI_UNSIGNED_LONG_LONG,MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
        int blocklen[4] = {1, 1, 1, 1};
        MPI_Aint disp[4] = {offsetof(cacheOrder, opId), offsetof(cacheOrder, hostAddr), offsetof(cacheOrder, devAddr), offsetof(cacheOrder, size)};
        MPI_Type_create_struct(3, blocklen, disp, typelist, &cacheStruct);
        MPI_Type_commit(&cacheStruct);
    }
}

/*! \brief MPIDevice destructor
 */
MPIDevice::~MPIDevice() {
};

/* \breif allocate size bytes in the device
 */
void * MPIDevice::allocate(size_t size, ProcessingElement *pe) {
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor * ) pe;
    cacheOrder order;
    order.opId = OPID_ALLOCATE;
    order.hostAddr = 0;
    order.size = size;
    MPI_Status status;
    MPI_Send(&order, sizeof (order), cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    MPI_Recv(&order, sizeof (order), cacheStruct, myPE->_rank, TAG_CACHE_ANSWER, myPE->_communicator, &status);
    return (void *) order.devAddr;
}

/* \brief free address
 */
void MPIDevice::free(void *address, ProcessingElement *pe) {
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_FREE;
    order.devAddr = (uint64_t) address;
    order.size = 0;
    MPI_Send(&order, sizeof (order), cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
}

/* \brief Reallocate and copy from address.
 */
void * MPIDevice::realloc(void *address, size_t size, size_t old_size, ProcessingElement *pe) {
    free(address, pe);
    return allocate(size, pe);
}

/* \brief Copy from remoteSrc in the host to localDst in the device
 *        Returns true if the operation is synchronous
 */
bool MPIDevice::copyIn(void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe) {
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYIN;
    order.devAddr = (uint64_t) localDst;
    order.hostAddr = (uint64_t) remoteSrc.getTag();
    order.size = size;
    MPI_Send(&order, sizeof (order), cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    return true;
}

/* \brief Copy from localSrc in the device to remoteDst in the host
 *        Returns true if the operation is synchronous
 */
bool MPIDevice::copyOut(CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe) {
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYOUT;
    order.devAddr = (uint64_t) localSrc;
    order.hostAddr = (uint64_t) remoteDst.getTag();
    order.size = size;
    MPI_Status status;
    MPI_Send(&order, sizeof (order), cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    MPI_Recv((void*) order.hostAddr, order.size, MPI_BYTE, myPE->_rank, TAG_CACHE_DATA, myPE->_communicator, &status);
    return true;
}

/* \brief Copy localy in the device from src to dst
 */
void MPIDevice::copyLocal(void *dst, void *src, size_t size, ProcessingElement *pe) {
    //HostAddr will be src and DevAddr will be dst (both are device addresses)
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYLOCAL;
    order.devAddr = (uint64_t) dst;
    order.hostAddr = (uint64_t) src;
    order.size = size;
    MPI_Status status;
    MPI_Send(&order, sizeof (order), cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
}

void MPIDevice::syncTransfer(uint64_t hostAddress, ProcessingElement *pe) {
}

bool MPIDevice::copyDevToDev(void * addrDst, CopyDescriptor& cdDst, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc) {
    return true;
}
