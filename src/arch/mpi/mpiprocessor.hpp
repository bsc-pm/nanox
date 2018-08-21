/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef _NANOS_MPI_PROCESSOR
#define _NANOS_MPI_PROCESSOR

#include "mpiprocessor_decl.hpp"

namespace nanos {
namespace ext {

inline size_t MPIProcessor::getCacheDefaultSize() {
    return _cacheDefaultSize;
}

inline System::CachePolicyType MPIProcessor::getCachePolicy() {
    return _cachePolicy;
}

inline std::string MPIProcessor::getMpiHosts() {
    return _mpiHosts;
}

inline std::string MPIProcessor::getMpiHostsFile() {
    return _mpiHostsFile;
}

inline std::string MPIProcessor::getMpiExecFile() {
    return _mpiExecFile;
}

inline std::string MPIProcessor::getMpiControlFile() {
    return _mpiControlFile;
}

inline bool MPIProcessor::getAllocWide() {
    return _allocWide;
}

inline size_t MPIProcessor::getMaxWorkers() {
    return _maxWorkers;
}

inline bool MPIProcessor::isUseMultiThread() {
    return _useMultiThread;
}

inline std::string MPIProcessor::getMpiLauncherFile() {
    return _mpiLauncherFile;
}

inline size_t MPIProcessor::getAlignment() {
    return _alignment;
}

inline size_t MPIProcessor::getAlignThreshold() {
    return _alignThreshold;
}

inline MPI_Comm MPIProcessor::getCommunicator() const {
    return _communicator;
}

inline void MPIProcessor::setCommunicator( MPI_Comm comm ) {
    _communicator = comm;
}

inline MPI_Comm MPIProcessor::getCommOfParents() const {
    return _commOfParents;
}

inline int MPIProcessor::getRank() const {
    return _rank;
}

inline bool MPIProcessor::isOwner() const {
    return _owner;
}

inline bool MPIProcessor::getHasWorkerThread() const {
    return _hasWorkerThread;
}

inline void MPIProcessor::setHasWorkerThread(bool hwt) {
    _hasWorkerThread=hwt;
}

inline WD* MPIProcessor::getCurrExecutingWd() const {
    return _currExecutingWd;
}

inline void MPIProcessor::setCurrExecutingWd(WD* currExecutingWd) {
    this->_currExecutingWd = currExecutingWd;
}

inline bool MPIProcessor::isBusy() {
    return _busy.load();
}

inline void MPIProcessor::setPphList(int* list){
    _pphList=list;
}

inline int* MPIProcessor::getPphList() {
    return _pphList;
}

inline mpi::persistent_request& MPIProcessor::getTaskEndRequest() {
    return _taskEndRequest;
}

//Try to reserve this PE, if the one who reserves it is the same
//which already has the PE, return true
inline bool MPIProcessor::acquire( int dduid ) {
    if( _busy.load() && dduid == _currExecutingDD )
        return true;

    bool idle = !_busy.test_and_set();
    if( idle )
        _currExecutingDD = dduid;

    return idle;
}

inline void MPIProcessor::release() {
   _busy.clear();
}

inline int MPIProcessor::getCurrExecutingDD() const {
    return _currExecutingDD;
}

inline void MPIProcessor::setCurrExecutingDD(int currExecutingDD) {
    this->_currExecutingDD = currExecutingDD;
}

inline void MPIProcessor::appendToPendingRequests( mpi::request const& req ) {
    _pendingReqs.push_back(req);
}

} // namespace ext
} // namespace nanos

#endif
