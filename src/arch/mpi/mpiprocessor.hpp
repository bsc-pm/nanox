/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_MPI_PROCESSOR
#define _NANOS_MPI_PROCESSOR

#include "mpi.h"
#include "atomic_decl.hpp"
#include "mpiprocessor_decl.hpp"
#include "config.hpp"
#include "mpidevice.hpp"
#include "mpithread.hpp"
#include "cachedaccelerator.hpp"
#include "copydescriptor_decl.hpp"
#include "processingelement.hpp"
#include <fcntl.h>
#include <unistd.h>

using namespace nanos;
using namespace ext;
    
size_t MPIProcessor::_workers_per_process=0;

System::CachePolicyType MPIProcessor::_cachePolicy = System::WRITE_THROUGH;
size_t MPIProcessor::_cacheDefaultSize = (size_t) -1;
size_t MPIProcessor::_alignThreshold = 128;
size_t MPIProcessor::_alignment = 4096;
size_t MPIProcessor::_maxWorkers = 1;
std::string MPIProcessor::_mpiExecFile;
std::string MPIProcessor::_mpiLauncherFile=NANOX_PREFIX"/bin/offload_slave_launch.sh";
std::string MPIProcessor::_mpiNodeType;
std::string MPIProcessor::_mpiHosts;
std::string MPIProcessor::_mpiHostsFile;
std::string MPIProcessor::_mpiControlFile;
int MPIProcessor::_numPrevPEs=-1;
int MPIProcessor::_numFreeCores;
int MPIProcessor::_currPE;
bool MPIProcessor::_useMultiThread=false;
bool MPIProcessor::_allocWide=false;
#ifndef OPEN_MPI
bool MPIProcessor::_disableSpawnLock=false;

bool MPIProcessor::isDisableSpawnLock() {
    return _disableSpawnLock;
}
#endif

size_t MPIProcessor::getCacheDefaultSize() {
    return _cacheDefaultSize;
}

System::CachePolicyType MPIProcessor::getCachePolicy() {
    return _cachePolicy;
}

std::string MPIProcessor::getMpiHosts() {
    return _mpiHosts;
}

std::string MPIProcessor::getMpiHostsFile() {
    return _mpiHostsFile;
}

std::string MPIProcessor::getMpiExecFile() {
    return _mpiExecFile;
}

std::string MPIProcessor::getMpiControlFile() {
    return _mpiControlFile;
}

bool MPIProcessor::getAllocWide() {
    return _allocWide;
}

size_t MPIProcessor::getMaxWorkers() {
    return _maxWorkers;
}

bool MPIProcessor::isUseMultiThread() {
    return _useMultiThread;
}

std::string MPIProcessor::getMpiLauncherFile() {
    return _mpiLauncherFile;
}

size_t MPIProcessor::getAlignment() {
    return _alignment;
}

size_t MPIProcessor::getAlignThreshold() {
    return _alignThreshold;
}

MPI_Comm MPIProcessor::getCommunicator() const {
    return _communicator;
}

void MPIProcessor::setCommunicator( MPI_Comm comm ) {
    _communicator = comm;
}

MPI_Comm MPIProcessor::getCommOfParents() const {
    return _commOfParents;
}

int MPIProcessor::getRank() const {
    return _rank;
}

bool MPIProcessor::getOwner() const {
    return _owner;
}

void MPIProcessor::setOwner(bool owner) {
    _owner=owner;
}

bool MPIProcessor::getHasWorkerThread() const {
    return _hasWorkerThread;
}

void MPIProcessor::setHasWorkerThread(bool hwt) {
    _hasWorkerThread=hwt;
}

bool MPIProcessor::getShared() const {
    return _shared;
}     

WD* MPIProcessor::getCurrExecutingWd() const {
    return _currExecutingWd;
}

void MPIProcessor::setCurrExecutingWd(WD* currExecutingWd) {
    this->_currExecutingWd = currExecutingWd;
}

bool MPIProcessor::isBusy() {
    return _busy.load();
}

void MPIProcessor::setPphList(int* list){
    _pphList=list;
}

int* MPIProcessor::getPphList() {
    return _pphList;
}

mpi::persistent_request& MPIProcessor::getTaskEndRequest() {
    return _taskEndRequest;
}

//Try to reserve this PE, if the one who reserves it is the same
//which already has the PE, return true
bool MPIProcessor::acquire( int dduid ) {
    if( dduid == _currExecutingDD )
        return true;

    bool idle = !_busy.test_and_set();
    return idle;
}     

void MPIProcessor::release() {
   _busy.clear();
}

int MPIProcessor::getCurrExecutingDD() const {
    return _currExecutingDD;
}

void MPIProcessor::setCurrExecutingDD(int currExecutingDD) {
    this->_currExecutingDD = currExecutingDD;
}

void MPIProcessor::appendToPendingRequests( mpi::request const& req ) {
    _pendingReqs.push_back(req);
}

#endif
