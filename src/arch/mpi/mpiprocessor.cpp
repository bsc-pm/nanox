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

#include "mpiprocessor.hpp"

#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include "mpithread.hpp"
#include "mpiremotenode_decl.hpp"
#include "smpprocessor.hpp"

#include "createauxthread.hpp"

#include <iostream>
#include <fstream>

#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>

using namespace nanos;
using namespace nanos::ext;

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

MPIProcessor::MPIProcessor( MPI_Comm communicator, int rank, bool owner,
        MPI_Comm communicatorOfParents, SMPProcessor* core, memory_space_id_t memId ) :
    ProcessingElement( &MPI, memId, rank /*node id*/, 0 /* TODO: see clusternode.cpp */, true, 0, false ),
    _communicator( communicator ),
    _rank( rank ),
    _owner( owner ),
    _hasWorkerThread(false),
    _pphList(NULL),
    _busy(),
    _currExecutingWd(NULL),
    _currExecutingFunctionId(-1),
    _currExecutingDD(0),
    _pendingReqs(),
    _taskEndRequest(),
    _commOfParents( communicatorOfParents ),
    _core(core),
    _peLock()
{
    // Create taskEnd reception persistent request
    _busy.clear();
    MPI_Recv_init( &_currExecutingFunctionId, 1, MPI_INT, _rank, TAG_END_TASK, _communicator, _taskEndRequest );

    // Synchronize linker arrays
    if( owner ) {
        int arrSize = 0;
        while( ompss_mpi_masks[arrSize] == MASK_TASK_NUMBER ) {
            arrSize++;
        }

	_pendingReqs.push_back( mpi::request() );
        MPI_Isend(ompss_mpi_filenames,  arrSize, MPI_UNSIGNED, rank, TAG_FP_NAME_SYNC, _communicator, _pendingReqs.back() );

	_pendingReqs.push_back( mpi::request() );
        MPI_Isend(ompss_mpi_file_sizes, arrSize, MPI_UNSIGNED, rank, TAG_FP_SIZE_SYNC, _communicator, _pendingReqs.back() );

        //If user defined multithread cache behaviour, send the creation order
        if (nanos::ext::MPIProcessor::isUseMultiThread()) {
            mpi::command::CreateAuxiliaryThread::Requestor createThread( *this );
            createThread.dispatch();

            setHasWorkerThread(true);
        }
        waitAndClearRequests();
    }
}


MPIProcessor::~MPIProcessor() {
    // Free taskEnd reception persistent request
    _taskEndRequest.free();
}

void MPIProcessor::prepareConfig(Config &config) {

    config.registerConfigOption("offl-exec", NEW Config::StringVar(_mpiExecFile), "Defines executable path (in child nodes) used in DEEP_Booster_Alloc");
    config.registerArgOption("offl-exec", "offl-exec");
    config.registerEnvOption("offl-exec", "NX_OFFL_EXEC");

    config.registerConfigOption("offl-launcher", NEW Config::StringVar(_mpiLauncherFile), "Defines launcher script path (in child nodes) used in DEEP_Booster_Alloc");
    config.registerArgOption("offl-launcher", "offl-launcher");
    config.registerEnvOption("offl-launcher", "NX_OFFL_LAUNCHER");

    config.registerConfigOption("offl-nodetype", NEW Config::StringVar(_mpiNodeType), "Defines which type of nodes will be used for Offload. "
                                                                                      "Only applies for Parastation MPI. Values: {cluster, booster}");
    config.registerArgOption("offl-nodetype", "offl-nodetype");
    config.registerEnvOption("offl-nodetype", "NX_OFFL_NODETYPE");


    config.registerConfigOption("offl-hostfile", NEW Config::StringVar(_mpiHostsFile), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\nThe format of the file is: "
    "One host per line with blank lines and lines beginning with # ignored\n"
    "Multiple processes per host can be specified by specifying the host name as follows: hostA:n\n"
    "Environment variables for the host can be specified separated by comma using hostA:n<env_var1,envar2... or hostA<env_var1,envar2...");
    config.registerArgOption("offl-hostfile", "offl-hostfile");
    config.registerEnvOption("offl-hostfile", "NX_OFFL_HOSTFILE");

    config.registerConfigOption("offl-hosts", NEW Config::StringVar(_mpiHosts), "Defines hosts list where secondary process can spawn in DEEP_Booster_Alloc\n Same format than NX_OFFLHOSTFILE but in a single line and separated with \';\'\nExample: hostZ hostA<env_vars hostB:2<env_vars hostC:3 hostD:4");
    config.registerArgOption("offl-hosts", "offl-hosts");
    config.registerEnvOption("offl-hosts", "NX_OFFL_HOSTS");


    config.registerConfigOption("offl-controlfile", NEW Config::StringVar(_mpiControlFile), "Defines a shared (GPFS or similar) file which will be used "
                                 " to automatically manage offload hosts (round robin). This means that each alloc will consume hosts, so future allocs"
                                 " do not oversubscribe on the same host.");
    config.registerArgOption("offl-controlfile", "offl-controlfile");
    config.registerEnvOption("offl-controlfile", "NX_OFFL_CONTROLFILE");


    // Set the cache policy for MPI devices
//    System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig(_cachePolicy);
//    cachePolicyCfg->addOption("wt", System::WRITE_THROUGH);
//    cachePolicyCfg->addOption("wb", System::WRITE_BACK);
//    cachePolicyCfg->addOption("nocache", System::NONE);
//    config.registerConfigOption("offl-cache-policy", cachePolicyCfg, "Defines the cache policy for offload architectures: write-through / write-back (wb by default)");
//    config.registerEnvOption("offl-cache-policy", "NX_OFFL_CACHE_POLICY");
//    config.registerArgOption("offl-cache-policy", "offl-cache-policy");


//    config.registerConfigOption("offl-cache-size", NEW Config::SizeVar(_cacheDefaultSize), "Defines maximum possible size of the cache for MPI allocated devices (Default: 144115188075855871)");
//    config.registerArgOption("offl-cache-size", "offl-cache-size");
//    config.registerEnvOption("offl-cache-size", "NX_OFFLCACHESIZE");


//    config.registerConfigOption("offl-buffer-size", NEW Config::SizeVar(_bufferDefaultSize), "Defines size of the nanox MPI Buffer (MPI_Buffer_Attach/detach)");
//    config.registerArgOption("offl-buffer-size", "offl-buffer-size");
//    config.registerEnvOption("offl-buffer-size", "NX_OFFLBUFFERSIZE");

    config.registerConfigOption("offl-align-threshold", NEW Config::SizeVar(_alignThreshold), "Defines minimum size (bytes) which determines if offloaded variables (copy_in/out) will be aligned (default value: 128), arrays with size bigger or equal than this value will be aligned when offloaded");
    config.registerArgOption("offl-align-threshold", "offl-align-threshold");
    config.registerEnvOption("offl-align-threshold", "NX_OFFL_ALIGNTHRESHOLD");

    config.registerConfigOption("offl-alignment", NEW Config::SizeVar(_alignment), "Defines the alignment (bytes) applied to offloaded variables (copy_in/out) (default value: 4096)");
    config.registerArgOption("offl-alignment", "offl-alignment");
    config.registerEnvOption("offl-alignment", "NX_OFFL_ALIGNMENT");

    config.registerConfigOption("offl-workers", NEW Config::SizeVar(_maxWorkers), "Defines the maximum number of worker threads created per alloc (Default: 1) ");
    config.registerArgOption("offl-workers", "offl-max-workers");
    config.registerEnvOption("offl-workers", "NX_OFFL_MAX_WORKERS");

    config.registerConfigOption("offl-cache-threads", NEW Config::BoolVar(_useMultiThread), "Defines if offload processes will have an extra cache thread,"
        " this is good for applications which need data from other tasks so they don't have to wait until task in owner node finishes. "
        "(Default: False, but if this kind of behaviour is detected, the thread will be created)");
    config.registerArgOption("offl-cache-threads", "offl-cache-threads");
    config.registerEnvOption("offl-cache-threads", "NX_OFFL_CACHE_THREADS");


    config.registerConfigOption( "offl-alloc-wide", NEW Config::FlagOption( _allocWide ),
                                "Alloc full objects in the cache." );
    config.registerEnvOption( "offl-alloc-wide", "NX_OFFL_ENABLE_ALLOCWIDE" );
    config.registerArgOption( "offl-alloc-wide", "offl-enable-alloc-wide" );

    config.registerConfigOption("offl-workers", NEW Config::SizeVar(_workers_per_process, 0), "Specifies how many worker threads per spawned process should be created.");
    config.registerArgOption("offl-workers", "offl-workers");
    config.registerEnvOption("offl-workers", "OFFL_NX_SMP_WORKERS");

    config.registerAlias( "offl-workers", "offl-num-threads","Specifies how many worker threads per spawned process should be created.");
    config.registerEnvOption("offl-num-threads", "OFFL_OMP_NUM_THREADS");
}

WorkDescriptor & MPIProcessor::getWorkerWD() const {
    MPIDD * dd = NEW MPIDD((MPIDD::work_fct)Scheduler::workerLoop);
    WD *wd = NEW WD(dd);
    return *wd;
}

WorkDescriptor & MPIProcessor::getMasterWD() const {
    WD * wd = NEW WD(NEW MPIDD());
    return *wd;
}

BaseThread &MPIProcessor::createThread(WorkDescriptor &helper, SMPMultiThread *parent ) {
     MPIThread &th = *NEW MPIThread(helper, this, _core);

     return th;
}

void MPIProcessor::waitAllRequests() {
    nanos::mpi::request::wait_all( _pendingReqs.begin(), _pendingReqs.end() );
}

void MPIProcessor::clearAllRequests() {
    _pendingReqs.clear();
}

void MPIProcessor::waitAndClearRequests() {
    waitAllRequests();
    clearAllRequests();
}

bool MPIProcessor::testAllRequests() {
    //Wait and clear for all the requests
    bool completed = false;
    if (!_pendingReqs.empty()) {
        UniqueLock<Lock> guard( _peLock );
        if (!_pendingReqs.empty()) {
            completed = mpi::request::test_all( _pendingReqs.begin(), _pendingReqs.end() );
            if ( completed ) {
                _pendingReqs.clear();
            }
        }
    }
    return completed;
}

MPIThread& MPIProcessor::startMPIThread(WD* wd ) {
   if ( wd == NULL )
      wd = &getWorkerWD();

   NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) wd->getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = wd->getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   return static_cast<MPIThread&>(_core->startThread( *this, *wd, NULL ));
}

WD* MPIProcessor::freeCurrExecutingWd() {

    WD* wd = getCurrExecutingWd();
    setCurrExecutingWd(NULL);

    //Clear all async requests on this PE (they finished a while ago)
    waitAndClearRequests();

    //Set the PE as free so we can re-schedule work to it
    release();

    return wd;
}

