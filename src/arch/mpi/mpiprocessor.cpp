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

#include "mpiprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include "mpithread.hpp"
#include "smpprocessor.hpp"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <mpi.h>

using namespace nanos;
using namespace nanos::ext;

MPIProcessor::MPIProcessor( void* communicator, int rank, int uid, bool owner, bool shared, 
        MPI_Comm communicatorOfParents, SMPProcessor* core, memory_space_id_t memId ) : 
ProcessingElement( &MPI, memId, rank /*node id*/, 0 /* TODO: see clusternode.cpp */, true, 0, false ), _pendingReqs(), _core(core), _peLock() {
    _communicator = *((MPI_Comm *)communicator);
    _commOfParents=communicatorOfParents;
    _rank = rank;
    _owner=owner;
    _shared=shared;
    _currExecutingWd=NULL;
    _busy=false;
    _currExecutingDD=0;
    _hasWorkerThread=false;
    _pphList=NULL;
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
    
    #ifndef OPEN_MPI
    config.registerConfigOption("offl-disable-spawn-lock", NEW Config::BoolVar(_disableSpawnLock), "Disables file lock used to serialize"
                  "different calls to DEEP_BOOSTER_ALLOC performed by different processes when using MPI_COMM_SELF,"
                  "if disabled user will be responsible of serializing calls (Default: Lock enabled).");
    config.registerArgOption("offl-disable-spawn-lock", "offl-disable-spawn-lock");
    config.registerEnvOption("offl-disable-spawn-lock", "NX_OFFL_DISABLE_SPAWN_LOCK");
    #endif

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
        _peLock.acquire();
        if (!_pendingReqs.empty()) {
            completed = mpi::request::test_all( _pendingReqs.begin(), _pendingReqs.end() );
            if ( completed ) {
                _pendingReqs.clear();
            }
        }
        _peLock.release();
    }
    return completed;
}

BaseThread &MPIProcessor::startMPIThread(WD* wd) {
   if (wd==NULL) wd=&getWorkerWD();
    
   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) wd->getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = wd->getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   return _core->startThread( *this, *wd, NULL );
}
