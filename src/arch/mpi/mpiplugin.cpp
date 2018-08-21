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

#include "mpiplugin.hpp"
#include "mpidd.hpp"
#include "system.hpp"

#include "mpiprocessor_decl.hpp"
#include "mpiremotenode_decl.hpp"

#include <iostream>

namespace nanos {
namespace ext {

Atomic<unsigned int> MPIPlugin::_numWorkers;
Atomic<unsigned int> MPIPlugin::_numPEs;

void MPIPlugin::config ( Config& cfg )
{
   cfg.setOptionsSection( "Offload Arch", "Offload specific options" );
   MPIProcessor::prepareConfig( cfg );
}

bool MPIPlugin::configurable() {
    char *offload_trace_on = getenv(const_cast<char*> ("NX_OFFLOAD_INSTRUMENTATION"));
    char* isSlave = getenv(const_cast<char*> ("OMPSS_OFFLOAD_SLAVE"));
    //Non-slaves do not preInitialize
    return ( _preinitialized || !isSlave ) && ( offload_trace_on == NULL || _extraeInitialized );
}

void MPIPlugin::init() {
   char *offload_trace_on = getenv(const_cast<char*> ("NX_OFFLOAD_INSTRUMENTATION"));
   char* isSlave = getenv(const_cast<char*> ("OMPSS_OFFLOAD_SLAVE"));

   if ( !_preinitialized && isSlave )
   {
      _preinitialized=true;
      //Do not initialize if we have extrae or we are not slaves
      if( offload_trace_on == NULL ) nanos::ext::MPIRemoteNode::preInit();
      return;
   }

   //If we have extrae, initialize it
   if (offload_trace_on != NULL && !_extraeInitialized){
      _extraeInitialized=true;
      if (getenv("I_MPI_WAIT_MODE")==NULL) putenv( const_cast<char*> ("I_MPI_WAIT_MODE=1"));
      int provided;
      MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);
      return;
   }

   if (!_initialized && isSlave) {
      _initialized=true;
      //Extrae "mode" only initializes MPI, initialize everything else now
      //Do not initialize if we have extrae or we are not slaves
      if (_extraeInitialized) nanos::ext::MPIRemoteNode::preInit();
      nanos::ext::MPIRemoteNode::mpiOffloadSlaveMain();
      // After slave main finished we must exit the application
      // Otherwise 'main()' will be executed.
      exit(0);
   }
}

void MPIPlugin::addPECount(unsigned int count) {
    _numPEs+=count;
}

void MPIPlugin::addWorkerCount(unsigned int count) {
    _numWorkers+=count;
}

unsigned MPIPlugin::getNumThreads() const
{
     return 0;
}

unsigned int MPIPlugin::getNumPEs() const {
   return _numPEs.value();
}
unsigned int MPIPlugin::getMaxPEs() const {
   return 0;
}
unsigned int MPIPlugin::getNumWorkers() const {
   return _numWorkers.value();
}
unsigned int MPIPlugin::getMaxWorkers() const {
   return 0;
}

void MPIPlugin::createBindingList()
{
//        /* As we now how many devices we have and how many helper threads we
//         * need, reserve a PE for them */
//        for ( unsigned i = 0; i < OpenCLConfig::getOpenCLDevicesCount(); ++i )
//        {
//           // TODO: if HWLOC is available, use it.
//           int node = sys.getNumSockets() - 1;
//           unsigned pe = sys.reservePE( node );
//           // Now add this node to the binding list
//           addBinding( pe );
//        }
}

void MPIPlugin::addPEs( PEMap &pes ) const {
}

void MPIPlugin::addDevices( DeviceList &devices ) const {}

void MPIPlugin::startSupportThreads() {
}

void MPIPlugin::startWorkerThreads( std::map<unsigned int, BaseThread *> &workers ) {
}

void MPIPlugin::finalize() {
}


PE* MPIPlugin::createPE( unsigned id, unsigned uid )
{
   //Not used
   return NULL;
}

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN("arch-mpi",nanos::ext::MPIPlugin);

