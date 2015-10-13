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

#include "plugin.hpp"
#include "mpidd.hpp"
#include "system.hpp"

namespace nanos {
namespace ext {

class MPIPlugin : public ArchPlugin
{
   //The three boleans below implement the initialization order
   //First system will "pre-initialize" before initializing threads (not if extrae enabled and only when we are slaves)
   //Then extrae will initialize
   //Then system will "post-initialice" as part of user main (ompss_nanox_main)
   bool _extraeInitialized;
   bool _initialized;
   bool _preinitialized;
   static Atomic<unsigned int> _numWorkers;
   static Atomic<unsigned int> _numPEs;
   
   public:
    MPIPlugin() : ArchPlugin( "MPI PE Plugin",1 ), _extraeInitialized(false),_initialized(false), _preinitialized(false) {}

    virtual void config ( Config& cfg )
    {
       cfg.setOptionsSection( "Offload Arch", "Offload specific options" );
       MPIProcessor::prepareConfig( cfg );
    }

    virtual bool configurable() { 
        char *offload_trace_on = getenv(const_cast<char*> ("NX_OFFLOAD_INSTRUMENTATION")); 
        char* isSlave = getenv(const_cast<char*> ("OMPSS_OFFLOAD_SLAVE"));
        //Non-slaves do not preInitialize
        return ( _preinitialized || !isSlave ) && ( offload_trace_on == NULL || _extraeInitialized );
    }    

    virtual void init() {    
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
       }
    }

    static void addPECount(unsigned int count) {
        _numPEs+=count;
    }

    static void addWorkerCount(unsigned int count) {
        _numWorkers+=count;
    }

    virtual unsigned getNumThreads() const
    {
         return 0;
    }

    virtual unsigned int getNumPEs() const {
       return _numPEs.value();
    }
    virtual unsigned int getMaxPEs() const {
       return 0;
    }
    virtual unsigned int getNumWorkers() const {
       return _numWorkers.value();
    }
    virtual unsigned int getMaxWorkers() const {
       return 0;
    }

    virtual void createBindingList()
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
       
    virtual void addPEs( std::map<unsigned int, ProcessingElement *> &pes ) const {
    }

    virtual void addDevices( DeviceList &devices ) const {}

    virtual void startSupportThreads() {
    }

    virtual void startWorkerThreads( std::map<unsigned int, BaseThread *> &workers ) {
    }
    
    virtual void finalize() {
    }
      

    virtual PE* createPE( unsigned id, unsigned uid )
    {
       //Not used
       return NULL;
    }
};
}
}

Atomic<unsigned int> nanos::ext::MPIPlugin::_numWorkers=0;
Atomic<unsigned int> nanos::ext::MPIPlugin::_numPEs=0;
DECLARE_PLUGIN("arch-mpi",nanos::ext::MPIPlugin);
