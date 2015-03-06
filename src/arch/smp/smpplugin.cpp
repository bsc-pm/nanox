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

#include "smpbaseplugin_decl.hpp"
#include "plugin.hpp"
#include "smpprocessor.hpp"
#include "smpdd.hpp"
#include "os.hpp"
#include "osallocator_decl.hpp"
#include "system.hpp"
#include "basethread.hpp"
#include "printbt_decl.hpp"
#include <limits>

#ifdef NANOX_MEMKIND_SUPPORT
#include <memkind.h>
#endif

//#include <numa.h>

namespace nanos {
namespace ext {

nanos::PE * smpProcessorFactory ( int id, int uid );

nanos::PE * smpProcessorFactory ( int id, int uid )
{
   return NULL;//NEW SMPProcessor( );
}

class SMPPlugin : public SMPBasePlugin
{

   //! CPU id binding list
   typedef std::vector<int> Bindings;

   Atomic<unsigned int>         _idSeed;
   int                          _requestedCores;
   int                          _requestedCoresByMask;
   int                          _availableCores;
   int                          _currentCores;
   int                          _requestedWorkers;
   int                          _requestedWorkersOMPSS;
   std::vector<SMPProcessor *> *_cpus;
   std::vector<SMPProcessor *> *_cpusByCpuId;
   std::vector<SMPThread *>     _workers;
   int                          _bindingStart;
   int                          _bindingStride;
   bool                         _bindThreads;
   bool                         _smpPrivateMemory;
   bool                         _smpAllocWide;
   int                          _smpHostCpus;
   std::size_t                  _smpPrivateMemorySize;
   bool                         _workersCreated;
   unsigned int                 _numWorkers; //must be updated if the number of workers increases after calling startWorkerThreads

   // Nanos++ scheduling domain
   cpu_set_t                    _cpuSet;          /*!< \brief system's default cpu_set */
   cpu_set_t                    _cpuActiveSet;    /*!< \brief mask of current active cpus */


   //! Physical NUMA nodes
   int                          _numSockets;
   int                          _coresPerSocket;
   //! The socket that will be assigned to the next WD
   int                          _currentSocket;


   //! CPU id binding list
   Bindings                     _bindings;

   bool                         _memkindSupport;
   std::size_t                  _memkindMemorySize;

   public:
   SMPPlugin() : SMPBasePlugin( "SMP PE Plugin", 1 )
                 , _idSeed( 0 )
                 , _requestedCores( 0 )
                 , _requestedCoresByMask( 0 )
                 , _availableCores( 0 )
                 , _currentCores( 0 )
                 , _requestedWorkers( -1 )
                 , _requestedWorkersOMPSS( -1 )
                 , _cpus( NULL )
                 , _cpusByCpuId( NULL )
                 , _workers( 0, (SMPThread *) NULL )
                 , _bindingStart( 0 )
                 , _bindingStride( 1 )
                 , _bindThreads( true )
                 , _smpPrivateMemory( false )
                 , _smpAllocWide( false )
                 , _smpHostCpus( 0 )
                 , _smpPrivateMemorySize( 256 * 1024 * 1024 ) // 256 Mb
                 , _workersCreated( false )
                 , _numWorkers( 0 )
                 , _cpuActiveSet()
                 , _numSockets( 0 )
                 , _coresPerSocket( 0 )
                 , _bindings()
                 , _memkindSupport( false )
                 , _memkindMemorySize( 1024*1024*1024 ) // 1Gb
   {}

   virtual unsigned int getNewSMPThreadId() {
      return _idSeed++;
   }

   virtual void config ( Config& cfg )
   {
      cfg.setOptionsSection( "SMP Arch", "SMP specific options" );
      SMPProcessor::prepareConfig( cfg );
      SMPDD::prepareConfig( cfg );
      cfg.registerConfigOption ( "smp-num-pes", NEW Config::PositiveVar ( _requestedCores ), "Cores requested." );
      cfg.registerArgOption ( "smp-num-pes", "smp-cpus" );
      cfg.registerEnvOption( "smp-num-pes", "NX_SMP_CPUS" );

      cfg.registerConfigOption ( "smp-workers", NEW Config::PositiveVar ( _requestedWorkersOMPSS ), "Worker threads requested." );
      cfg.registerArgOption ( "smp-workers", "smp-workers" );
      cfg.registerEnvOption( "smp-workers", "NX_SMP_WORKERS" );

      cfg.registerConfigOption( "cores-per-socket", NEW Config::PositiveVar( _coresPerSocket ),
            "Number of cores per socket." );
      cfg.registerArgOption( "cores-per-socket", "cores-per-socket" );

      cfg.registerConfigOption( "num-sockets", NEW Config::PositiveVar( _numSockets ),
            "Number of sockets available." );
      cfg.registerArgOption( "num-sockets", "num-sockets" );


      cfg.registerConfigOption( "binding-start", NEW Config::IntegerVar ( _bindingStart ),
            "Set initial cpu id for binding (binding required)" );
      cfg.registerArgOption( "binding-start", "binding-start" );
      cfg.registerEnvOption( "binding-start", "NX_BINDING_START" );

      cfg.registerConfigOption( "binding-stride", NEW Config::IntegerVar ( _bindingStride ),
            "Set binding stride (binding required)" );
      cfg.registerArgOption( "binding-stride", "binding-stride" );
      cfg.registerEnvOption( "binding-stride", "NX_BINDING_STRIDE" );

      cfg.registerConfigOption( "no-binding", NEW Config::FlagOption( _bindThreads, false ),
            "Disables thread binding" );
      cfg.registerArgOption( "no-binding", "disable-binding" );

      cfg.registerConfigOption( "smp-private-memory", NEW Config::FlagOption( _smpPrivateMemory, true ),
            "SMP devices use a private memory area." );
      cfg.registerArgOption( "smp-private-memory", "smp-private-memory" );
      cfg.registerEnvOption( "smp-private-memory", "NX_SMP_PRIVATE_MEMORY" );

      cfg.registerConfigOption( "smp-alloc-wide", NEW Config::FlagOption( _smpAllocWide, true ),
            "SMP devices use a private memory area." );
      cfg.registerArgOption( "smp-alloc-wide", "smp-alloc-wide" );
      cfg.registerEnvOption( "smp-alloc-wide", "NX_SMP_ALLOC_WIDE" );

      cfg.registerConfigOption( "smp-host-cpus", NEW Config::IntegerVar( _smpHostCpus ),
            "When using SMP devices with private memory, set how many CPUs will work with the host memory. Minimum value is 1 (which is also the default)." );
      cfg.registerArgOption( "smp-host-cpus", "smp-host-cpus" );
      cfg.registerEnvOption( "smp-host-cpus", "NX_SMP_HOST_CPUS" );

      cfg.registerConfigOption( "smp-private-memory-size", NEW Config::SizeVar( _smpPrivateMemorySize ),
            "Set the size of SMP devices private memory area." );
      cfg.registerArgOption( "smp-private-memory-size", "smp-private-memory-size" );
      cfg.registerEnvOption( "smp-private-memory-size", "NX_SMP_PRIVATE_MEMORY_SIZE" );

#ifdef NANOX_MEMKIND_SUPPORT
      cfg.registerConfigOption( "smp-memkind", NEW Config::FlagOption( _memkindSupport, true ),
            "SMP memkind support." );
      cfg.registerArgOption( "smp-memkind", "smp-memkind" );
      cfg.registerEnvOption( "smp-memkind", "NX_SMP_MEMKIND" );

      cfg.registerConfigOption( "smp-memkind-memory-size", NEW Config::SizeVar( _memkindMemorySize ),
            "Set the size of SMP memkind memory area." );
      cfg.registerArgOption( "smp-memkind-memory-size", "smp-memkind-memory-size" );
      cfg.registerEnvOption( "smp-memkind-memory-size", "NX_SMP_MEMKIND_MEMORY_SIZE" );
#endif
   }

   virtual void init() {
      sys.setHostFactory( smpProcessorFactory );
      sys.setSMPPlugin( this );

      OS::getProcessAffinity( &_cpuSet );
      int available_cores_by_mask = CPU_COUNT( &_cpuSet );
      _availableCores = OS::getMaxProcessors();
      _requestedCoresByMask = available_cores_by_mask;

      if ( _availableCores == 0 ) {
         if ( available_cores_by_mask > 0 ) {
            warning0("SMPPlugin: Unable to detect the number of processors in the system, using the value provided by the process cpu mask (" << available_cores_by_mask << ").");
            _availableCores = available_cores_by_mask;
         } else if ( _requestedCores > 0 ) {
            warning0("SMPPlugin: Unable to detect the number of processors in the system and cpu mask not set, using the number of requested cpus (" << _requestedCores << ").");
            _availableCores = _requestedCores;
         } else {
            fatal0("SMPPlugin: Unable to detect the number of cpus of the system and --smp-cpus was unset or with a value less than 1.");
         }
      }
      //at this point _availableCores has a valid value

      if ( _requestedCores > 0 ) { //--smp-cpus flag was set
         if ( _requestedCores > available_cores_by_mask ) {
            warning0("SMPPlugin: Requested number of cpus is greater than the cpu mask provided, using the value specified by the mask (" << available_cores_by_mask << ").");
            _currentCores = available_cores_by_mask;
         } else {
            _currentCores = _requestedCores;
         }
      } else if ( _requestedCores == 0 ) { //no cpus requested through --smp-cpus
         _currentCores = available_cores_by_mask;
      } else {
         fatal0("Invalid number of requested cpus (--smp-cpus)");
      }
      verbose0("requested cpus: " << _requestedCores << " available: " << _availableCores << " to be used: " << _currentCores);


      _bindings.reserve( _availableCores );
      for ( unsigned int i=0; i<CPU_SETSIZE; i++ ) {
         if ( CPU_ISSET(i, &_cpuSet) ) {
            _bindings.push_back(i);
         }
      }

      //add the cpus that were not in the mask
      for ( int i = 0; i < _availableCores; i++ ) {
         if ( !CPU_ISSET(i, &_cpuSet) ) {
            _bindings.push_back(i);
         }
      }
      //std::cerr << "[ ";
      //for ( std::vector<int>::iterator it = _bindings.begin(); it != _bindings.end(); it++ ) {
      //   std::cerr << *it << " ";
      //}
      //std::cerr << "]" << std::endl;

      // Set _bindings structure once we have the system mask and the binding info
      // _bindings.reserve( _availableCores );
      // for ( int i=0, collisions = 0; i < _availableCores; ) {

      //    // The cast over cpu_affinity is needed because std::vector::size() returns a size_t type
      //    //int pos = (_bindingStart + _bindingStride*i + collisions) % (int)cpu_affinity.size();
      //    int pos = ( i + collisions) % (int)cpu_affinity.size();

      //    // 'pos' may be negative if either bindingStart or bindingStride were negative
      //    // this loop fixes that % operator is the remainder, not the modulo operation
      //    while ( pos < 0 ) pos+=cpu_affinity.size();

      //    if ( std::find( _bindings.begin(), _bindings.end(), cpu_affinity[pos] ) != _bindings.end() ) {
      //       collisions++;
      //       ensure( collisions != _availableCores, "Reached limit of collisions. We should never get here." );
      //       continue;
      //    }
      //    _bindings.push_back( cpu_affinity[pos] );
      //    i++;
      // }

      // std::cerr << "[ ";
      // for ( Bindings::iterator it = _bindings.begin(); it != _bindings.end(); it++ ) {
      //    std::cerr << *it << " ";
      // }
      // std::cerr << "]" << std::endl;

      CPU_ZERO( &_cpuActiveSet );

      _cpus = NEW std::vector<SMPProcessor *>( _availableCores, (SMPProcessor *) NULL ); 
      _cpusByCpuId = NEW std::vector<SMPProcessor *>( _availableCores, (SMPProcessor *) NULL ); 

      // Load & check NUMA config
      loadNUMAInfo();

      int count = 0;

      memory_space_id_t mem_id = sys.getRootMemorySpaceId();
#ifdef NANOX_MEMKIND_SUPPORT
      if ( _memkindSupport ) {
         mem_id = sys.addSeparateMemoryAddressSpace( ext::SMP, _smpAllocWide, sys.getRegionCacheSlabSize() );
         SeparateMemoryAddressSpace &memkindMem = sys.getSeparateMemory( mem_id );
         void *addr = memkind_malloc(MEMKIND_HBW, _memkindMemorySize);
         if ( addr == NULL ) {
            OSAllocator a;
            warning0("Could not allocate memory with memkind_malloc(), requested " << _memkindMemorySize << " bytes. Continuing with a regular allocator.");
            addr = a.allocate(_memkindMemorySize);
            if ( addr == NULL ) {
               fatal0("Could not allocate memory with a regullar allocator.");
            }
         }
         memkindMem.setSpecificData( NEW SimpleAllocator( ( uintptr_t ) addr, _memkindMemorySize ) );
      }
#endif

      for ( std::vector<int>::iterator it = _bindings.begin(); it != _bindings.end(); it++ ) {
         SMPProcessor *cpu;
         bool active = ( (count < _currentCores) && CPU_ISSET( *it, &_cpuSet) );
         unsigned numaNode = active ? getNodeOfPE( *it ) : std::numeric_limits<unsigned>::max();
         unsigned socket = numaNode;   /* FIXME: socket */
         
         if ( _smpPrivateMemory && count >= _smpHostCpus && !_memkindSupport ) {
            OSAllocator a;
            memory_space_id_t id = sys.addSeparateMemoryAddressSpace( ext::SMP, _smpAllocWide, sys.getRegionCacheSlabSize() );
            SeparateMemoryAddressSpace &numaMem = sys.getSeparateMemory( id );
            numaMem.setSpecificData( NEW SimpleAllocator( ( uintptr_t ) a.allocate(_smpPrivateMemorySize), _smpPrivateMemorySize ) );
            cpu = NEW SMPProcessor( *it, id, active, numaNode, socket );
         } else {

            cpu = NEW SMPProcessor( *it, mem_id, active, numaNode, socket );
         }
         CPU_SET( cpu->getBindingId() , &_cpuActiveSet );
         //cpu->setNUMANode( getNodeOfPE( cpu->getId() ) );
         (*_cpus)[count] = cpu;
         (*_cpusByCpuId)[ *it ] = cpu;
         count += 1;
      }

#ifdef NANOS_DEBUG_ENABLED
      if ( sys.getVerbose() ) {
         std::cerr << "Bindings: [ ";
         for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            std::cerr << (*it)->getBindingId() << ( (*it)->isActive() ? "a" : "i" ) << (*it)->getNumaNode() << " " ;
         }
         std::cerr << "]" << std::endl;
      }
#endif /* NANOS_DEBUG_ENABLED */

      /* reserve it for main thread */
      getFirstSMPProcessor()->reserve();
      getFirstSMPProcessor()->setNumFutureThreads( 1 );
   }

#if 0
   virtual unsigned int getPEsInNode( unsigned node ) const
   {
      // TODO (gmiranda): if HWLOC is available, use it.
      return getCoresPerSocket();
   }
#endif

   virtual unsigned int getEstimatedNumThreads() const {
      unsigned int count = 0;
      /* This function is called from getNumThreads() when no threads have been created,
       * which happens when the instrumentation plugin is initialized. At that point
       * the arch plugins have been loaded and have discovered the hardware resources
       * available, but no threads have been created yet.
       * Threads can not be created because they require the instrumentation plugin to
       * be correctly initialized (the creation process emits instrumentation events).
       *
       * The number of threads computed now is:
       * "number of smp workers" + "number of support threads"
       *
       * The number of smp workers is computed with this:
       * if a certain number of workers was requested:
       *    min of "the number of active non-reserved CPUs" and requested workers
       * else
       *    "the number of active non-reserved CPUs"
       *
       * The number of support threads is computed with this:
       * sum of "future threads" of each reserved cpu
       */
      int active_cpus = 0;
      int reserved_cpus = 0;
      int future_threads = 0; /* this is the amount of threads that the devices say they will create on reserved cpus */
      for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         active_cpus += (*it)->isActive();
         reserved_cpus += (*it)->isReserved();
         future_threads += (*it)->isReserved() ? (*it)->getNumFutureThreads() : 0;
      }

      if ( _requestedWorkers > 0 ) {
         count = std::min( ( _requestedWorkers - 1 /* main thd is already reserved */), active_cpus - reserved_cpus );
      } else {
         count = active_cpus - reserved_cpus;
      }
      debug0( __FUNCTION__ << " called before creating the SMP workers, the estimated number of workers is: " << count);
      return count + future_threads;

   }

   virtual unsigned int getNumThreads() const
   {
      return ( _idSeed.value() ? _idSeed.value() : getEstimatedNumThreads() );
   }

   virtual ProcessingElement* createPE( unsigned id, unsigned uid )
   {
      return NULL;
   }

   virtual void initialize() {
   }

   virtual void finalize() {
      if ( _memkindSupport ) {
         std::cerr << "memkind: SMP soft invalidations: " << sys.getSeparateMemory(1).getSoftInvalidationCount() << std::endl;
         std::cerr << "memkind: SMP hard invalidations: " << sys.getSeparateMemory(1).getHardInvalidationCount() << std::endl;
      } else if ( _smpPrivateMemory ) {
         for ( std::vector<SMPProcessor *>::const_iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            if ( (*it)->isActive() ) {
               std::cerr << "PrivateMem: cpu " << (*it)->getId()  << " SMP soft invalidations: " << sys.getSeparateMemory((*it)->getMemorySpaceId()).getSoftInvalidationCount() << std::endl;
               std::cerr << "PrivateMem: cpu " << (*it)->getId()  << " SMP hard invalidations: " << sys.getSeparateMemory((*it)->getMemorySpaceId()).getHardInvalidationCount() << std::endl;
            }
         }
      }
   }

   virtual void addPEs( std::map<unsigned int, ProcessingElement *> &pes ) const {
      for ( std::vector<SMPProcessor *>::const_iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         if ( (*it)->isActive() ) {
            pes.insert( std::make_pair( (*it)->getId(), *it ) );
         }
      }
   }

   virtual void addDevices( DeviceList &devices ) const
   {
      if ( !_cpus->empty() )
         devices.insert( ( *_cpus->begin() )->getDeviceType() );
   }

   virtual void startSupportThreads() {
   }

   virtual void startWorkerThreads( std::map<unsigned int, BaseThread *> &workers ) {
      //associateThisThread( sys.getUntieMaster() );
      ensure( _workers.size() == 1, "Main thread should be the only worker created so far." );
      workers.insert( std::make_pair( _workers[0]->getId(), _workers[0] ) );
      //create as much workers as possible
      int available_cpus = 0; /* my cpu is unavailable, numthreads is 1 */
      for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         available_cpus += ( (*it)->getNumThreads() == 0 && (*it)->isActive() );
      }

      int max_workers;  
      if ( _requestedWorkers != -1 && (_requestedWorkers - 1) > available_cpus ) {
         warning("SMPPlugin: requested number of workers (" << _requestedWorkers << ") is greater than the number of available cpus (" << available_cpus+1 << ") a total of " << available_cpus+1 << " workers will be created.");
         if ( _availableCores > _currentCores ) {
            warning("SMPPlugin: The system has more cpus available (" << _availableCores << ") but only " << _currentCores << " are being used. Try increasing the cpus available to Nanos++ using the --smp-cpus flag or the setting appropiate cpu mask (using the 'taskset' command). Please note that if both the --smp-cpus flag and the cpu mask are set, the most restrictive value will be considered.");
         } else {
            warning("SMPPlugin: All cpus are being used by Nanos++ (" << _availableCores << ") so you may have requested too many smp workers.");
         }
         max_workers = available_cpus + 1;
      } else {
         max_workers = ( _requestedWorkers == -1 ) ? available_cpus + 1 : _requestedWorkers;
      }

      int current_workers = 1;

      int idx = _bindingStart + _bindingStride;
      while ( current_workers < max_workers ) {
         idx = idx % _cpus->size();
         SMPProcessor *cpu = (*_cpus)[idx];
         if ( cpu->getNumThreads() == 0 && cpu->isActive() ) {
            BaseThread *thd = &cpu->startWorker();
            _workers.push_back( (SMPThread *) thd );
            workers.insert( std::make_pair( thd->getId(), thd ) );
            current_workers += 1;
            idx += _bindingStride;
         } else {
            idx += 1;
         }
      }
      _numWorkers = _workers.size();
      _workersCreated = true;

      //FIXME: this makes sense in OpenMP, also, in OpenMP this value is already set (see omp_init.cpp)
      //       In OmpSs, this will make omp_get_max_threads to return the number of SMP worker threads. 
      sys.getPMInterface().setNumThreads_globalState( _workers.size() );
   }

   virtual void setRequestedWorkers( int workers ) {
      _requestedWorkers = workers;
   }

   virtual ext::SMPProcessor *getFirstSMPProcessor() const {
      //ensure( _cpus != NULL, "Uninitialized SMP plugin.");
      ext::SMPProcessor *target = NULL;
      int bindingStart=_bindingStart%_cpus->size();
      //Get the first (aka bindingStart) active processor
      for ( std::vector<ext::SMPProcessor *>::const_iterator it = _cpus->begin();
            it != _cpus->end() && bindingStart>=0 ;
            it++ ) {
         if ( (*it)->isActive() ) {
            target = *it;
            --bindingStart;
         }
      }
      return target;
   }

   virtual cpu_set_t &getActiveSet() {
      return _cpuActiveSet;
   }

   virtual ext::SMPProcessor *getFirstFreeSMPProcessor() const {
      ensure( _cpus != NULL, "Uninitialized SMP plugin.");
      ext::SMPProcessor *target = NULL;
      for ( std::vector<ext::SMPProcessor *>::const_iterator it = _cpus->begin();
            it != _cpus->end() && !target;
            it++ ) {
         if ( (*it)->getNumThreads() == 0 && (*it)->isActive() ) {
            target = *it;
         }
      }
      return target;
   }

   virtual ext::SMPProcessor *getLastFreeSMPProcessorAndReserve() {
      ensure( _cpus != NULL, "Uninitialized SMP plugin.");
      ext::SMPProcessor *target = NULL;
      for ( std::vector<ext::SMPProcessor *>::const_reverse_iterator it = _cpus->rbegin();
            it != _cpus->rend() && !target;
            it++ ) {
         if ( (*it)->getNumThreads() == 0 && !(*it)->isReserved() && (*it)->isActive() ) {
            target = *it;
            target->reserve();
         }
      }
      return target;
   }

   virtual ext::SMPProcessor *getLastSMPProcessor() {
      ensure( _cpus != NULL, "Uninitialized SMP plugin.");
      ext::SMPProcessor *target = NULL;
      for ( std::vector<ext::SMPProcessor *>::const_reverse_iterator it = _cpus->rbegin();
            it != _cpus->rend() && !target;
            it++ ) {
         if ( (*it)->isActive() ) {
            target = *it;
         }
      }
      return target;
   }

   virtual ext::SMPProcessor *getFreeSMPProcessorByNUMAnodeAndReserve(int node) {
      ensure( _cpus != NULL, "Uninitialized SMP plugin.");
      ext::SMPProcessor *target = NULL;
      for ( std::vector<ext::SMPProcessor *>::const_reverse_iterator it = _cpus->rbegin();
            it != _cpus->rend() && !target;
            it++ ) {
         if ( (int) (*it)->getNumaNode() == node && (*it)->getNumThreads() == 0 && !(*it)->isReserved() && (*it)->isActive() ) {
            target = *it;
            target->reserve();
         }
      }
      return target;
   }

   virtual ext::SMPProcessor *getSMPProcessorByNUMAnode(int node, unsigned int idx) const {
      ensure( _cpus != NULL, "Uninitialized SMP plugin.");
      ext::SMPProcessor *target = NULL;
      std::vector<ext::SMPProcessor *>::const_reverse_iterator it = _cpus->rbegin();
      unsigned int counter = idx;

      while ( !target ) {
         if ( it == _cpus->rend() ) {
            it = _cpus->rbegin();
            if ( idx == counter ) {
               break;
            }
         }
         if ( (int) (*it)->getNumaNode() == node ) {
            if ( counter <= 0 ) {
               target = *it;
            }
            counter--;
         }
         ++it;
      }
      return target;
   }


   void loadNUMAInfo ()
   {
      if ( _numSockets == 0 ) {
         if ( sys._hwloc.isHwlocAvailable() ) {
            unsigned int allowedNodes = 0;
            unsigned int hwThreads = 0;
            sys._hwloc.getNumSockets(allowedNodes, _numSockets, hwThreads);
            if ( hwThreads == 0 ) hwThreads = _cpus->size(); //failed to read hwloc info
            if( _coresPerSocket == 0 )
               _coresPerSocket = std::ceil( hwThreads / static_cast<float>( allowedNodes ) );
         } else {
            _numSockets = 1;
         }
      }
      ensure0(_numSockets > 0, "Invalid number of sockets!");
      if ( _coresPerSocket == 0 )
         _coresPerSocket = std::ceil( _cpus->size() / static_cast<float>( _numSockets ) );
      ensure0(_coresPerSocket > 0, "Invalid number of cores per socket!");
//#ifdef HWLOC
//      hwloc_topology_t topology = ( hwloc_topology_t ) _hwlocTopology;
//
//      // Nodes that can be seen by hwloc
//      unsigned allowedNodes = 0;
//      // Hardware threads
//      unsigned hwThreads = 0;
//
//      // Read the number of numa nodes if the user didn't set that value
//      if ( _numSockets == 0 )
//      {
//         int depth = hwloc_get_type_depth( topology, HWLOC_OBJ_NODE );
//
//
//         // If there are NUMA nodes in this machine
//         if ( depth != HWLOC_TYPE_DEPTH_UNKNOWN ) {
//            //hwloc_const_cpuset_t cpuset = hwloc_topology_get_online_cpuset( topology );
//            //allowedNodes = hwloc_get_nbobjs_inside_cpuset_by_type( topology, cpuset, HWLOC_OBJ_NODE );
//            //hwThreads = hwloc_get_nbobjs_inside_cpuset_by_type( topology, cpuset, HWLOC_OBJ_PU );
//            unsigned nodes = hwloc_get_nbobjs_by_depth( topology, depth );
//            //hwloc_cpuset_t set = i
//
//            // For each node, count how many hardware threads there are below.
//            for ( unsigned nodeIdx = 0; nodeIdx < nodes; ++nodeIdx )
//            {
//               hwloc_obj_t node = hwloc_get_obj_by_depth( topology, depth, nodeIdx );
//               int localThreads = hwloc_get_nbobjs_inside_cpuset_by_type( topology, node->cpuset, HWLOC_OBJ_PU );
//               // Increase hw thread count
//               hwThreads += localThreads;
//               // If this node has hw threads beneath, increase the number of viewable nodes
//               if ( localThreads > 0 ) ++allowedNodes;
//            }
//            _numSockets = nodes;
//         }
//         // Otherwise, set it to 1
//         else {
//            allowedNodes = 1; 
//            _numSockets = 1;
//         }
//      }
//
//      if( _coresPerSocket == 0 )
//         _coresPerSocket = std::ceil( hwThreads / static_cast<float>( allowedNodes ) );
//#else
//      // Number of sockets can be read with
//      // cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l
//      // Cores per socket:
//      // cat /proc/cpuinfo | grep 'core id' | sort | uniq | wc -l
//
//      // Assume just 1 socket
//      if ( _numSockets == 0 )
//         _numSockets = 1;
//
//      // Same thing, just change the value if the user didn't provide one
//      if ( _coresPerSocket == 0 )
//         _coresPerSocket = std::ceil( _cpus->size() / static_cast<float>( _numSockets ) );
//#endif
      verbose0( toString( "[NUMA] " ) + toString( _numSockets ) + toString( " NUMA nodes, " ) + toString( _coresPerSocket ) + toString( " HW threads each." ) );
   }

   unsigned getNodeOfPE ( unsigned pe )
   {
      if ( sys._hwloc.isHwlocAvailable() ) {
         return sys._hwloc.getNumaNodeOfCpu( pe );
      } else {
         // Dirty way, will not work with hyperthreading
         // Use /sys/bus/cpu/devices/cpuX/
         //return pe / getCoresPerSocket();

         // Otherwise, return
         // FIXME: return NUMA nodes -1
         return getNumSockets() - 1;
      }
   }

   void setBindingStart ( int value ) { _bindingStart = value; }

   int getBindingStart () const { return _bindingStart; }

   void setBindingStride ( int value ) { _bindingStride = value;  }

   int getBindingStride () const { return _bindingStride; }

   void setBinding ( bool set ) { _bindThreads = set; }

   virtual bool getBinding () const { return _bindThreads; }

   virtual int getNumSockets() const { return _numSockets; }

   virtual void setNumSockets ( int numSockets ) { _numSockets = numSockets; }

   virtual int getCurrentSocket() const { return _currentSocket; }
   virtual void setCurrentSocket( int currentSocket ) { _currentSocket = currentSocket; }

   virtual int getCoresPerSocket() const { return _coresPerSocket; }

   virtual void setCoresPerSocket ( int coresPerSocket ) { _coresPerSocket = coresPerSocket; }


   // Not thread-safe
   void applyCpuMask()
   {
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
         NANOS_INSTRUMENT ( nanos_event_value_t num_threads_val = (nanos_event_value_t ) CPU_COUNT(&_cpuActiveSet ) )
         NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &num_threads_val); )

         BaseThread *thread;
      ThreadTeam *team = myThread->getTeam();
      unsigned int active_cpus = 0;

      for ( unsigned cpu_id = 0; cpu_id < _cpus->size() || active_cpus < (size_t)CPU_COUNT( &_cpuActiveSet ); cpu_id += 1 ) {
         int binding_id = (*_cpus)[ cpu_id ]->getBindingId();
         if ( CPU_ISSET( binding_id, &_cpuActiveSet ) ) {
            active_cpus += 1;
            // This PE should be running FIXME: this code should be inside ProcessingElement (wakeupWorkers ?)
            while ( (thread = (*_cpus)[cpu_id]->getUnassignedThread()) != NULL ) {
               sys.acquireWorker( team, thread, /* enterOthers */ true, /* starringOthers */ false, /* creator */ false );
               team->increaseFinalSize();
            }
         } else {
            // This PE should not FIXME: this code should be inside ProcessingElement (sleepWorkers ?)
            while ( (thread = (*_cpus)[cpu_id]->getActiveThread()) != NULL ) {
               thread->lock();
               thread->sleep();
               thread->unlock();
               team->decreaseFinalSize();
            }
         }
      }
   }

   void getCpuMask ( cpu_set_t *mask ) const
   {
      ::memcpy( mask, &_cpuActiveSet , sizeof(cpu_set_t) );
   }

   void setCpuMask ( const cpu_set_t *mask )
   {
      ::memcpy( &_cpuActiveSet , mask, sizeof(cpu_set_t) );
      processCpuMask();
   }

   void addCpuMask ( const cpu_set_t *mask )
   {
      CPU_OR( &_cpuActiveSet , &_cpuActiveSet , mask );
      processCpuMask();
   }


   void processCpuMask( void )
   {
      // if _bindThreads is enabled, update _bindings adding new elements of _cpuActiveSet
      if ( getBinding() ) {
         std::ostringstream oss_cpu_idx;
         oss_cpu_idx << "[";
         for ( int cpu=0; cpu<CPU_SETSIZE; cpu++) {
            if ( cpu > OS::getMaxProcessors()-1 && !sys.isSimulator() ) {
               CPU_CLR( cpu, &_cpuActiveSet );
               debug("Trying to use more cpus than available is not allowed (do you forget --simulator option?)");
               continue;
            }
            if ( CPU_ISSET( cpu, &_cpuActiveSet  ) ) {

               //      if ( std::find( _bindings.begin(), _bindings.end(), cpu ) == _bindings.end() ) {
               //         _bindings.push_back( cpu );
               //      }

               oss_cpu_idx << cpu << ", ";
            }
         }
         oss_cpu_idx << "]";
         verbose0( "PID[" << getpid() << "]. CPU affinity " << oss_cpu_idx.str() );
         if ( sys.getPMInterface().isMalleable() ) {
            applyCpuMask();
         }
      } else {
         verbose0( "PID[" << getpid() << "]. Changing number of threads: " << (int) myThread->getTeam()->getFinalSize() << " to " << (int) CPU_COUNT( &_cpuActiveSet ) );
         if ( sys.getPMInterface().isMalleable() ) {
            updateActiveWorkers( CPU_COUNT( &_cpuActiveSet ) );
         }
      }
   }

   SMPThread * getInactiveWorker( void )
   {
      SMPThread *thread;

      for ( unsigned i = 0; i < _workers.size(); i++ ) {
         thread = _workers[i];
         if ( thread->tryWakeUp() ) {
            return thread;
         }
      }
      return NULL;
   }

   virtual void updateActiveWorkers ( int nthreads )
   {
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
         NANOS_INSTRUMENT ( nanos_event_value_t num_threads_val = (nanos_event_value_t) nthreads; )
         NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &num_threads_val); )

         BaseThread *thread;
      //! \bug Team variable must be received as a function parameter
      ThreadTeam *team = myThread->getTeam();

      int num_threads = nthreads - team->getFinalSize();

      while ( !(team->isStable()) ) memoryFence();

      if ( num_threads < 0 ) team->setStable(false);

      team->setFinalSize(nthreads);

      //! \note If requested threads are more than current increase number of threads
      while ( num_threads > 0 ) {
         thread = getUnassignedWorker();
         if (!thread) thread = getInactiveWorker();
         if (thread) {
            sys.acquireWorker( team, thread, /* enterOthers */ true, /* starringOthers */ false, /* creator */ false );
            num_threads--;
         }
      }

      //! \note If requested threads are less than current decrease number of threads
      while ( num_threads < 0 ) {
         thread = getAssignedWorker( team );
         if ( thread ) {
            thread->sleep();
            thread->unlock();
            num_threads++;
         }
      }


   }

   SMPThread * getUnassignedWorker ( void )
   {
      SMPThread *thread;

      for ( unsigned i = 0; i < _workers.size(); i++ ) {
         thread = _workers[i];
         if ( !thread->hasTeam() && !thread->isSleeping() ) {

            // skip if the thread is not in the mask
            if ( getBinding() && !CPU_ISSET( thread->getCpuId(), &_cpuActiveSet ) ) {
               continue;
            }

            // recheck availability with exclusive access
            thread->lock();

            if ( thread->hasTeam() || thread->isSleeping()) {
               // we lost it
               thread->unlock();
               continue;
            }

            thread->reserve(); // set team flag only
            thread->unlock();

            return thread;
         }
      }
      return NULL;
   }


   SMPThread * getAssignedWorker ( ThreadTeam *team )
   {
      SMPThread *thread;

      std::vector<SMPThread *>::reverse_iterator rit;
      for ( rit = _workers.rbegin(); rit != _workers.rend(); ++rit ) {
         thread = *rit;
         thread->lock();
         //! \note Checking thread availabitity.
         if ( (thread->getTeam() == team) && !thread->isSleeping() && !thread->isTeamCreator() ) {
            //! \note return this thread LOCKED!!!
            return thread;
         }
         thread->unlock();
      }

      //! \note If no thread has found, return NULL.
      return NULL;
   }

   virtual void admitCurrentThread( std::map<unsigned int, BaseThread *> &workers ) {

      ext::SMPProcessor *cpu = getFirstFreeSMPProcessor();

      //! \note Create a new Thread object and associate it to the current thread
      BaseThread *thread = &cpu->associateThisThread ( /* untie */ true ) ;
      workers.insert( std::make_pair( thread->getId(), thread ) );
      _workers.push_back( (SMPThread *) thread );

      //! \note Update current cpu active set mask
      CPU_SET( cpu->getBindingId(), &_cpuActiveSet );

      //! \note Getting Programming Model interface data
      WD &mainWD = *myThread->getCurrentWD();
      if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
         char *data = NEW char[sys.getPMInterface().getInternalDataSize()];
         sys.getPMInterface().initInternalData( data );
         mainWD.setInternalData( data );
      }

      //! \note Include thread into main thread
      sys.acquireWorker( sys.getMainTeam(), thread, /* enter */ true, /* starring */ false, /* creator */ false );
   }

   virtual int getCpuCount() const {
      return _cpus->size();
   }


   virtual unsigned int getNumPEs() const {
      return _currentCores;
   }
   virtual unsigned int getMaxPEs() const {
      return _availableCores;
   }

   unsigned int getEstimatedNumWorkers() const {
      unsigned int count = 0;
      /*if a certain number of workers was requested, pick the minimum between that value
       * and the number of cpus and the support threads requested
       */
      int active_cpus = 0;
      int reserved_cpus = 0;
      for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         active_cpus += (*it)->isActive();
         reserved_cpus += (*it)->isReserved();
      }


      if ( _requestedWorkers > 0 ) {
         count = std::min( _requestedWorkers, active_cpus - reserved_cpus );
      } else {
         count = active_cpus - reserved_cpus;
      }
      debug0( __FUNCTION__ << " called before creating the SMP workers, the estimated number of workers is: " << count);
      return count;
   }

   virtual unsigned int getNumWorkers() const {
      return _workersCreated ? _numWorkers : getEstimatedNumWorkers();
   }

   virtual unsigned int getMaxWorkers() const {
      return _currentCores;
   }

   virtual SMPThread &associateThisThread( bool untie ) {
      SMPThread &thd = getFirstSMPProcessor()->associateThisThread( untie );
      _workers.push_back( &thd );
      return thd;
   }
   
   virtual int getRequestedWorkersOMPSS() const {
      return _requestedWorkersOMPSS;
   }

   virtual void getBindingMaskString( std::ostream &o ) const {
      o << "[ ";
      for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         o << (*it)->getBindingId() << ( (*it)->isActive() ? "a " : "i ");
      }
      o << "]";
   }

};
}
}

DECLARE_PLUGIN("arch-smp",nanos::ext::SMPPlugin);
