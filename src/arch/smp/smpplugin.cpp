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

#include "smpbaseplugin_decl.hpp"
#include "plugin.hpp"
#include "smpprocessor.hpp"
#include "smpdd.hpp"
#include "os.hpp"
#include "osallocator_decl.hpp"
#include "system.hpp"
#include "basethread.hpp"
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
   int                          _requestedCPUs;
   int                          _availableCPUs;
   int                          _currentCPUs;
   int                          _requestedWorkers;
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

   // Nanos++ scheduling domain
   cpu_set_t                    _cpuSystemMask;   /*!< \brief system's default cpu_set */
   cpu_set_t                    _cpuProcessMask;  /*!< \brief process' default cpu_set */
   cpu_set_t                    _cpuActiveMask;   /*!< \brief mask of current active cpus */

   //! Physical NUMA nodes
   int                          _numSockets;
   int                          _CPUsPerSocket;
   //! The socket that will be assigned to the next WD
   int                          _currentSocket;


   //! CPU id binding list
   Bindings                     _bindings;

   bool                         _memkindSupport;
   std::size_t                  _memkindMemorySize;

   public:
   SMPPlugin() : SMPBasePlugin( "SMP PE Plugin", 1 )
                 , _idSeed( 0 )
                 , _requestedCPUs( 0 )
                 , _availableCPUs( 0 )
                 , _currentCPUs( 0 )
                 , _requestedWorkers( -1 )
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
                 , _cpuSystemMask()
                 , _cpuProcessMask()
                 , _cpuActiveMask()
                 , _numSockets( 0 )
                 , _CPUsPerSocket( 0 )
                 , _bindings()
                 , _memkindSupport( false )
                 , _memkindMemorySize( 1024*1024*1024 ) // 1Gb
   {}

   virtual unsigned int getNewSMPThreadId()
   {
      return _idSeed++;
   }

   virtual void config ( Config& cfg )
   {
      cfg.setOptionsSection( "SMP Arch", "SMP specific options" );
      SMPProcessor::prepareConfig( cfg );
      SMPDD::prepareConfig( cfg );
      cfg.registerConfigOption ( "smp-num-pes", NEW Config::PositiveVar ( _requestedCPUs ), "CPUs requested." );
      cfg.registerArgOption ( "smp-num-pes", "smp-cpus" );
      cfg.registerEnvOption( "smp-num-pes", "NX_SMP_CPUS" );

      cfg.registerConfigOption ( "smp-workers", NEW Config::PositiveVar ( _requestedWorkers ), "Worker threads requested." );
      cfg.registerArgOption ( "smp-workers", "smp-workers" );
      cfg.registerEnvOption( "smp-workers", "NX_SMP_WORKERS" );

      cfg.registerConfigOption( "cpus-per-socket", NEW Config::PositiveVar( _CPUsPerSocket ),
            "Number of CPUs per socket." );
      cfg.registerArgOption( "cpus-per-socket", "cpus-per-socket" );

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

   virtual void init()
   {
      sys.setHostFactory( smpProcessorFactory );
      sys.setSMPPlugin( this );

      //! \note Set initial CPU architecture variables
      OS::getSystemAffinity( &_cpuSystemMask );
      OS::getProcessAffinity( &_cpuProcessMask );
      int available_cpus_by_mask = CPU_COUNT( &_cpuSystemMask );
      _availableCPUs = OS::getMaxProcessors();

      if ( _availableCPUs == 0 ) {
         if ( available_cpus_by_mask > 0 ) {
            warning0("SMPPlugin: Unable to detect the number of processors in the system.");
            warning0("SMPPlugin: Using the value provided by the process cpu mask (" << available_cpus_by_mask << ").");
            _availableCPUs = available_cpus_by_mask;
         } else if ( _requestedCPUs > 0 ) {
            warning0("SMPPlugin: Unable to detect the number of processors in the system and cpu mask not set");
            warning0("Using the number of requested cpus (" << _requestedCPUs << ").");
            _availableCPUs = _requestedCPUs;
         } else {
            fatal0("SMPPlugin: Unable to detect the number of cpus of the system and --smp-cpus was unset or with a value less than 1.");
         }
      }

      if ( _requestedCPUs > 0 ) { //--smp-cpus flag was set
         if ( _requestedCPUs > available_cpus_by_mask ) {
            warning0("SMPPlugin: Requested number of cpus is greater than the cpu mask provided.");
            warning0("Using the value specified by the mask (" << available_cpus_by_mask << ").");
            _currentCPUs = available_cpus_by_mask;
         } else {
            _currentCPUs = _requestedCPUs;
         }
      } else if ( _requestedCPUs == 0 ) { //no cpus requested through --smp-cpus
         _currentCPUs = available_cpus_by_mask;
      } else {
         fatal0("Invalid number of requested cpus (--smp-cpus)");
      }
      verbose0("requested cpus: " << _requestedCPUs << " available: " << _availableCPUs << " to be used: " << _currentCPUs);

      //! \note Fill _bindings vector with the active CPUs first, then the not active
      _bindings.reserve( _availableCPUs );
      for ( int i = 0; i < _availableCPUs; i++ ) {
         if ( CPU_ISSET(i, &_cpuProcessMask) ) {
            _bindings.push_back(i);
         }
      }
      for ( int i = 0; i < _availableCPUs; i++ ) {
         if ( !CPU_ISSET(i, &_cpuProcessMask) ) {
            _bindings.push_back(i);
         }
      }

      //! \note Load & check NUMA config (_cpus vectors must be created before)
      _cpus = NEW std::vector<SMPProcessor *>( _availableCPUs, (SMPProcessor *) NULL );
      _cpusByCpuId = NEW std::vector<SMPProcessor *>( _availableCPUs, (SMPProcessor *) NULL );

      loadNUMAInfo();

      //! \note Create the SMPProcessors in _cpus array
      CPU_ZERO( &_cpuActiveMask );
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
         message0("Memkind address range: " << addr << " - " << (void *) ((uintptr_t)addr + _memkindMemorySize ));
         memkindMem.setSpecificData( NEW SimpleAllocator( ( uintptr_t ) addr, _memkindMemorySize ) );
      }
#endif

      for ( std::vector<int>::iterator it = _bindings.begin(); it != _bindings.end(); it++ ) {
         SMPProcessor *cpu;
         bool active = ( (count < _currentCPUs) && CPU_ISSET( *it, &_cpuProcessMask) );
         unsigned numaNode;

         // If this PE can't be seen by hwloc (weird case in Altix 2, for instance)
         if ( !sys._hwloc.isCpuAvailable( *it ) ) {
            /* There's a problem: we can't query it's numa
            node. Let's give it 0 (ticket #1090), consider throwing a warning */
            numaNode = 0;
         }
         else
            numaNode = getNodeOfPE( *it );
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
         if ( active ) {
            CPU_SET( cpu->getBindingId() , &_cpuActiveMask );
         }
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
   }

   virtual unsigned int getEstimatedNumThreads() const
   {
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
       *    "the number of requested workers"
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
         count = _requestedWorkers;
      } else {
         count = active_cpus - reserved_cpus;
      }
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

   virtual void initialize() { }

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

   virtual void addPEs( std::map<unsigned int, ProcessingElement *> &pes ) const
   {
      for ( std::vector<SMPProcessor *>::const_iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            pes.insert( std::make_pair( (*it)->getId(), *it ) );
      }
   }

   virtual void addDevices( DeviceList &devices ) const
   {
      if ( !_cpus->empty() )
         devices.insert( ( *_cpus->begin() )->getDeviceType() );
   }

   virtual void startSupportThreads() { }

   virtual void startWorkerThreads( std::map<unsigned int, BaseThread *> &workers )
   {
      ensure( _workers.size() == 1, "Main thread should be the only worker created so far." );
      workers.insert( std::make_pair( _workers[0]->getId(), _workers[0] ) );
      //create as much workers as possible
      int available_cpus = 0; /* my cpu is unavailable, numthreads is 1 */
      int active_cpus = 0;
      for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         available_cpus += ( (*it)->getNumThreads() == 0 && (*it)->isActive() );
         active_cpus += (*it)->isActive();
      }

      int max_workers = 0;
      bool ignore_reserved_cpus = false;

      if ( available_cpus+1 >= _requestedWorkers ) {
         /* we have plenty of cpus to create requested threads or the user
          * did not request a specific amount of workers
          */
         max_workers = ( _requestedWorkers == -1 ) ? available_cpus + 1 : _requestedWorkers;
      } else {
         /* more workers than cpus available have been requested */
         max_workers = _requestedWorkers;
         ignore_reserved_cpus = true;
         warning0( "You have explicitly requested more SMP workers than available CPUs" );
      }

      //! These variables are used to support oversubscription of threads
      int limit_workers_per_cpu = 1;
      int num_cpus_with_current_limit = 0;

      int workers_per_cpu[_cpus->size()];
      for (unsigned int i = 0; i < _cpus->size(); ++i)
          workers_per_cpu[i] = 0;

      int bindingStart = _bindingStart % _cpus->size();
      //! \bug FIXME: This may be wrong in some cases...
      workers_per_cpu[bindingStart] = 1;
      num_cpus_with_current_limit++;

      if (active_cpus == 1) {
         //! This is a special case, we have to increase the limit of workers
         //! per cpu because we have already added a worker to the first (and
         //! unique) CPU
         limit_workers_per_cpu++;
         num_cpus_with_current_limit = 0;
      }

      int current_workers = 1;
      int idx = bindingStart + _bindingStride;
      while ( current_workers < max_workers ) {

         idx = (idx % _cpus->size());

         SMPProcessor *cpu = (*_cpus)[idx];
         if ( cpu->isActive()
               && (workers_per_cpu[idx] < limit_workers_per_cpu)
               && (!cpu->isReserved() || ignore_reserved_cpus) ) {

            BaseThread *thd = &cpu->startWorker();
            _workers.push_back( (SMPThread *) thd );
            workers.insert( std::make_pair( thd->getId(), thd ) );

            workers_per_cpu[idx]++;
            num_cpus_with_current_limit++;

            current_workers++;
            idx += _bindingStride;

            if (num_cpus_with_current_limit == active_cpus) {
               //! All the enabled cpus have the same number of workers. So, if
               //! we need to add more workers, we have to increase the limit of
               //! workers per cpu
               limit_workers_per_cpu++;

               //! No cpu has reached the current limit because we have
               //! increased it in the previous statement
               num_cpus_with_current_limit = 0;
            }
         } else {
            idx++;
         }
      }
      _workersCreated = true;

      //FIXME: this makes sense in OpenMP, also, in OpenMP this value is already set (see omp_init.cpp)
      //       In OmpSs, this will make omp_get_max_threads to return the number of SMP worker threads.
      sys.getPMInterface().setNumThreads_globalState( _workers.size() );

      // Remove from Active Mask those CPUs that do not contain any thread nor are reserved by any device
      for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         SMPProcessor *cpu = (*it);
         int cpuid = cpu->getBindingId();
         if ( CPU_ISSET( cpuid, &_cpuActiveMask )
               && cpu->getNumThreads() == 0
               && !cpu->isReserved() ) {
            CPU_CLR( cpuid, &_cpuActiveMask );
         }
      }
   }

   virtual void setRequestedWorkers( int workers )
   {
      _requestedWorkers = workers;
   }

   virtual int getRequestedWorkers( void ) const
   {
      return _requestedWorkers;
   }

   virtual ext::SMPProcessor *getFirstSMPProcessor() const
   {
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

   virtual ext::SMPProcessor *getFirstFreeSMPProcessor() const
   {
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


   virtual ext::SMPProcessor *getLastFreeSMPProcessorAndReserve()
   {
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

   virtual ext::SMPProcessor *getFreeSMPProcessorByNUMAnodeAndReserve(int node)
   {
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

   virtual ext::SMPProcessor *getSMPProcessorByNUMAnode(int node, unsigned int idx) const
   {
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
            if( _CPUsPerSocket == 0 )
               _CPUsPerSocket = std::ceil( hwThreads / static_cast<float>( allowedNodes ) );
         } else {
            _numSockets = 1;
         }
      }
      ensure0(_numSockets > 0, "Invalid number of sockets!");
      if ( _CPUsPerSocket == 0 )
         _CPUsPerSocket = std::ceil( _cpus->size() / static_cast<float>( _numSockets ) );
      ensure0(_CPUsPerSocket > 0, "Invalid number of CPUs per socket!");
      verbose0( toString( "[NUMA] " ) + toString( _numSockets ) + toString( " NUMA nodes, " ) +
            toString( _CPUsPerSocket ) + toString( " HW threads each." ) );
   }

   unsigned getNodeOfPE ( unsigned pe )
   {
      if ( sys._hwloc.isHwlocAvailable() ) {
         return sys._hwloc.getNumaNodeOfCpu( pe );
      } else {
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

   virtual int getCPUsPerSocket() const { return _CPUsPerSocket; }

   virtual void setCPUsPerSocket ( int cpus_per_socket ) { _CPUsPerSocket = cpus_per_socket; }

   virtual const cpu_set_t& getCpuProcessMask () const
   {
      return _cpuProcessMask;
   }

   virtual void getCpuProcessMask ( cpu_set_t *mask ) const
   {
      ::memcpy( mask, &_cpuProcessMask , sizeof(cpu_set_t) );
   }

   virtual bool setCpuProcessMask ( const cpu_set_t *mask, std::map<unsigned int, BaseThread *> &workers )
   {
      bool success = false;
      if ( isValidMask( mask ) ) {
         ::memcpy( &_cpuProcessMask, mask, sizeof(cpu_set_t) );
         ::memcpy( &_cpuActiveMask, mask, sizeof(cpu_set_t) );
         int master_cpu = workers[0]->getCpuId();
         if ( !sys.getUntieMaster() && !CPU_ISSET( master_cpu, mask ) ) {
            // If master thread is tied and mask does not include master's cpu, force it
            CPU_SET( master_cpu, &_cpuProcessMask );
            CPU_SET( master_cpu, &_cpuActiveMask );
         } else {
            // Return only true success when we have set an unmodified user mask
            success = true;
         }
         applyCpuMask( workers );
      }
      return success;
   }

   virtual void addCpuProcessMask ( const cpu_set_t *mask, std::map<unsigned int, BaseThread *> &workers )
   {
      CPU_OR( &_cpuProcessMask , &_cpuProcessMask , mask );
      ::memcpy( &_cpuActiveMask, &_cpuProcessMask, sizeof(cpu_set_t) );
      applyCpuMask( workers );
   }

   virtual const cpu_set_t& getCpuActiveMask () const
   {
      return _cpuActiveMask;
   }

   virtual void getCpuActiveMask ( cpu_set_t *mask ) const
   {
      ::memcpy( mask, &_cpuActiveMask, sizeof(cpu_set_t) );
   }

   virtual bool setCpuActiveMask ( const cpu_set_t *mask, std::map<unsigned int, BaseThread *> &workers )
   {
      bool success = false;
      if ( isValidMask( mask ) ) {
         ::memcpy( &_cpuActiveMask, mask, sizeof(cpu_set_t) );
         int master_cpu = workers[0]->getCpuId();
         if ( !sys.getUntieMaster() && !CPU_ISSET( master_cpu, mask ) ) {
            // If master thread is tied and mask does not include master's cpu, force it
            CPU_SET( master_cpu, &_cpuActiveMask );
         } else {
            // Return only true success when we have set an unmodified user mask
            success = true;
         }
         applyCpuMask( workers );
      }
      return success;
   }

   virtual void addCpuActiveMask ( const cpu_set_t *mask, std::map<unsigned int, BaseThread *> &workers )
   {
      CPU_OR( &_cpuActiveMask, &_cpuActiveMask, mask );
      applyCpuMask( workers );
   }

#if 0
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
#endif

   virtual void updateActiveWorkers ( int nthreads, std::map<unsigned int, BaseThread *> &workers, ThreadTeam *team )
   {
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
      NANOS_INSTRUMENT ( nanos_event_value_t num_threads_val = (nanos_event_value_t) nthreads; )
      NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &num_threads_val); )

      int num_threads = nthreads - team->getFinalSize();
      int new_workers = nthreads - _workers.size();

      //! \note Probably it can be relaxed, but at the moment running in a safe mode
      //! waiting for the team to be stable
      while ( !(team->isStable()) ) {
         memoryFence();
         // weird scenario: Only one thread left to leave the team, but it's me and I'm blocked here
         if ( myThread->isSleeping() && team->size() == team->getFinalSize()+1  ) {
            team->setStable(true);
         }
      }
      if ( num_threads < 0 ) team->setStable(false);

      //! \note Creating new workers (if needed)
      ensure( _cpus != NULL, "Uninitialized SMP plugin.");
      // FIXME: we are considering _cpus as the active ones. This will not be always right
      unsigned int max_thds_per_cpu = std::ceil( nthreads / static_cast<float>(_cpus->size()) );
      std::vector<ext::SMPProcessor *>::const_iterator cpu_it;
      for ( cpu_it = _cpus->begin(); cpu_it != _cpus->end() && new_workers > 0; ++cpu_it ) {
         if ( (*cpu_it)->getNumThreads() < max_thds_per_cpu ) {
            createWorker( (*cpu_it), workers );
            new_workers--;
         }
      }

      //! \note We can safely iterate over workers, since threads are created in a round-robin way per CPU
      int active_threads_checked = 0;
      std::vector<ext::SMPThread *>::const_iterator w_it;
      for ( w_it = _workers.begin(); w_it != _workers.end(); ++w_it ) {
         if ( active_threads_checked < nthreads ) {
            (*w_it)->tryWakeUp( team );
            active_threads_checked++;
         } else {
            (*w_it)->sleep();
         }
      }

#if 0
      BaseThread *thread;
      //! \note If requested threads are more than current increase number of threads
      while ( num_threads > 0 ) {
         thread = getUnassignedWorker();
         if (!thread) thread = getInactiveWorker();
         if (thread) {
            team->increaseFinalSize();
            sys.acquireWorker( team, thread, /* enter */ true, /* starring */ false, /* creator */ false );
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
#endif

   }

#if 0
   SMPThread * getUnassignedWorker ( void )
   {
      SMPThread *thread;

      for ( unsigned i = 0; i < _workers.size(); i++ ) {
         thread = _workers[i];
         if ( !thread->hasTeam() && !thread->isSleeping() ) {

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
#endif

   virtual void admitCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker )
   {

      ext::SMPProcessor *cpu = getFirstFreeSMPProcessor();

      if ( cpu == NULL ) cpu = getFirstSMPProcessor();

      //! \note Create a new Thread object and associate it to the current thread
      BaseThread *thread = &cpu->associateThisThread ( /* untie */ true ) ;

      if ( !isWorker ) return;

      workers.insert( std::make_pair( thread->getId(), thread ) );
      _workers.push_back( ( SMPThread * ) thread );

      //! \note Update current cpu active set mask
      CPU_SET( cpu->getBindingId(), &_cpuActiveMask );

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

   virtual void expelCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker )
   {
      if ( isWorker ) {
         workers.erase( myThread->getId() );
      }
   }

   virtual int getCpuCount() const
   {
      return _cpus->size();
   }

   virtual unsigned int getNumPEs() const
   {
      return _currentCPUs;
   }

   virtual unsigned int getMaxPEs() const
   {
      return _availableCPUs;
   }

   unsigned int getEstimatedNumWorkers() const
   {
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
         count = _requestedWorkers;
      } else {
         count = active_cpus - reserved_cpus;
      }
      debug0( __FUNCTION__ << " called before creating the SMP workers, the estimated number of workers is: " << count);
      return count;
   }

   virtual unsigned int getNumWorkers() const
   {
      return _workersCreated ? _workers.size() : getEstimatedNumWorkers();
   }

   //! \brief Get the max number of Workers that could run with the current Active Mask
   virtual unsigned int getMaxWorkers() const
   {
      int max_workers = 0;
      for ( std::vector<SMPProcessor *>::const_iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         if ( (*it)->isActive() ) {
            max_workers += std::max( (*it)->getNumThreads(), static_cast<std::size_t>(1U) );
         }
      }
      return max_workers;
   }

   virtual SMPThread &associateThisThread( bool untie )
   {
      SMPThread &thd = getFirstSMPProcessor()->associateThisThread( untie );
      _workers.push_back( &thd );
      return thd;
   }

   /*! \brief Force the creation of at least 1 thread per CPU.
    */
   virtual void forceMaxThreadCreation()
   {
      cpu_set_t mask;
      // Save original active mask
      getCpuActiveMask( &mask );
      // Set all CPUs active
      sys.setCpuActiveMask( &_cpuSystemMask );
      // Fall back
      sys.setCpuActiveMask( &mask );
   }

   /*! \brief Create a worker in a suitable CPU
    */
   void createWorker( std::map<unsigned int, BaseThread *> &workers )
   {
      unsigned int active_cpus = 0;
      std::vector<ext::SMPProcessor *>::const_iterator cpu_it;
      for ( cpu_it = _cpus->begin(); cpu_it != _cpus->end(); ++cpu_it ) {
         if ( (*cpu_it)->isActive() ) active_cpus++;
      }

      unsigned int max_thds_per_cpu = std::ceil( (_workers.size()+1) / static_cast<float>(active_cpus) );
      for ( cpu_it = _cpus->begin(); cpu_it != _cpus->end(); ++cpu_it ) {
         SMPProcessor *cpu = (*cpu_it);
         if ( cpu->isActive() && cpu->getNumThreads() < max_thds_per_cpu ) {
            createWorker( cpu, workers );
            break;
         }
      }
   }

   /*! \brief returns a human readable string containing information about the binding mask, detecting ranks.
    *       format e.g.,
    *           active[ i-j, m, o-p, ] - inactive[ k-l, n, ]
    */
   virtual std::string getBindingMaskString() const {
      if ( _cpusByCpuId->empty() ) return "";

      // inactive/active cpus list
      std::ostringstream a, i;

      // Initialize rank limits with the first cpu in the list
      int a0 = -1, aN = -1, i0 = -1, iN = -1;
      SMPProcessor *first_cpu = _cpusByCpuId->front();
      first_cpu->isActive() ? a0 = first_cpu->getBindingId() : i0 = first_cpu->getBindingId();

      // Iterate through begin+1..end
      for ( std::vector<SMPProcessor *>::iterator curr = _cpusByCpuId->begin()+1; curr != _cpusByCpuId->end(); curr++ ) {
         /* Detect whether there is a state change (a->i/i->a). If so,
          * close the rank and start a new one. If it's the last iteration
          * close it anyway.
          */
         std::vector<SMPProcessor *>::iterator prev = curr-1;
         if ( (*curr)->isActive() && !(*prev)->isActive() ) {
            // change, i->a
            iN = (*prev)->getBindingId();
            a0 = (*curr)->getBindingId();
            ( i0 != iN ) ? i << i0 << "-" << iN << ", " : i << i0 << ", ";
         } else if ( !(*curr)->isActive() && (*prev)->isActive() ) {
            // change, a->i
            aN = (*prev)->getBindingId();
            i0 = (*curr)->getBindingId();
            ( a0 != aN ) ? a << a0 << "-" << aN << ", " : a << a0 << ", ";
         }
      }

      // close ranks and append strings according to the last cpu
      SMPProcessor *last_cpu = _cpusByCpuId->back();
      if ( last_cpu->isActive() ) {
         aN = last_cpu->getBindingId();
         ( a0 != aN ) ? a << a0 << "-" << aN << ", " : a << a0 << ", ";
      } else {
         iN = last_cpu->getBindingId();
         ( i0 != iN ) ? i << i0 << "-" << iN << ", " : i << i0 << ", ";
      }

      // remove last comma
      std::string sa = a.str(), si = i.str();
      if (!sa.empty()) sa.erase(sa.length()-2);
      if (!si.empty()) si.erase(si.length()-2);

      return "active[ " + sa + " ] - inactive[ " + si + " ]";
    }

private:

   void applyCpuMask ( std::map<unsigned int, BaseThread *> &workers )
   {
      /* OmpSs */
      if ( sys.getPMInterface().isMalleable() ) {
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
         NANOS_INSTRUMENT ( nanos_event_value_t num_threads_val = (nanos_event_value_t ) CPU_COUNT(&_cpuActiveMask ) )
         NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &num_threads_val); )

         for ( std::vector<ext::SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            ext::SMPProcessor *target = *it;
            int binding_id = target->getBindingId();
            if ( CPU_ISSET( binding_id, &_cpuActiveMask ) ) {

               /* Create a new worker if the target PE is empty */
               if ( target->getNumThreads() == 0 ) {
                  createWorker( target, workers );
               }

               target->wakeUpThreads();
            } else {
               target->sleepThreads();
            }
         }
      }
      /* OpenMP */
      else {
         /* Modify the number of threads on the fly is not allowed in OpenMP. Just set PE flags */
         for ( std::vector<ext::SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            ext::SMPProcessor *target = *it;
            int binding_id = target->getBindingId();
            if ( CPU_ISSET( binding_id, &_cpuActiveMask ) ) {
               if ( !target->isActive() ) {
                  target->setActive();
               }
            }
         }
      }
   }

   void createWorker( ext::SMPProcessor *target, std::map<unsigned int, BaseThread *> &workers )
   {
      NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )
      if ( !target->isActive() ) {
         target->setActive();
      }
      BaseThread *thread = &(target->startWorker());
      _workers.push_back( (SMPThread *) thread );
      workers.insert( std::make_pair( thread->getId(), thread ) );

      /* Set up internal data */
      WD & threadWD = thread->getThreadWD();
      if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
         char *data = NEW char[sys.getPMInterface().getInternalDataSize()];
         sys.getPMInterface().initInternalData( data );
         threadWD.setInternalData( data );
      }
      sys.getPMInterface().setupWD( threadWD );
   }

   bool isValidMask( const cpu_set_t *mask )
   {
      // A mask is valid if it shares at least 1 bit with the system mask
      cpu_set_t m;
      CPU_AND( &m, mask, &_cpuSystemMask );
      return CPU_COUNT( &m ) > 0;
   }

};
}
}

DECLARE_PLUGIN("arch-smp",nanos::ext::SMPPlugin);
