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

#include "smpplugin_decl.hpp"

#include <iostream>

#include "atomic.hpp"
#include "debug.hpp"
#include "smpprocessor.hpp"
#include "os.hpp"
#include "osallocator_decl.hpp"

#include "cpuset.hpp"
#include <limits>

#ifdef HAVE_MEMKIND_H
#include <memkind.h>
#endif

namespace nanos {
namespace ext {

nanos::PE * smpProcessorFactory ( int id, int uid )
{
   return NULL;//NEW SMPProcessor( );
}

    SMPPlugin::SMPPlugin() : SMPBasePlugin( "SMP PE Plugin", 1 )
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
                 , _threadsPerCore( 0 )
                 , _cpuSystemMask()
                 , _cpuProcessMask()
                 , _cpuActiveMask()
                 , _numSockets( 0 )
                 , _CPUsPerSocket( 0 )
                 , _bindings()
                 , _memkindSupport( false )
                 , _memkindMemorySize( 1024*1024*1024 ) // 1Gb
                 , _asyncSMPTransfers( true )
   {}

   SMPPlugin::~SMPPlugin() {
      delete _cpus;
      delete _cpusByCpuId;
   }
   unsigned int SMPPlugin::getNewSMPThreadId()
   {
      return _idSeed++;
   }

   void SMPPlugin::config ( Config& cfg )
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

#ifdef MEMKIND_SUPPORT
      cfg.registerConfigOption( "smp-memkind", NEW Config::FlagOption( _memkindSupport, true ),
            "SMP memkind support." );
      cfg.registerArgOption( "smp-memkind", "smp-memkind" );
      cfg.registerEnvOption( "smp-memkind", "NX_SMP_MEMKIND" );

      cfg.registerConfigOption( "smp-memkind-memory-size", NEW Config::SizeVar( _memkindMemorySize ),
            "Set the size of SMP memkind memory area." );
      cfg.registerArgOption( "smp-memkind-memory-size", "smp-memkind-memory-size" );
      cfg.registerEnvOption( "smp-memkind-memory-size", "NX_SMP_MEMKIND_MEMORY_SIZE" );
#endif
      cfg.registerConfigOption( "smp-sync-transfers", NEW Config::FlagOption( _asyncSMPTransfers, false ),
            "SMP sync transfers." );
      cfg.registerArgOption( "smp-sync-transfers", "smp-sync-transfers" );
      cfg.registerEnvOption( "smp-sync-transfers", "NX_SMP_SYNC_TRANSFERS" );

      cfg.registerConfigOption( "smp-threads-per-core", NEW Config::PositiveVar( _threadsPerCore ),
            "Limit the number of threads per core on SMT processors." );
      cfg.registerArgOption( "smp-threads-per-core", "smp-threads-per-core" );
   }

   void SMPPlugin::init()
   {
      sys.setHostFactory( smpProcessorFactory );
      sys.setSMPPlugin( this );

      //! \note Set initial CPU architecture variables
      _cpuSystemMask = OS::getSystemAffinity();
      _cpuProcessMask = OS::getProcessAffinity();
      _availableCPUs = OS::getMaxProcessors();
      int available_cpus_by_mask = _cpuSystemMask.size();

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

      std::map<int, CpuSet> bindings_list;
      // If --smp-threads-per-core option is used, adjust CPUs used
      if ( _threadsPerCore > 0 ) {
         fatal_cond0( !sys._hwloc.isHwlocAvailable(),
               "Option --smp-threads-per-core requires HWLOC" );

         std::list<CpuSet> core_cpusets = sys._hwloc.getCoreCpusetsOf( _cpuProcessMask );
         fatal_cond0( core_cpusets.size() == 0, "Process mask [" << _cpuProcessMask <<
               "] does not contain a full core cpuset. Cannot run with --smp-threads-per-core" );
         {
            // Append not owned CPUs to core list
            CpuSet cpus_not_owned = _cpuSystemMask - _cpuProcessMask;
            std::list<CpuSet> core_cpusets_not_owned =
               sys._hwloc.getCoreCpusetsOf( cpus_not_owned );
            core_cpusets.splice( core_cpusets.end(), core_cpusets_not_owned );
         }
         for ( std::list<CpuSet>::iterator it = core_cpusets.begin();
               it != core_cpusets.end(); ++it) {
            CpuSet& core = *it;
            unsigned int nthreads = std::min<unsigned int>( _threadsPerCore, core.size() );
            unsigned int offset = core.size() / nthreads;
            unsigned int steps_left = 0;
            unsigned int inserted = 0;
            int last_binding = -1;
            // Iterate core cpuset up to nthreads and add them to the _bindings list
            for ( CpuSet::const_iterator cit = core.begin(); cit != core.end(); ++cit ) {
               int cpuid = *cit;
               // We try to evenly distribute CPUs inside Core
               if ( steps_left == 0 && inserted < nthreads) {
                  _bindings.push_back(cpuid);
                  last_binding = cpuid;
                  bindings_list.insert( std::pair<int,CpuSet>(last_binding, CpuSet(cpuid)) );
                  steps_left += offset - 1;
                  ++inserted;
               } else {
                  --steps_left;
                  bindings_list[last_binding].set(cpuid);
               }
            }
         }
         _availableCPUs = _bindings.size();
      } else {

         //! \note Fill _bindings vector with the active CPUs first, then the not active
         _bindings.reserve( _availableCPUs );
         for ( int i = 0; i < _availableCPUs; i++ ) {
            if ( _cpuProcessMask.isSet(i) ) {
               _bindings.push_back(i);
               bindings_list.insert( std::pair<int,CpuSet>(i, CpuSet(i)) );
            }
         }
         for ( int i = 0; i < _availableCPUs; i++ ) {
            if ( !_cpuProcessMask.isSet(i) ) {
               _bindings.push_back(i);
               bindings_list.insert( std::pair<int,CpuSet>(i, CpuSet(i)) );
            }
         }
      }

      //! \note Load & check NUMA config (_cpus vectors must be created before)
      _cpus = NEW std::vector<SMPProcessor *>( _availableCPUs, (SMPProcessor *) NULL );
      _cpusByCpuId = NEW std::map<int, SMPProcessor *>();

      loadNUMAInfo();

      memory_space_id_t mem_id = sys.getRootMemorySpaceId();
#ifdef MEMKIND_SUPPORT
      if ( _memkindSupport ) {
         mem_id = sys.addSeparateMemoryAddressSpace( ext::getSMPDevice(), _smpAllocWide, sys.getRegionCacheSlabSize() );
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
         memkindMem.setAcceleratorNumber( sys.getNewAcceleratorId() );
      }
#endif

      //! \note Create the SMPProcessors in _cpus array
      int count = 0;
      for ( std::vector<int>::iterator it = _bindings.begin(); it != _bindings.end(); it++ ) {
         int cpuid = *it;
         SMPProcessor *cpu;
         bool active = (count < _currentCPUs) && _cpuProcessMask.isSet(cpuid);
         unsigned numaNode;

         // If this PE can't be seen by hwloc (weird case in Altix 2, for instance)
         if ( !sys._hwloc.isCpuAvailable( cpuid ) ) {
            /* There's a problem: we can't query it's numa
            node. Let's give it 0 (ticket #1090), consider throwing a warning */
            numaNode = 0;
         }
         else
            numaNode = getNodeOfPE( cpuid );
         unsigned socket = numaNode;   /* FIXME: socket */

         memory_space_id_t id;
         if ( _smpPrivateMemory && count >= _smpHostCpus && !_memkindSupport ) {
            OSAllocator a;
            id = sys.addSeparateMemoryAddressSpace( ext::getSMPDevice(),
                  _smpAllocWide, sys.getRegionCacheSlabSize() );
            SeparateMemoryAddressSpace &numaMem = sys.getSeparateMemory( id );
            numaMem.setSpecificData( NEW SimpleAllocator( ( uintptr_t ) a.allocate(_smpPrivateMemorySize), _smpPrivateMemorySize ) );
            numaMem.setAcceleratorNumber( sys.getNewAcceleratorId() );
         } else {
            id = mem_id;
         }

         // Create SMPProcessor object
         cpu = NEW SMPProcessor( cpuid, bindings_list[cpuid], id, active, numaNode, socket );

         if ( active ) {
            _cpuActiveMask.set( cpu->getBindingId() );
         }
         //cpu->setNUMANode( getNodeOfPE( cpu->getId() ) );
         (*_cpus)[count] = cpu;
         (*_cpusByCpuId)[cpuid] = cpu;
         count += 1;
      }

      /*! NOTE: This SMPProcessor will be associated to the master thread. We need to mark
       *        it as reserved to avoid that other architecture plugins reserve it.
       *        Otherwise, SMP and the other will assume that the PE only runs one thread
       *        and they won't notify the over-subscription to the instrumentation plugin.
       */
      SMPProcessor *masterPE = getFirstSMPProcessor();
      masterPE->reserve();
      //NOTE: Disabling next setter as no thread will be created in the PE, only a thread will be associated
      //masterPE->setNumFutureThreads( 1 /* Master thread */ );

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

   unsigned int SMPPlugin::getEstimatedNumThreads() const
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
         count += 1; //< First CPU is reserved in ::init() for the master worker thread
      }
      return count + future_threads;

   }

   unsigned int SMPPlugin::getNumThreads() const
   {
      return ( _idSeed.value() ? _idSeed.value() : getEstimatedNumThreads() );
   }

   ProcessingElement* SMPPlugin::createPE( unsigned id, unsigned uid )
   {
      return NULL;
   }

   void SMPPlugin::initialize() { }

   void SMPPlugin::finalize() {
      if ( _memkindSupport ) {
         SeparateMemoryAddressSpace &mem = sys.getSeparateMemory( 1 );
         std::cerr << "memkind: SMP soft replacements: " << mem.getSoftInvalidationCount() << std::endl;
         std::cerr << "memkind: SMP hard replacements: " << mem.getHardInvalidationCount() << std::endl;
         std::cerr << "memkind: SMP Xfer IN bytes: " << mem.getCache().getTransferredInData() << std::endl;
         std::cerr << "memkind: SMP Xfer OUT bytes: " << mem.getCache().getTransferredOutData() << std::endl;
         std::cerr << "memkind: SMP Xfer OUT (Replacements) bytes: " << mem.getCache().getTransferredReplacedOutData() << std::endl;
         SimpleAllocator *allocator = (SimpleAllocator *) mem.getSpecificData();
         delete allocator;
      } else if ( _smpPrivateMemory ) {
         std::size_t total_in = 0;
         std::size_t total_out = 0;
         for ( std::vector<SMPProcessor *>::const_iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            if ( (*it)->getMemorySpaceId() > 0 ) {
               SeparateMemoryAddressSpace &mem = sys.getSeparateMemory( (*it)->getMemorySpaceId() );
               if ( (*it)->isActive() ) {
                  std::cerr << "PrivateMem: cpu " << (*it)->getId()  << " SMP soft replacements: " << mem.getSoftInvalidationCount() << std::endl;
                  std::cerr << "PrivateMem: cpu " << (*it)->getId()  << " SMP hard replacements: " << mem.getHardInvalidationCount() << std::endl;
                  std::cerr << "PrivateMem: cpu " << (*it)->getId()  << " Xfer IN bytes: " << mem.getCache().getTransferredInData() << std::endl;
                  std::cerr << "PrivateMem: cpu " << (*it)->getId()  << " Xfer OUT bytes: " << mem.getCache().getTransferredOutData() << std::endl;
                  std::cerr << "PrivateMem: cpu " << (*it)->getId()  << " Xfer OUT (Replacements) bytes: " << mem.getCache().getTransferredReplacedOutData() << std::endl;
                  total_in += mem.getCache().getTransferredInData();
                  total_out += mem.getCache().getTransferredOutData();
               }
               SimpleAllocator *allocator = (SimpleAllocator *) mem.getSpecificData();
               delete allocator;
            }
         }
         std::cerr << "Total IN bytes: " << total_in << std::endl;
         std::cerr << "Total OUT bytes: " << total_out << std::endl;
      }
   }

   void SMPPlugin::addPEs( PEMap &pes ) const
   {
      for ( std::vector<SMPProcessor *>::const_iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            pes.insert( std::make_pair( (*it)->getId(), *it ) );
      }
   }

   void SMPPlugin::addDevices( DeviceList &devices ) const
   {
      if ( !_cpus->empty() ) {
         std::vector<const Device *> const &pe_archs = ( *_cpus->begin() )->getDeviceTypes();
         for ( std::vector<const Device *>::const_iterator it = pe_archs.begin();
               it != pe_archs.end(); it++ ) {
            devices.insert( *it );
         }
      }
   }

   void SMPPlugin::startSupportThreads() { }

   void SMPPlugin::startWorkerThreads( std::map<unsigned int, BaseThread *> &workers )
   {
      ensure( _workers.size() == 1, "Main thread should be the only worker created so far." );
      workers.insert( std::make_pair( _workers[0]->getId(), _workers[0] ) );
      //create as much workers as possible
      int available_cpus = 0; /* my cpu is unavailable, numthreads is 1 */
      int active_cpus = 0;
      for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         available_cpus += ( (*it)->getNumThreads() == 0 && !((*it)->isReserved()) && (*it)->isActive() );
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
         if ( _cpuActiveMask.isSet( cpuid )
               && cpu->getNumThreads() == 0
               && !cpu->isReserved() ) {
            _cpuActiveMask.clear( cpuid );
         }
      }
   }

   void SMPPlugin::setRequestedWorkers( int workers )
   {
      _requestedWorkers = workers;
   }

   int SMPPlugin::getRequestedWorkers( void ) const
   {
      return _requestedWorkers;
   }

   ext::SMPProcessor * SMPPlugin::getFirstSMPProcessor() const
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

   ext::SMPProcessor * SMPPlugin::getFirstFreeSMPProcessor() const
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


   ext::SMPProcessor * SMPPlugin::getLastFreeSMPProcessorAndReserve()
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

   ext::SMPProcessor * SMPPlugin::getLastSMPProcessor() {
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

   ext::SMPProcessor * SMPPlugin::getFreeSMPProcessorByNUMAnodeAndReserve(int node)
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

   ext::SMPProcessor * SMPPlugin::getSMPProcessorByNUMAnode(int node, unsigned int idx) const
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

   void SMPPlugin::loadNUMAInfo ()
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

   unsigned SMPPlugin::getNodeOfPE ( unsigned pe )
   {
      if ( sys._hwloc.isHwlocAvailable() ) {
         return sys._hwloc.getNumaNodeOfCpu( pe );
      } else {
         return getNumSockets() - 1;
      }
   }

   void SMPPlugin::setBindingStart ( int value ) { _bindingStart = value; }

   int SMPPlugin::getBindingStart () const { return _bindingStart; }

   void SMPPlugin::setBindingStride ( int value ) { _bindingStride = value;  }

   int SMPPlugin::getBindingStride () const { return _bindingStride; }

   void SMPPlugin::setBinding ( bool set ) { _bindThreads = set; }

   bool SMPPlugin::getBinding () const { return _bindThreads; }

   int SMPPlugin::getNumSockets() const { return _numSockets; }

   void SMPPlugin::setNumSockets ( int numSockets ) { _numSockets = numSockets; }

   int SMPPlugin::getCurrentSocket() const { return _currentSocket; }

   void SMPPlugin::setCurrentSocket( int currentSocket ) { _currentSocket = currentSocket; }

   int SMPPlugin::getCPUsPerSocket() const { return _CPUsPerSocket; }

   void SMPPlugin::setCPUsPerSocket ( int cpus_per_socket ) { _CPUsPerSocket = cpus_per_socket; }

   const CpuSet& SMPPlugin::getCpuProcessMask () const { return _cpuProcessMask; }

   bool SMPPlugin::setCpuProcessMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers )
   {
      bool success = false;
      if ( isValidMask( mask ) ) {
         // The new process mask is copied and assigned as a new active mask
         _cpuProcessMask = mask;
         _cpuActiveMask = mask;
         int master_cpu = workers[0]->getCpuId();
         if ( !sys.getUntieMaster() && !mask.isSet(master_cpu) ) {
            // If master thread is tied and mask does not include master's cpu, force it
            _cpuProcessMask.set( master_cpu );
            _cpuActiveMask.set( master_cpu );
         } else {
            // Return only true success when we have set an unmodified user mask
            success = true;
         }
         applyCpuMask( workers );
      }
      return success;
   }

   void SMPPlugin::addCpuProcessMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers )
   {
      _cpuProcessMask.add( mask );
      _cpuActiveMask.add( mask );
      applyCpuMask( workers );
   }

   const CpuSet& SMPPlugin::getCpuActiveMask () const
   {
      return _cpuActiveMask;
   }

   bool SMPPlugin::setCpuActiveMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers )
   {
      bool success = false;
      if ( isValidMask( mask ) ) {
         _cpuActiveMask = mask;
         int master_cpu = workers[0]->getCpuId();
         if ( !sys.getUntieMaster() && !_cpuActiveMask.isSet(master_cpu) ) {
            // If master thread is tied and mask does not include master's cpu, force it
            _cpuActiveMask.set( master_cpu );
         } else {
            // Return only true success when we have set an unmodified user mask
            success = true;
         }
         applyCpuMask( workers );
      }
      return success;
   }

   void SMPPlugin::addCpuActiveMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers )
   {
      _cpuActiveMask.add( mask );
      applyCpuMask( workers );
   }

   void SMPPlugin::enableCpu ( int cpuid, std::map<unsigned int, BaseThread *> &workers )
   {
      if ( _cpuSystemMask.isSet( cpuid ) && !_cpuActiveMask.isSet( cpuid ) ) {
         _cpuActiveMask.set( cpuid );
         applyCpuMask( workers );
      }
   }

   void SMPPlugin::disableCpu ( int cpuid, std::map<unsigned int, BaseThread *> &workers )
   {
      if ( _cpuActiveMask.isSet( cpuid ) ) {
         _cpuActiveMask.clear( cpuid );
         applyCpuMask( workers );
      }
   }

   void SMPPlugin::updateActiveWorkers ( int nthreads, std::map<unsigned int, BaseThread *> &workers, ThreadTeam *team )
   {
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
      NANOS_INSTRUMENT ( nanos_event_value_t num_threads_val = (nanos_event_value_t) nthreads; )
      NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &num_threads_val); )

      int new_workers = nthreads - _workers.size();

      //! \note Probably it can be relaxed, but at the moment running in a safe mode
      //! waiting for the team to be stable
      while ( !(team->isStable()) ) {
         memoryFence();
      }

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
         BaseThread *thread = *w_it;
         if ( active_threads_checked < nthreads ) {
            thread->lock();
            thread->tryWakeUp( team );
            thread->unlock();
            active_threads_checked++;
         } else {
            thread->lock();
            thread->setLeaveTeam( true );
            thread->sleep();
            thread->unlock();
         }
      }
   }

   void SMPPlugin::updateCpuStatus( int cpuid )
   {
      SMPProcessor *cpu = (*_cpusByCpuId)[cpuid];
      if ( cpu->getRunningThreads() > 0 ) {
         _cpuActiveMask.set( cpuid );
      } else {
         _cpuActiveMask.clear( cpuid );
      }
   }


   void SMPPlugin::admitCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker )
   {

      ext::SMPProcessor *cpu = getFirstFreeSMPProcessor();

      if ( cpu == NULL ) cpu = getFirstSMPProcessor();

      //! \note Create a new Thread object and associate it to the current thread
      BaseThread *thread = &cpu->associateThisThread ( /* untie */ true ) ;

      if ( !isWorker ) return;

      workers.insert( std::make_pair( thread->getId(), thread ) );
      _workers.push_back( ( SMPThread * ) thread );

      //! \note Update current cpu active set mask
      _cpuActiveMask.set( cpu->getBindingId() );

      //! \note Getting Programming Model interface data
      WD &mainWD = *myThread->getCurrentWD();

      mainWD.tieTo(*thread);

      if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
         char *data = NEW char[sys.getPMInterface().getInternalDataSize()];
         sys.getPMInterface().initInternalData( data );
         mainWD.setInternalData( data );
      }

      //! \note Include thread into main thread
      sys.acquireWorker( sys.getMainTeam(), thread, /* enter */ true, /* starring */ false, /* creator */ false );
   }

   void SMPPlugin::expelCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker )
   {
      BaseThread *thread = getMyThreadSafe();

      thread->lock();

      thread->setLeaveTeam(true);
      thread->leaveTeam( );
      thread->unlock();

      if ( isWorker ) {
         workers.erase( thread->getId() );
      }
   }

   int SMPPlugin::getCpuCount() const
   {
      return _cpus->size();
   }

   unsigned int SMPPlugin::getNumPEs() const
   {
      return _currentCPUs;
   }

   unsigned int SMPPlugin::getMaxPEs() const
   {
      return _availableCPUs;
   }

   unsigned int SMPPlugin::getEstimatedNumWorkers() const
   {
      unsigned int count = 0;
      /*if a certain number of workers was requested, pick that value, otherwise
       * pick the number of cpus minus the support threads requested
       */
      if ( _requestedWorkers > 0 ) {
         count = _requestedWorkers;
      } else {
         int active_cpus = 0;
         int reserved_cpus = 0;
         for ( std::vector<SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            active_cpus += (*it)->isActive();
            reserved_cpus += (*it)->isReserved();
         }

         count = active_cpus - reserved_cpus;
         count += 1; //< First CPU is reserved in ::init() for the master worker thread
      }
      debug0( __FUNCTION__ << " called before creating the SMP workers, the estimated number of workers is: " << count);
      return count;
   }

   unsigned int SMPPlugin::getNumWorkers() const
   {
      return _workersCreated ? _workers.size() : getEstimatedNumWorkers();
   }

   //! \brief Get the max number of Workers that could run with the current Active Mask
   unsigned int SMPPlugin::getMaxWorkers() const
   {
      int max_workers = 0;
      for ( std::vector<SMPProcessor *>::const_iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
         if ( (*it)->isActive() ) {
            max_workers += std::max( (*it)->getNumThreads(), static_cast<std::size_t>(1U) );
         }
      }
      return max_workers;
   }

   SMPThread & SMPPlugin::associateThisThread( bool untie )
   {
      SMPThread &thd = getFirstSMPProcessor()->associateThisThread( untie );
      _workers.push_back( &thd );
      return thd;
   }

   /*! \brief Force the creation of at least 1 thread per CPU.
    */
   void SMPPlugin::forceMaxThreadCreation( std::map<unsigned int, BaseThread *> &workers )
   {
      std::vector<ext::SMPProcessor *>::iterator it;
      for ( it = _cpus->begin(); it != _cpus->end(); it++ ) {
         ext::SMPProcessor *target = *it;
         // for each empty PE, create one thread and sleep it
         if ( target->getNumThreads() == 0 ) {
            createWorker( target, workers );
            target->sleepThreads();
         }
      }
   }

   /*! \brief Create a worker in a suitable CPU
    */
   void SMPPlugin::createWorker( std::map<unsigned int, BaseThread *> &workers )
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

   /*! \brief returns human readable strings containing the active and inactive masks
    *
    * The first value of the pair contains the active mask.
    * The second value of the pair contains the inactive mask
    */
   std::pair<std::string, std::string> SMPPlugin::getBindingStrings() const
   {
      CpuSet active, inactive;
      std::string multiple;
      bool multiple_binding = false;
      for ( std::map<int,SMPProcessor*>::const_iterator it = _cpusByCpuId->begin();
            it != _cpusByCpuId->end(); ++it) {
         SMPProcessor *cpu = it->second;
         if ( cpu->isActive() ) {
            // Append binding set to multiple string
            CpuSet binding_list = cpu->getBindingList();
            if ( binding_list.size() > 1 ) {
               multiple_binding = true;
               multiple += "(" + binding_list.toString() + "), ";
            } else {
               multiple += binding_list.toString() + ", ";
            }

            // Add to active cpuset
            active.add( binding_list );
         } else {
            // Add to inactive cpuset
            inactive.add( cpu->getBindingList() );
         }
      }

      // remove last ', ' from the string multiple
      if ( !multiple.empty() ) {
         multiple.resize( multiple.size() - 2 );
      }

      // construct return pair
      std::pair<std::string, std::string> strings;
      strings.first = multiple_binding ? multiple : active.toString();
      strings.second = inactive.toString();

      return strings;
   }

   void SMPPlugin::applyCpuMask ( std::map<unsigned int, BaseThread *> &workers )
   {
      /* OmpSs */
      if ( sys.getPMInterface().isMalleable() ) {
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
         NANOS_INSTRUMENT ( nanos_event_value_t num_threads_val = (nanos_event_value_t ) _cpuActiveMask.size(); )
         NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &num_threads_val); )

         for ( std::vector<ext::SMPProcessor *>::iterator it = _cpus->begin(); it != _cpus->end(); it++ ) {
            ext::SMPProcessor *target = *it;
            int binding_id = target->getBindingId();
            target->setActive( _cpuProcessMask.isSet(binding_id) );
            if ( _cpuActiveMask.isSet(binding_id) ) {

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
            // FIXME: cpuActive or cpuProcess?
            target->setActive( _cpuActiveMask.isSet(binding_id) );
         }
      }
   }

   void SMPPlugin::createWorker( ext::SMPProcessor *target, std::map<unsigned int, BaseThread *> &workers )
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

   bool SMPPlugin::isValidMask( const CpuSet& mask ) const
   {
      // A mask is valid if it shares at least 1 bit with the system mask
      return (mask * _cpuSystemMask).size() > 0;
   }

   bool SMPPlugin::asyncTransfersEnabled() const {
      return _asyncSMPTransfers;
   }

}
}

DECLARE_PLUGIN("arch-smp",nanos::ext::SMPPlugin);
