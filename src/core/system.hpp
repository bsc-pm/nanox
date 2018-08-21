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

#ifndef _NANOS_SYSTEM_H
#define _NANOS_SYSTEM_H

#include "system_decl.hpp"
#include "dependenciesdomain_decl.hpp"
#include <vector>
#include <string>
#include "schedule_decl.hpp"
#include "threadteam.hpp"
#include "slicer.hpp"
#include "nanos-int.h"
#include "dataaccess.hpp"
#include "instrumentation_decl.hpp"
#include "synchronizedcondition.hpp"
#include "regioncache.hpp"
#include <cmath>
#include <climits>


namespace nanos {

// methods to access configuration variable
//inline void System::setNumPEs ( int npes ) { _numPEs = npes; }
//
//inline int System::getNumPEs () const { return _numPEs; }
//
//inline unsigned System::getMaxThreads () const { return _targetThreads; }
//
//inline void System::setNumThreads ( int nthreads ) { _numThreads = nthreads; }
//
//inline int System::getNumThreads () const { return _numThreads; }

inline DeviceList & System::getSupportedDevices() { return _devices; }

inline void System::setDeviceStackSize ( size_t stackSize ) { _deviceStackSize = stackSize; }

inline size_t System::getDeviceStackSize () const {return _deviceStackSize; }

inline System::ExecutionMode System::getExecutionMode () const { return _executionMode; }

inline bool System::getVerbose () const { return _verboseMode; }

inline void System::setVerbose ( bool value ) { _verboseMode = value; }

inline void System::setInitialMode ( System::InitialMode mode ) { _initialMode = mode; }

inline System::InitialMode System::getInitialMode() const { return _initialMode; }

inline void System::setDelayedStart ( bool set) { _delayedStart = set; }

inline bool System::getDelayedStart () const { return _delayedStart; }

inline int System::getCreatedTasks() const { return _schedStats._createdTasks.value(); }

inline int System::getTaskNum() const { return _schedStats._totalTasks.value(); }

inline int System::getReadyNum() const { return _schedStats._readyTasks.value(); }

inline int System::getIdleNum() const { return _schedStats._idleThreads.value(); }

inline int System::getRunningTasks() const { return _workers.size() - _schedStats._idleThreads.value(); }

inline void System::setUntieMaster ( bool value ) { _untieMaster = value; }
inline bool System::getUntieMaster () const { return _untieMaster; }

inline void System::setSynchronizedStart ( bool value ) { _synchronizedStart = value; }
inline bool System::getSynchronizedStart ( void ) const { return _synchronizedStart; }

inline void System::setPredecessorLists ( bool value ) { _predecessorLists = value; }
inline bool System::getPredecessorLists ( void ) const { return _predecessorLists; }

inline int System::getWorkDescriptorId( void ) { return _atomicWDSeed++; }

inline int System::getNumWorkers() const { return _workers.size(); }

inline int System::getNumCreatedPEs() const { return _pes.size(); }

inline System::ThreadList::iterator System::getWorkersBegin() { return _workers.begin(); }
inline System::ThreadList::iterator System::getWorkersEnd() { return _workers.end(); }

//inline int System::getNumSockets() const { return _numSockets; }
//inline void System::setNumSockets ( int numSockets ) { _numSockets = numSockets; }
//
//inline int System::getNumAvailSockets() const
//{
//   return _numAvailSockets;
//}
//
inline int System::getVirtualNUMANode( int physicalNode ) const
{
   return ( physicalNode < (int)_numaNodeMap.size() ) ? _numaNodeMap[ physicalNode ] : INT_MIN;
}

inline const std::vector<int> & System::getNumaNodeMap() const
{
	return _numaNodeMap;
}

//
//inline int System::getCurrentSocket() const { return _currentSocket; }
//inline void System::setCurrentSocket( int currentSocket ) { _currentSocket = currentSocket; }
//
//inline int System::getCoresPerSocket() const { return _coresPerSocket; }
//inline void System::setCoresPerSocket ( int coresPerSocket ) { _coresPerSocket = coresPerSocket; }

//inline int System::getBindingId ( int pe ) const
//{
//   return _bindings[ pe % _bindings.size() ];
//}

//inline bool System::isHwlocAvailable () const
//{
//#ifdef HWLOC
//   return true;
//#else
//   return false;
//#endif
//}

#if 0
inline void System::loadHwloc ()
{
#ifdef HWLOC
   // Allocate and initialize topology object.
   hwloc_topology_init( ( hwloc_topology_t* )&_hwlocTopology );

   // If the user provided an alternate topology
   if ( !_topologyPath.empty() )
   {
      int res = hwloc_topology_set_xml( ( hwloc_topology_t ) _hwlocTopology, _topologyPath.c_str() );
      fatal_cond0( res != 0, "Could not load hwloc topology xml file." );
   }

   // Enable GPU detection
   hwloc_topology_set_flags( ( hwloc_topology_t ) _hwlocTopology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES );

   // Perform the topology detection.
   hwloc_topology_load( ( hwloc_topology_t ) _hwlocTopology );
#endif
}

inline void System::loadNUMAInfo ()
{
#ifdef HWLOC
   hwloc_topology_t topology = ( hwloc_topology_t ) _hwlocTopology;

   // Nodes that can be seen by hwloc
   unsigned allowedNodes = 0;
   // Hardware threads
   unsigned hwThreads = 0;

   // Read the number of numa nodes if the user didn't set that value
   if ( _numSockets == 0 )
   {
      int depth = hwloc_get_type_depth( topology, HWLOC_OBJ_NODE );


      // If there are NUMA nodes in this machine
      if ( depth != HWLOC_TYPE_DEPTH_UNKNOWN ) {
         //hwloc_const_cpuset_t cpuset = hwloc_topology_get_online_cpuset( topology );
         //allowedNodes = hwloc_get_nbobjs_inside_cpuset_by_type( topology, cpuset, HWLOC_OBJ_NODE );
         //hwThreads = hwloc_get_nbobjs_inside_cpuset_by_type( topology, cpuset, HWLOC_OBJ_PU );
         unsigned nodes = hwloc_get_nbobjs_by_depth( topology, depth );
         //hwloc_cpuset_t set = i

         // For each node, count how many hardware threads there are below.
         for ( unsigned nodeIdx = 0; nodeIdx < nodes; ++nodeIdx )
         {
            hwloc_obj_t node = hwloc_get_obj_by_depth( topology, depth, nodeIdx );
            int localThreads = hwloc_get_nbobjs_inside_cpuset_by_type( topology, node->cpuset, HWLOC_OBJ_PU );
            // Increase hw thread count
            hwThreads += localThreads;
            // If this node has hw threads beneath, increase the number of viewable nodes
            if ( localThreads > 0 ) ++allowedNodes;
         }
         _numSockets = nodes;
      }
      // Otherwise, set it to 1
      else {
         allowedNodes = 1;
         _numSockets = 1;
      }
   }

   if( _coresPerSocket == 0 )
      _coresPerSocket = std::ceil( hwThreads / static_cast<float>( allowedNodes ) );
#else
   // Number of sockets can be read with
   // cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l
   // Cores per socket:
   // cat /proc/cpuinfo | grep 'core id' | sort | uniq | wc -l

   // Assume just 1 socket
   if ( _numSockets == 0 )
      _numSockets = 1;

   // Same thing, just change the value if the user didn't provide one
   if ( _coresPerSocket == 0 )
      _coresPerSocket = std::ceil( _targetThreads / static_cast<float>( _numSockets ) );
#endif
   verbose0( toString( "[NUMA] " ) + toString( _numSockets ) + toString( " NUMA nodes, " ) + toString( _coresPerSocket ) + toString( " HW threads each." ) );
}

inline void System::completeNUMAInfo()
{
   // Create the NUMA node translation table. Do this before creating the team,
   // as the schedulers might need the information.
   _numaNodeMap.resize( _numSockets, INT_MIN );

   /* As all PEs are already created by this time, count how many physical
    * NUMA nodes are available, and map from a physical id to a virtual ID
    * that can be selected by the user via nanos_current_socket() */
   for ( PEMap::const_iterator it = _pes.begin(); it != _pes.end(); ++it )
   {
      int node = (*it)->getNUMANode();
      // If that node has not been translated, yet
      if ( _numaNodeMap[ node ] == INT_MIN )
      {
         verbose0( "Mapping from physical node " << node << " to user node " << _numAvailSockets );
         _numaNodeMap[ node ] = _numAvailSockets;
         // Increase the number of available sockets
         ++_numAvailSockets;
      }
      // Otherwise, do nothing
   }

   verbose0( _numAvailSockets << " NUMA node(s) available for the user." );

}

inline void System::unloadHwloc ()
{
#ifdef HWLOC
   /* Destroy topology object. */
   hwloc_topology_destroy( ( hwloc_topology_t )_hwlocTopology );
#endif
}

inline unsigned System::getNodeOfPE ( unsigned pe )
{
#ifdef HWLOC
   // cast just once
   hwloc_topology_t topology = ( hwloc_topology_t ) _hwlocTopology;

   hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index( topology, pe );

   // Now we have the PU object, go find its parent numa node
   hwloc_obj_t numaNode =
      hwloc_get_ancestor_obj_by_type( topology, HWLOC_OBJ_NODE, pu );

   // If the machine is not NUMA
   if ( numaNode == NULL )
      return 0;

   return numaNode->os_index;
#else
   // Dirty way, will not work with hyperthreading
   // Use /sys/bus/cpu/devices/cpuX/
   //return pe / getCoresPerSocket();

   // Otherwise, return
   return sys.getNumSockets() - 1;
#endif
}
#endif

inline void System::setThrottlePolicy( ThrottlePolicy * policy ) { _throttlePolicy = policy; }

inline const std::string & System::getDefaultSchedule() const { return _defSchedule; }

inline const std::string & System::getDefaultThrottlePolicy() const { return _defThrottlePolicy; }

inline const std::string & System::getDefaultBarrier() const { return _defBarr; }

inline const std::string & System::getDefaultInstrumentation() const { return _defInstr; }

inline void System::setHostFactory ( peFactory factory ) { _hostFactory = factory; }

inline void System::setDefaultBarrFactory ( barrFactory factory ) { _defBarrFactory = factory; }

inline Slicer * System::getSlicer( const std::string &label ) const
{
   Slicers::const_iterator it = _slicers.find(label);
   if ( it == _slicers.end() ) return NULL;
   return (*it).second;
}

inline WorkSharing * System::getWorkSharing( const std::string &label ) const
{
   WorkSharings::const_iterator it = _worksharings.find(label);
   if ( it == _worksharings.end() ) return NULL;
   return (*it).second;
}

inline Instrumentation * System::getInstrumentation ( void ) const { return _instrumentation; }

inline void System::setInstrumentation ( Instrumentation *instr ) { _instrumentation = instr; }

#ifdef NANOS_INSTRUMENTATION_ENABLED
inline bool System::isCpuidEventEnabled ( void ) const { return _enableCpuidEvent; }
#endif

inline void System::registerSlicer ( const std::string &label, Slicer *slicer) { _slicers[label] = slicer; }

inline void System::registerWorkSharing ( const std::string &label, WorkSharing *ws) { _worksharings[label] = ws; }

inline void System::setDefaultSchedulePolicy ( SchedulePolicy *policy ) { _defSchedulePolicy = policy; }
inline SchedulePolicy * System::getDefaultSchedulePolicy ( ) const  { return _defSchedulePolicy; }

inline SchedulerStats & System::getSchedulerStats () { return _schedStats; }
inline SchedulerConf  & System::getSchedulerConf ()  { return _schedConf; }

inline void System::stopScheduler ()
{
   myThread->pause();
   _schedConf.setSchedulerEnabled( false );
}

inline void System::startScheduler ()
{
   myThread->unpause();
   _schedConf.setSchedulerEnabled( true );
}

inline bool System::isSchedulerStopped () const
{
   return _schedConf.getSchedulerEnabled();
}

inline void System::pausedThread ()
{
   _pausedThreadsCond.reference();
   _unpausedThreadsCond.reference();
   ++_pausedThreads;
   if ( _pausedThreadsCond.check() ) {
      _pausedThreadsCond.signal();
   }
   _pausedThreadsCond.unreference();
   _unpausedThreadsCond.unreference();
}

inline void System::unpausedThread ()
{
   _pausedThreadsCond.reference();
   _unpausedThreadsCond.reference();
   // TODO (#582): Do we need a reference and unreference block here?
   --_pausedThreads;
   if ( _unpausedThreadsCond.check() ) {
      _unpausedThreadsCond.signal();
   }
   _unpausedThreadsCond.unreference();
   _pausedThreadsCond.unreference();
}

inline const std::string & System::getDefaultDependenciesManager() const
{
   return _defDepsManager;
}

inline void System::setDependenciesManager ( DependenciesManager *manager )
{
   _dependenciesManager = manager;
}

inline DependenciesManager * System::getDependenciesManager ( ) const
{
   return _dependenciesManager;
}

inline const std::string & System::getDefaultArch() const { return _defArch; }
inline void System::setDefaultArch( const std::string &arch ) { _defArch = arch; }

inline Network * System::getNetwork( void ) { return &_net; }
inline bool System::usingCluster( void ) const { return _usingCluster; }
inline bool System::usingClusterMPI( void ) const { return _usingClusterMPI; }
inline bool System::useNode2Node( void ) const { return _usingNode2Node; }
inline bool System::usePacking( void ) const { return _usingPacking; }
inline const std::string & System::getNetworkConduit( void ) const { return _conduit; }

inline void System::setPMInterface(PMInterface *pm)
{
   ensure0(!_pmInterface,"PM interface already in place!");
   _pmInterface = pm;
}

inline PMInterface &  System::getPMInterface(void) const { return *_pmInterface; }

inline size_t System::registerArchitecture( ArchPlugin * plugin )
{
   size_t id = _archs.size();
   _archs.push_back( plugin );
   return id;
}

#ifdef GPU_DEV
inline PinnedAllocator& System::getPinnedAllocatorCUDA() { return _pinnedMemoryCUDA; }
#endif


inline bool System::throttleTaskIn ( void ) const { return _throttlePolicy->throttleIn(); }
inline void System::throttleTaskOut ( void ) const { _throttlePolicy->throttleOut(); }

inline void System::threadReady()
{
   _initializedThreads++;

   /*! It's better not to call Scheduler::waitOnCondition here as the initialization is not
       yet finished

      TODO: we can consider thread yielding */
   while (_initializedThreads.value() < _targetThreads) {}
}

inline void System::registerPlugin ( const char *name, Plugin &plugin )
{
   _pluginManager.registerPlugin(name, plugin);
}

inline bool System::loadPlugin ( const char * name )
{
   return _pluginManager.load(name);
}

inline bool System::loadPlugin ( const std::string & name )
{
   return _pluginManager.load(name);
}

inline Plugin * System::loadAndGetPlugin ( const char *name )
{
   return _pluginManager.loadAndGetPlugin(name, false);
}

inline Plugin * System::loadAndGetPlugin ( const std::string & name )
{
   return _pluginManager.loadAndGetPlugin(name, false);
}

inline int System::getWgId() { return _atomicSeedWg++; }
inline unsigned int System::getRootMemorySpaceId() { return 0; }

inline ProcessingElement &System::getPEWithMemorySpaceId( memory_space_id_t id ) {
   bool found = false;
   PE *target = NULL;
   for ( PEMap::iterator it = _pes.begin(); it != _pes.end() && !found; it++ ) {
      if ( it->second->getMemorySpaceId() == id ) {
         target = it->second;
         found = true;
      }
   }
   return *target;
}

inline void System::setValidPlugin ( const std::string &module,  const std::string &plugin )
{
   _validPlugins.insert( make_pair( module, plugin ) );
}

inline void System::registerPluginOption ( const std::string &option, const std::string &module,
                                          std::string &var, const std::string &helpMessage,
                                          Config& cfg )
{
   if ( !_validPlugins.empty() ) {
      // Get the list of valid plugins
      Config::PluginVar* pluginVar = NEW Config::PluginVar( _defDepsManager, NULL, 0 );
      ModulesPlugins::const_iterator it;
      // Find deps
      std::pair<ModulesPlugins::const_iterator, ModulesPlugins::const_iterator> ret
         = _validPlugins.equal_range( module );

      // For each deps plugin, add it as an option
      for ( it = ret.first; it != ret.second; ++it ){
         pluginVar->addOption( it->second );
      }

      cfg.registerConfigOption ( option, pluginVar, helpMessage );
   }
   else {
      cfg.registerConfigOption ( option, NEW Config::StringVar( var ), helpMessage );
   }
}

inline int System::nextThreadId () { return _threadIdSeed++; }

inline unsigned int System::nextPEId () { return _peIdSeed++; }

inline Lock * System::getLockAddress ( void *addr ) const { return &_lockPool[((((uintptr_t)addr)>>8)%_lockPoolSize)];} ;

inline bool System::haveDependencePendantWrites ( void *addr ) const
{
   return myThread->getCurrentWD()->getDependenciesDomain().haveDependencePendantWrites ( addr );
}

inline int System::getTaskMaxRetries() const { return _task_max_retries; }

inline void System::setSMPPlugin(SMPBasePlugin *p) {
   _smpPlugin = p;
}

inline SMPBasePlugin *System::getSMPPlugin() const {
   return _smpPlugin;
}

inline bool System::isSimulator() const {
   return _simulator;
}

inline ThreadTeam *System::getMainTeam() {
   return _mainTeam;
}

inline void System::setVerboseDevOps(bool value) {
   _verboseDevOps = value;
}

inline bool System::getVerboseDevOps() const {
   return _verboseDevOps;
}

inline void System::setVerboseCopies(bool value) {
   _verboseCopies = value;
}

inline bool System::getVerboseCopies() const {
   return _verboseCopies;
}

inline bool System::getSplitOutputForThreads() const {
   return _splitOutputForThreads;
}

inline std::string System::getRegionCachePolicyStr() const {
   return _regionCachePolicyStr;
}
inline void System::setRegionCachePolicyStr( std::string policy ) {
   _regionCachePolicyStr = policy;
}

inline RegionCache::CachePolicy System::getRegionCachePolicy() const {
   return _regionCachePolicy;
}

inline std::size_t System::getRegionCacheSlabSize() const {
   return _regionCacheSlabSize;
}

inline void System::createDependence( WD* pred, WD* succ)
{
   DOSubmit *pred_do = pred->getDOSubmit(), *succ_do = succ->getDOSubmit();
   pred_do->addSuccessor(*succ_do);
   succ_do->increasePredecessors();
}

inline unsigned int System::getNumClusterNodes() const {
   return _clusterNodes.size();
}

inline unsigned int System::getNumNumaNodes() const {
   return _numaNodes.size();
}

inline std::set<unsigned int> const &System::getClusterNodeSet() const {
   return _clusterNodes;
}

inline memory_space_id_t System::getMemorySpaceIdOfClusterNode( unsigned int node ) const {
   memory_space_id_t id = 0;
   if ( node != 0 ) {
      for ( PEMap::const_iterator it = _pes.begin(); it != _pes.end(); it++ ) {
         if ( it->second->getClusterNode() == node ) {
            id = it->second->getMemorySpaceId();
         }
      }
   }
   return id;
}

inline int System::getUserDefinedNUMANode() const {
   return _userDefinedNUMANode;
}

inline void System::setUserDefinedNUMANode( int nodeId ) {
   _userDefinedNUMANode = nodeId;
}

inline unsigned int System::getNumAccelerators() const {
   return _acceleratorCount;
}

inline unsigned int System::getNewAcceleratorId() {
   return _acceleratorCount++;
}

inline ThreadManagerConf& System::getThreadManagerConf() {
   return _threadManagerConf;
}

inline ThreadManager* System::getThreadManager() const {
   return _threadManager;
}

inline bool System::getPrioritiesNeeded() const {
   return _compilerSuppliedFlags.prioritiesNeeded;
}

/* SMPPlugin functions */
inline void System::admitCurrentThread( bool isWorker ) { _smpPlugin->admitCurrentThread( _workers, isWorker ); }
inline void System::expelCurrentThread( bool isWorker ) { _smpPlugin->expelCurrentThread( _workers, isWorker ); }

inline void System::updateActiveWorkers( int nthreads ) { _smpPlugin->updateActiveWorkers( nthreads, _workers, myThread->getTeam() ); }

inline const CpuSet& System::getCpuProcessMask() const { return _smpPlugin->getCpuProcessMask(); }
inline bool System::setCpuProcessMask( const CpuSet& mask ) { return _smpPlugin->setCpuProcessMask( mask, _workers ); }
inline void System::addCpuProcessMask( const CpuSet& mask ) { _smpPlugin->addCpuProcessMask( mask, _workers ); }

inline const CpuSet& System::getCpuActiveMask() const { return _smpPlugin->getCpuActiveMask(); }
inline bool System::setCpuActiveMask( const CpuSet& mask ) { return _smpPlugin->setCpuActiveMask( mask, _workers ); }
inline void System::addCpuActiveMask( const CpuSet& mask ) { _smpPlugin->addCpuActiveMask( mask, _workers ); }
inline void System::enableCpu( int cpuid ) { _smpPlugin->enableCpu( cpuid, _workers ); }
inline void System::disableCpu( int cpuid ) { _smpPlugin->disableCpu( cpuid, _workers ); }

inline void System::forceMaxThreadCreation() { _smpPlugin->forceMaxThreadCreation( _workers ); }

inline memory_space_id_t System::getMemorySpaceIdOfAccelerator( unsigned int accelerator_id ) const {
   memory_space_id_t id = ( memory_space_id_t ) -1;
   for ( memory_space_id_t mem_idx = 1; mem_idx < _separateMemorySpacesCount; mem_idx += 1 ) {
      if ( _separateAddressSpaces[ mem_idx ]->getAcceleratorNumber() == accelerator_id ) {
         id = mem_idx;
         break;
      }
   }
   return id;
}

inline Router &System::getRouter() {
   return _router;
}

inline bool System::isImmediateSuccessorEnabled() const {
   return !_immediateSuccessorDisabled;
}

inline bool System::usePredecessorCopyInfo() const {
   return !_predecessorCopyInfoDisabled;
}

inline bool System::invalControlEnabled() const {
   return _invalControl;
}

inline std::set<memory_space_id_t> const &System::getActiveMemorySpaces() const {
   return _activeMemorySpaces;
}

inline PEMap& System::getPEs()  {
   return _pes;
}

inline void System::allocLock() {
   while ( !_allocLock.tryAcquire() ) {
      myThread->processTransfers();
   }
}

inline void System::allocUnlock() {
   _allocLock.release();
}

inline bool System::useFineAllocLock() const {
   return !_cgAlloc;
}

inline SMPDevice &System::_getSMPDevice() {
   return _SMP;
}

} // namespace nanos

#endif
