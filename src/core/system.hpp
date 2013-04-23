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

#ifndef _NANOS_SYSTEM_H
#define _NANOS_SYSTEM_H

#include "system_decl.hpp"
#include <vector>
#include <string>
#include "schedule.hpp"
#include "threadteam.hpp"
#include "slicer.hpp"
#include "nanos-int.h"
#include "dataaccess.hpp"
#include "instrumentation_decl.hpp"
#include "cache_map.hpp"
#include <cmath>


#ifdef HWLOC
#include <hwloc.h>
#endif

using namespace nanos;

// methods to access configuration variable         
inline void System::setNumPEs ( int npes ) { _numPEs = npes; }

inline int System::getNumPEs () const { return _numPEs; }

inline unsigned System::getMaxThreads () const { return _targetThreads; } 

inline void System::setNumThreads ( int nthreads ) { _numThreads = nthreads; }

inline int System::getNumThreads () const { return _numThreads; }

inline int System::getCpuCount () const { return _cpu_mask.size(); };

inline void System::setCpuAffinity(const pid_t pid, size_t cpusetsize, cpu_set_t *mask){
   //ensure( checkCpuMask(mask), "invalid CPU mask set" );
   sched_setaffinity( pid, cpusetsize, mask);
}

inline void System::setDeviceStackSize ( int stackSize ) { _deviceStackSize = stackSize; }

inline int System::getDeviceStackSize () const {return _deviceStackSize; }

inline void System::setBindingStart ( int value ) { _bindingStart = value; }

inline int System::getBindingStart () const { return _bindingStart; }

inline void System::setBindingStride ( int value ) { _bindingStride = value;  }

inline int System::getBindingStride () const { return _bindingStride; }

inline void System::setBinding ( bool set ) { _bindThreads = set; }

inline bool System::getBinding () const { return _bindThreads; }

inline System::ExecutionMode System::getExecutionMode () const { return _executionMode; }

inline bool System::getVerbose () const { return _verboseMode; }

inline void System::setVerbose ( bool value ) { _verboseMode = value; }

inline void System::setInitialMode ( System::InitialMode mode ) { _initialMode = mode; }

inline System::InitialMode System::getInitialMode() const { return _initialMode; }

inline void System::setDelayedStart ( bool set) { _delayedStart = set; }

inline bool System::getDelayedStart () const { return _delayedStart; }

inline bool System::useYield() const { return _useYield; }

inline int System::getTaskNum() const { return _schedStats._totalTasks.value(); }

inline int System::getReadyNum() const { return _schedStats._readyTasks.value(); }

inline int System::getIdleNum() const { return _schedStats._idleThreads.value(); }

inline int System::getRunningTasks() const { return _workers.size() - _schedStats._idleThreads.value(); }

inline void System::setUntieMaster ( bool value ) { _untieMaster = value; }
inline bool System::getUntieMaster () const { return _untieMaster; }

inline void System::setSynchronizedStart ( bool value ) { _synchronizedStart = value; }
inline bool System::getSynchronizedStart ( void ) const { return _synchronizedStart; }

inline int System::getWorkDescriptorId( void ) { return _atomicWDSeed++; }

inline int System::getNumWorkers() const { return _workers.size(); }

inline int System::getNumSockets() const { return _numSockets; }
inline void System::setNumSockets ( int numSockets ) { _numSockets = numSockets; }

inline int System::getCurrentSocket() const { return _currentSocket; }
inline void System::setCurrentSocket( int currentSocket ) { _currentSocket = currentSocket; }

inline int System::getCoresPerSocket() const { return _coresPerSocket; }
inline void System::setCoresPerSocket ( int coresPerSocket ) { _coresPerSocket = coresPerSocket; }

inline int System::getBindingId ( int pe ) const
{
   int tmpId = ( pe * getBindingStride() + getBindingStart() ) % _pe_map.size();
   return _pe_map[tmpId];
}

inline bool System::isHwlocAvailable () const
{
#ifdef HWLOC
   return true;
#else
   return false;
#endif
}

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
   
   // Read the number of numa nodes if the user didn't set that value
   if ( _numSockets == 0 )
   {
      int depth = hwloc_get_type_depth( topology, HWLOC_OBJ_NODE );
      
      // If there are NUMA nodes in this machine
      if ( depth != HWLOC_TYPE_DEPTH_UNKNOWN ) {
         _numSockets = hwloc_get_nbobjs_by_depth( topology, depth );
      }
      // Otherwise, set it to 1
      else
         _numSockets = 1;
   }
   
   // Same thing, just change the value if the user didn't provide one
   if( _coresPerSocket == 0 )
   {
      int depth = hwloc_get_type_depth( topology, HWLOC_OBJ_PU );
      if ( depth != HWLOC_TYPE_DEPTH_UNKNOWN ) {
         _coresPerSocket = hwloc_get_nbobjs_by_depth( topology, depth ) / _numSockets;
      }
   }
   
#else
   // Number of sockets can be read with
   // cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l
   // Cores per socket:
   // cat /proc/cpuinfo | grep 'core id' | sort | uniq | wc -l
   
   // Assume just 1 socket
   if ( _numSockets == 0 )
      _numSockets = 1;
   if ( _coresPerSocket == 0 )
      _coresPerSocket = std::ceil( _targetThreads / static_cast<float>( _numSockets ) );
#endif
}

inline void System::checkArguments()
{
   // Check NUMA config
   if ( _numSockets != std::ceil( _targetThreads / static_cast<float>( _coresPerSocket ) ) )
   {
      unsigned validCoresPS = std::ceil( _targetThreads / static_cast<float>( _numSockets ) );
      warning0( "Adjusting cores-per-socket from " << _coresPerSocket << " to " << validCoresPS );
      _coresPerSocket = validCoresPS;
   }
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

inline bool System::isCpuidEventEnabled ( void ) const { return _enable_cpuid_event; }

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

inline void System::setPMInterface(PMInterface *pm)
{
   ensure0(!_pmInterface,"PM interface already in place!");
   _pmInterface = pm;
}

inline PMInterface &  System::getPMInterface(void) const { return *_pmInterface; }

inline bool System::isCacheEnabled() { return _useCaches; }

inline System::CachePolicyType System::getCachePolicy() { return _cachePolicy; }

inline CacheMap& System::getCacheMap() { return _cacheMap; }

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

inline bool System::haveDependencePendantWrites ( void *addr ) const
{
   return myThread->getCurrentWD()->getDependenciesDomain().haveDependencePendantWrites ( addr );
}

#endif

