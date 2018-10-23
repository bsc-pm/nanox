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

#include "threadmanager_decl.hpp"
#include "atomic_decl.hpp"
#include "lock_decl.hpp"
#include "system.hpp"
#include "config.hpp"
#include "os.hpp"

#ifdef DLB
#include <dlb.h>
#endif

using namespace nanos;

ThreadManager::ThreadManager( bool warmup, bool tie_master, unsigned int num_yields,
      unsigned int sleep_time, bool use_sleep, bool use_block, bool use_dlb ) :
   _lock(),
   _initialized( false ),
   _maxThreads(),
   _cpuProcessMask( sys.getCpuProcessMask() ),
   _cpuActiveMask( sys.getCpuActiveMask() ),
   _isMalleable(),
   _warmupThreads( warmup ),
   _numYields( num_yields ),
   _sleepTime( sleep_time ),
   _useSleep( use_sleep ),
   _useBlock( use_block ),
   _useDLB( use_dlb ),
   _self_managed_cpus()
{
}

ThreadManager::~ThreadManager()
{
#ifdef DLB
   if ( _initialized && _useDLB ) DLB_Finalize();
#endif
}

void ThreadManager::init()
{
   _isMalleable = sys.getPMInterface().isMalleable();

   if ( _warmupThreads ) {
      sys.forceMaxThreadCreation();
   }

#ifdef DLB
   if ( _useDLB ) {
      int err = DLB_Init( 0, &_cpuProcessMask, NULL );
      if (err == DLB_SUCCESS) {
         sys.getPMInterface().registerCallbacks();
         _maxThreads = OS::getMaxProcessors();
      } else {
         warning0( "DLB Init failed: " << DLB_Strerror(err) );
         _useDLB = false;
      }
   }
   if ( !_useDLB ) {
#endif
      _maxThreads = sys.getSMPPlugin()->getRequestedWorkers();
#ifdef DLB
   }
#endif

   // Consider TM not initialized if there isn't any related flag
   _initialized = _useSleep || _useBlock || _useDLB;
}

bool ThreadManager::isGreedy()
{
   if ( !_initialized ) return false;
   if ( !_isMalleable ) return false;
   return _cpuActiveMask.size() < _maxThreads;
}

bool ThreadManager::lastActiveThread()
{
   //! \note We omit the test if the Thread Manager is not yet initializated
   if ( !_initialized ) return false;

   //! \note We omit the test if the cpu does not belong to my process_mask
   BaseThread *thread = getMyThreadSafe();
   int my_cpu = thread->getCpuId();
   if ( !_cpuProcessMask.isSet( my_cpu ) ) return false;

   //! \note Getting initial process mask (not yielded threads) having into account
   //!       these currently active. Checking if the list of processor have only one
   //!       single active processor (and this processor is my cpu).
   CpuSet mine_and_active = _cpuProcessMask & _cpuActiveMask;
   bool last = mine_and_active.size() == 1 && mine_and_active.isSet(my_cpu);

   //! \note Watch out if we have oversubscription. As the current thread may be already
   //        marked as leaving the processor we need to take that into account
   last &= thread->runningOn()->getRunningThreads() <= (thread->isSleeping()? 0:1) ;

   return last;
}

void ThreadManager::idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
      , unsigned long long& total_yields, unsigned long long& total_blocks
      , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
      )
{
   if ( !_initialized ) return;

#ifdef DLB
   if ( _useDLB ) DLB_PollDROM_Update();
#endif

   BaseThread *thread = getMyThreadSafe();

   if ( yields > 0 ) {
#ifdef NANOS_INSTRUMENTATION_ENABLED
      total_yields++;
      double begin_yield = OS::getMonotonicTime();
#endif
      thread->yield();
#ifdef NANOS_INSTRUMENTATION_ENABLED
      double end_yield = OS::getMonotonicTime();
      time_yields += (unsigned long long) ( (end_yield - begin_yield) * 1e9 );
#endif
      --yields;
   } else {
      if ( _useBlock && thread->canBlock() ) {
#ifdef NANOS_INSTRUMENTATION_ENABLED
         total_blocks++;
         double begin_block = OS::getMonotonicTime();
#endif
         blockThread( thread );
#ifdef NANOS_INSTRUMENTATION_ENABLED
         double end_block = OS::getMonotonicTime();
         time_blocks += (unsigned long long) ( (end_block - begin_block) * 1e9 );
#endif
      } else if ( _useSleep ) {
         OS::nanosleep( _sleepTime );
      }
      yields = _numYields;
   }
}

void ThreadManager::blockThread( BaseThread *thread )
{
   if ( !_initialized ) return;
   if ( thread->isSleeping() ) return;

   // Do not release master thread if master WD is tied
   if ( thread->isMainThread() && !sys.getUntieMaster() ) return;

   int my_cpu = thread->getCpuId();

   LockBlock lock( _lock );

   // Do not release if this CPU is the last active within the process_mask
   CpuSet mine_and_active = _cpuProcessMask & _cpuActiveMask;
   if ( mine_and_active.size() == 1 && mine_and_active.isSet(my_cpu) )
      return;

   thread->lock();
   if ( _isMalleable ) {
      // OmpSs threads are inconditionally blocked if they reach here
      thread->sleep();
   } else {
      // OpenMP threads are only blocked if they are teamless
      if ( !thread->hasTeam() ) {
         thread->sleep();
      }
   }
   thread->unlock();

   // Clear CPU from active mask when all threads of a process are blocked
   sys.getSMPPlugin()->updateCpuStatus( my_cpu );
}

void ThreadManager::unblockThread( BaseThread* thread )
{
   if ( !_initialized ) return;
   if ( !_isMalleable ) return;
   if ( !thread->isSleeping() ) return;

   int cpuid = thread->getCpuId();

   LockBlock lock( _lock );

#ifdef DLB
   if ( _useDLB ) {
      std::deque<int>::iterator it =
         std::find( _self_managed_cpus.begin(), _self_managed_cpus.end(), cpuid );
      if( it != _self_managed_cpus.end() ) {
         /* CPU was lent while DLB was disabled, remove it from the deque
          * and continue waking up thread as if DLB were not involved
          */
         _self_managed_cpus.erase( it );
      } else {
         int dlb_err = DLB_AcquireCpu( cpuid );
         if ( dlb_err == DLB_SUCCESS || dlb_err == DLB_NOTED ) {
            /* if DLB_SUCCESS the CPU has been successfully acquired
             * if DLB_NOTED the CPU has been reclaimed, but also acquired
             *    and the target thread will manage the oversubscription
             */
            return;
         } else if ( dlb_err == DLB_NOUPDT || dlb_err == DLB_ERR_DISBLD ) {
            /* if DLB_NOUPDT the CPU was already assigned to this process
             * if DLB_ERR_DISBLD the petition was ignored
             * in both cases we continue as if DLB were not involved
             */
         } else {
            warning( "DLB returned error: " << DLB_Strerror(dlb_err) << " for cpuid " << cpuid );

         }
      }
   }
#endif

   thread->lock();
   thread->tryWakeUp( myThread->getTeam() );
   thread->unlock();
   sys.getSMPPlugin()->updateCpuStatus( cpuid );
}

void ThreadManager::lendCpu( BaseThread *thread )
{
#ifdef DLB
   // Lend CPU only if my_cpu has been cleared from the active mask
   int my_cpu = thread->getCpuId();
   if ( _useDLB ) {
      LockBlock lock( _lock );
      if ( !_cpuActiveMask.isSet(my_cpu) ) {
         int dlb_err = DLB_LendCpu( my_cpu );
         if ( dlb_err == DLB_ERR_DISBLD && _cpuProcessMask.isSet(my_cpu) ) {
            /* If DLB is disabled at this point, we need to keep track
             * of lent CPUs since DLB will be unaware of them
             */
            _self_managed_cpus.push_front( my_cpu );
         }
      }
   }
#endif
}

void ThreadManager::acquireOne()
{
   if ( !_initialized ) return;
   if ( !_isMalleable ) return;
   if ( _cpuActiveMask.size() >= _maxThreads ) return;

   ThreadTeam *team = getMyThreadSafe()->getTeam();
   if ( !team ) return;

   // Acquire CPUs is a best effort optimization within the critical path,
   // do not wait for a lock release if the lock is busy
   if ( _lock.tryAcquire() ) {

#ifdef DLB
      if ( _useDLB ) {
         if ( !_self_managed_cpus.empty() ) {
            // If there are CPUs lent while DLB was disabled, acquire one of them
            int cpuid = _self_managed_cpus.front();
            _self_managed_cpus.pop_front();
            CpuSet new_active_cpus = _cpuActiveMask;
            new_active_cpus.set( cpuid );
            sys.setCpuActiveMask( new_active_cpus );
         } else {
            // Otherwise ask DLB
            DLB_AcquireCpus( 1 );
         }
      } else {
#endif
         // Otherwise, we acquire one CPU from our process mask
         CpuSet new_active_cpus = _cpuActiveMask;
         CpuSet mine_and_active = _cpuProcessMask & _cpuActiveMask;

         // Check first that we have some owned CPU not active
         if ( mine_and_active != _cpuProcessMask ) {
            // Iterate over default cpus not running and wake them up if needed
            for ( CpuSet::const_iterator it=_cpuProcessMask.begin();
                  it!=_cpuProcessMask.end(); ++it ) {
               int cpuid = *it;
               if ( !new_active_cpus.isSet( cpuid ) ) {
                  new_active_cpus.set( cpuid );
                  sys.setCpuActiveMask( new_active_cpus );
                  break;
               }
            }
         }
#ifdef DLB
      }
#endif

      _lock.release();
   }
}

void ThreadManager::acquireDefaultCPUs( int max )
{
   if ( !_initialized ) return;
   if ( !_isMalleable ) return;
   if ( _cpuActiveMask.size() >= _maxThreads ) return;

   ThreadTeam *team = getMyThreadSafe()->getTeam();
   if ( !team ) return;

   // AcquireDefaultCPUs is called when one thread submits tasks for all threads
   // (e.g., worksharings), therefore we can't use a tryLock
   LockBlock lock( _lock );

#ifdef DLB
      if ( _useDLB ) {
         // Acquire CPUs lent while DLB was disabled
         if ( !_self_managed_cpus.empty() ) {
            CpuSet new_active_cpus = _cpuActiveMask;
            while ( !_self_managed_cpus.empty() && max > 0) {
               new_active_cpus.set( _self_managed_cpus.front() );
               _self_managed_cpus.pop_front();
               --max;
            }
            sys.setCpuActiveMask( new_active_cpus );
         }

         // Reclaim CPUs from this process
         DLB_ReclaimCpus( max );
      } else {
#endif
         // Otherwise, acquire CPUs from the process mask
         if ( !_cpuActiveMask.isSupersetOf( _cpuProcessMask ) ) {
            CpuSet new_active_cpus = _cpuActiveMask;
            for ( CpuSet::const_iterator it=_cpuProcessMask.begin();
                  it!=_cpuProcessMask.end() && max>0; ++it ) {
               new_active_cpus.set( *it );
               --max;
            }
            sys.setCpuActiveMask( new_active_cpus );
         }
#ifdef DLB
      }
#endif
}

int ThreadManager::borrowResources()
{
#ifdef DLB
   fatal_cond( _isMalleable, "borrowResources function should only be called"
                              " before opening OpenMP parallels" );

   if ( !_initialized ) return -1;
   if ( !_useDLB ) return -1;
   if ( !myThread->isMainThread() ) return -1;

   LockBlock lock( _lock );

   // Acquire all CPUs lent while DLB was disabled
   if ( !_self_managed_cpus.empty() ) {
      CpuSet new_active_cpus = _cpuActiveMask;
      while ( !_self_managed_cpus.empty() ) {
         new_active_cpus.set( _self_managed_cpus.front() );
         _self_managed_cpus.pop_front();
      }
      sys.setCpuActiveMask( new_active_cpus );
   }

   DLB_PollDROM_Update();
   DLB_Borrow();

   return _cpuActiveMask.size();
#else
   return -1;
#endif
}

void ThreadManager::returnMyCpuIfClaimed()
{
#ifdef DLB
   if ( !_initialized ) return;
   if ( !_useDLB ) return;

   BaseThread *thread = getMyThreadSafe();
   int my_cpu = thread->getCpuId();

   if ( _cpuProcessMask.isSet(my_cpu) ) return;

   if ( !thread->isSleeping() ) {
      if ( DLB_CheckCpuAvailability(my_cpu) == DLB_ERR_PERM ) {
         blockThread( thread );
      }
   }
#endif
}

void ThreadManager::waitForCpuAvailability()
{
#ifdef DLB
   if ( !_initialized ) return;
   if ( !_useDLB ) return;

   BaseThread *thread = getMyThreadSafe();
   int my_cpu = thread->getCpuId();

   int dlb_err = DLB_NOTED;
   while ( !lastActiveThread()
         && thread->isRunning()
         && dlb_err == DLB_NOTED ) {

      /* Query DLB for CPU availability */
      dlb_err = DLB_CheckCpuAvailability(my_cpu);

      if ( dlb_err == DLB_ERR_PERM ) {
         /* CPU has been reclaimed or disabled */
         blockThread( thread );
      } else if ( dlb_err == DLB_NOTED ) {
         /* CPU is not yet available */
         OS::nanosleep( ThreadManagerConf::DEFAULT_SLEEP_NS );
         sched_yield();
      } else if ( dlb_err == DLB_NOUPDT ) {
         /* CPU is not reclaimed, ask again */
         DLB_AcquireCpu( my_cpu );
      }
   }
#endif
}

void ThreadManager::poll()
{
#ifdef DLB
   if ( !_initialized ) return;
   if ( !_useDLB ) return;
   DLB_PollDROM_Update();
#endif
}

/**********************************/
/****** Thread Manager Conf *******/
/**********************************/

const unsigned int ThreadManagerConf::DEFAULT_SLEEP_NS = 20000;
const unsigned int ThreadManagerConf::DEFAULT_YIELDS = 10;

ThreadManagerConf::ThreadManagerConf() :
   _numYields( DEFAULT_YIELDS ),
   _sleepTime( DEFAULT_SLEEP_NS ),
   _useSleep( false ),
   _useBlock( false ),
   _useDLB( false ),
   _forceTieMaster( false ),
   _warmupThreads( false )
{
}

void ThreadManagerConf::config( Config &cfg )
{
   cfg.setOptionsSection("Thread Manager specific","Thread Manager related options");

   cfg.registerConfigOption( "enable-block", NEW Config::FlagOption( _useBlock, true ),
         "Thread block on idle and condition waits" );
   cfg.registerArgOption( "enable-block", "enable-block" );

   cfg.registerConfigOption( "enable-sleep", NEW Config::FlagOption( _useSleep, true ),
         "Thread sleep on idle and condition waits" );
   cfg.registerArgOption( "enable-sleep", "enable-sleep" );

   std::ostringstream sleep_sstream;
   sleep_sstream << "Set the amount of time (in nsec) in each sleeping phase (default = "
      << DEFAULT_SLEEP_NS << ")";
   cfg.registerConfigOption ( "sleep-time", NEW Config::UintVar( _sleepTime ), sleep_sstream.str() );
   cfg.registerArgOption ( "sleep-time", "sleep-time" );

   std::ostringstream yield_sstream;
   yield_sstream << "Set number of yields on idle before blocking (default = "
      << DEFAULT_YIELDS << ")";
   cfg.registerConfigOption ( "num-yields", NEW Config::UintVar( _numYields ), yield_sstream.str() );
   cfg.registerArgOption ( "num-yields", "yields" );

   cfg.registerConfigOption( "enable-dlb", NEW Config::FlagOption ( _useDLB ),
         "Tune Nanos Runtime to be used with Dynamic Load Balancing library" );
   cfg.registerArgOption( "enable-dlb", "enable-dlb" );

   cfg.registerConfigOption( "force-tie-master", NEW Config::FlagOption ( _forceTieMaster ),
         "Force Master WD (user code) to run on Master Thread" );
   cfg.registerArgOption( "force-tie-master", "force-tie-master" );

   cfg.registerConfigOption( "warmup-threads", NEW Config::FlagOption( _warmupThreads, true ),
         "Force the creation of as many threads as available CPUs at initialization time,"
         " then block them immediately if needed" );
   cfg.registerArgOption( "warmup-threads", "warmup-threads" );
}

ThreadManager* ThreadManagerConf::create()
{
#ifndef DLB
   if ( _useDLB ) {
      fatal0(
            "Thread Manager: DLB option was enabled but DLB library is not configured. "
            "Add DLB support at configure time." );
   }
#endif

   if ( _useSleep && _useBlock ) {
      warning0( "Option --enable-sleep is not compatible with --enable-block, disabling option." );
      _useSleep = false;
   }

   return NEW ThreadManager( _warmupThreads, _forceTieMaster, _numYields,
         _sleepTime, _useSleep, _useBlock, _useDLB );
}
