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

#ifndef THREADMANAGER_DECL_HPP
#define THREADMANAGER_DECL_HPP

#include "config.hpp"
#include "atomic_decl.hpp"
#include "cpuset.hpp"
#include "basethread_decl.hpp"

namespace nanos {

class ThreadManager
{
   private:
      Lock              _lock;
      bool              _initialized;
      unsigned int      _maxThreads;
      const CpuSet&     _cpuProcessMask;     /* Read-only masks from SMPPlugin */
      const CpuSet&     _cpuActiveMask;
      bool              _isMalleable;
      bool              _warmupThreads;
      unsigned int      _numYields;
      unsigned int      _sleepTime;
      bool              _useSleep;
      bool              _useBlock;
      bool              _useDLB;
      std::deque<int>   _self_managed_cpus;  /* List of CPUs lent while DLB is disabled */

   public:
      ThreadManager( bool warmup, bool tie_master, unsigned int num_yields,
            unsigned int sleep_time, bool use_sleep, bool use_block, bool use_dlb );

      ~ThreadManager();

      void init();
      bool isGreedy();
      bool lastActiveThread();
      void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
            , unsigned long long& total_yields, unsigned long long& total_blocks
            , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
            );
      void blockThread( BaseThread *thread );
      void unblockThread( BaseThread *thread );
      void lendCpu( BaseThread *thread );
      void acquireOne();
      void acquireDefaultCPUs( int max );
      int  borrowResources();
      void returnMyCpuIfClaimed();
      void waitForCpuAvailability();
      void poll();
      unsigned int getMaxThreads() { return _maxThreads; }
};

//! ThreadManagerConf class
/*!
   * This class is used to construct the right Thread Manager object.
   *
   * It firsts implements a Config method to configure the user options.
   * The create method work as a factory method according to the user options.
   * The other methods are needed by other classes when maybe the thread manager
   * is not yet created.
   */
class ThreadManagerConf
{
   private:
      unsigned int         _numYields;       //!< Number of yields before block
      unsigned int         _sleepTime;       //!< Number of nanoseconds to sleep
      bool                 _useSleep;        //!< Sleep is enabled
      bool                 _useBlock;        //!< Block is enabled
      bool                 _useDLB;          //!< DLB library will be used
      bool                 _forceTieMaster;  //!< Force Master WD (user code) to run on Master Thread
      bool                 _warmupThreads;   //!< Force the initialization of as many threads as number of CPUs, then block them if needed

   public:
      static const unsigned int DEFAULT_SLEEP_NS;
      static const unsigned int DEFAULT_YIELDS;

      ThreadManagerConf();
      unsigned int getNumYields ( void ) const { return _numYields; }
      void config( Config &cfg );
      ThreadManager* create();
      bool canUntieMaster() const { return !_forceTieMaster; }
};

} // namespace nanos

#endif /* THREADMANAGER_DECL_HPP */
