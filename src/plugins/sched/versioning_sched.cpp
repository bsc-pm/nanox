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

#include "schedule.hpp"
#include "atomic.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "os.hpp"
#include "config.hpp"
#include "hashmap.hpp"

#include <math.h>
#include <limits>


namespace nanos {
namespace ext
{

//#define MAX_STDDEV   0.1
//#define CHECK_STDDEV
//#define CHOOSE_ALWAYS_BEST
#define MAX_DEVIATION   0.01
#define MIN_RECORDS     3
#define MAX_DIFFERENCE  4

/*
 * Some terminology and data structures schemas to understand the code
 *  - WDs that implement the same functionality will have the same wdType
 *  - versionId is used to differentiate between WDs with the same wdType
 *
 *   WDBestRecord (hash)
 *  +-----------------------------------------------------+
 *  |  WDBestRecordKey                  WDBestRecordData  |
 *  | +------------------------+       +----------------+ |
 *  | | < WDType, paramsSize > | ----> | versionId      | |
 *  | +------------------------+       | PE             | |
 *  |                                  | elapsedTime    | |
 *  |                                  +----------------+ |
 *  +-----------------------------------------------------+
 *
 *   WDExecInfo (hash)
 *  +----------------------------------------------------------+
 *  |  WDExecInfoKey                    WDExecInfoData []      |
 *  | +------------------------+       +---------------------+ |
 *  | | < WDType, paramsSize > | ----> |  WDExecRecord       | |
 *  | +------------------------+       | +-----------------+ | |
 *  |                                  | | versionId       | | |
 *  |                                  | | PE              | | |
 *  |                                  | | elapsedTime     | | |
 *  |                                  | | lastElapsedTime | | |
 *  |                                  | | #records        | | |
 *  |                                  | | #assigned       | | |
 *  |                                  | +-----------------+ | |
 *  |                                  +---------------------+ |
 *  +----------------------------------------------------------+
 */


   struct WDBestRecordData {
      unsigned int            _versionId;
      ProcessingElement *     _pe;
      double                  _elapsedTime;
   };

   typedef std::pair< unsigned long, size_t > WDBestRecordKey;
   typedef PairHash< unsigned long, size_t > WDBestRecordHashKey;
   typedef HashMap< WDBestRecordKey, WDBestRecordData, false, 257, WDBestRecordHashKey > WDBestRecord;

   struct WDExecRecord {
      unsigned int            _versionId;
      ProcessingElement *     _pe;
      double                  _elapsedTime;
      double                  _lastElapsedTime;
      int                     _numRecords;
      Atomic<int>             _numAssigned;
   };

   typedef WDBestRecordKey WDExecInfoKey;
   typedef WDBestRecordHashKey WDExecInfoHashKey;
   typedef std::vector<WDExecRecord> WDExecInfoData;
   typedef HashMap< WDExecInfoKey, WDExecInfoData, false, 257, WDExecInfoHashKey > WDExecInfo;


   typedef enum {
      NANOS_SCHED_VER_NULL_EVENT,                        /* 0 */
      NANOS_SCHED_VER_SETDEVICE_CANRUN,
      NANOS_SCHED_VER_SETDEVICE_CANNOTRUN,
      NANOS_SCHED_VER_SELECTWD_FIRSTCANRUN,
      NANOS_SCHED_VER_SELECTWD_FIRSTCANNOTRUN,
      NANOS_SCHED_VER_SELECTWD_BELOWMINRECCANRUN,        /* 5 */
      NANOS_SCHED_VER_SELECTWD_UNDEFINED,
      NANOS_SCHED_VER_SELECTWD_GETFIRST,
      NANOS_SCHED_VER_ATIDLE_GETFIRST,
      NANOS_SCHED_VER_ATIDLE_NOFIRST,
      NANOS_SCHED_VER_ATPREFETCH_GETFIRST,               /* 10 */
      NANOS_SCHED_VER_ATPREFETCH_GETIMMSUCC,
      NANOS_SCHED_VER_ATPREFETCH_NOFIRST,
      NANOS_SCHED_VER_ATBEFEX_GETFIRST,
      NANOS_SCHED_VER_ATBEFEX_NOFIRST,
      NANOS_SCHED_VER_SETEARLIESTEW_FOUND,               /* 15 */
      NANOS_SCHED_VER_SETEARLIESTEW_NOTFOUND,
      NANOS_SCHED_VER_FINDEARLIESTEW_BETTERTIME,
      NANOS_SCHED_VER_FINDEARLIESTEW_IDLEWORKER
   } sched_versioning_event_value;

   // Macro's to instrument the code and make it cleaner
#define NANOS_SCHED_VER_RAISE_EVENT(x)   NANOS_INSTRUMENT( \
      sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "sched-versioning" ), (x) ); )

#define NANOS_SCHED_VER_CLOSE_EVENT       NANOS_INSTRUMENT( \
      sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "sched-versioning" ), 0 ); )

#define NANOS_SCHED_VER_POINT_EVENT(x) NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "sched-versioning" ), (x) ); \
		sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "sched-versioning" ), 0 ); )




   class WDTimingDeque : public WDPool
   {
      private:
         WDDeque           _queue;

      private:
         /*! \brief WDTimingDeque copy constructor
          */
         WDTimingDeque ( const WDTimingDeque &wdtq );

         /*! \brief WDTimingDeque copy assignment operator
          */
         const WDTimingDeque & operator= ( const WDTimingDeque &wdtq );

      public:
         /*! \brief WDTimingDeque default constructor
          */
         WDTimingDeque() : _queue() {}

         /*! \brief WDDeque destructor
          */
         ~WDTimingDeque() {}

         bool empty ( void ) const
         {
            return _queue.empty();
         }

         size_t size() const
         {
            return _queue.size();
         }

         void push_front ( WorkDescriptor *wd ) {}
         WorkDescriptor * pop_front ( BaseThread *thread ) { return NULL; }
         WorkDescriptor * pop_back ( BaseThread *thread ) { return NULL; }
         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next ) { return false; }

         void push_back( WD * task )
         {
            _queue.push_back( task );
         }

         WD * frontTask ( BaseThread * thread )
         {
            return _queue.pop_front( thread );
         }

         Lock& getLock(){
            return _queue.getLock();
         }

         void push_front( WD** wds, size_t numElems ) {}

         void push_back( WD** wds, size_t numElems )
         {
            _queue.push_back( wds, numElems );
         }
   };

   class WorkerExecPlan {
      public:
         double               _estimatedBusyTime;
         WDTimingDeque        _assignedTasksList;
         static Lock          _lock;

      private:
         /*! \brief WorkerExecPlan copy constructor
          */
         WorkerExecPlan ( const WorkerExecPlan &wep );

         /*! \brief WDTimingDeque copy assignment operator
          */
         const WorkerExecPlan & operator= ( const WorkerExecPlan &wep );

      public:

         WorkerExecPlan() : _estimatedBusyTime( 0 ), _assignedTasksList() {}

         ~WorkerExecPlan() {}

         void addTask ( double time, WD * task )
         {
            task->setEstimatedExecutionTime( time );

            _lock.acquire();
            _assignedTasksList.push_back( task );
            _estimatedBusyTime += time;
            _lock.release();
         }

         void finishTask ( WD * task )
         {
            _lock.acquire();
            _estimatedBusyTime = _assignedTasksList.empty() ? 0.0 : _estimatedBusyTime - task->getEstimatedExecutionTime();
            _estimatedBusyTime = _estimatedBusyTime < 0.0 ? 0.0 : _estimatedBusyTime;
           _lock.release();
         }

         WD * getFirstTask ( BaseThread * thread )
         {
            WD * wd = NULL;

            _lock.acquire();
            wd = _assignedTasksList.frontTask( thread );
            _lock.release();

            return wd;
         }
   };

   Lock WorkerExecPlan::_lock;
   typedef std::vector<WorkerExecPlan *> ResourceMap;


   class Versioning : public SchedulePolicy
   {
      public:
         using SchedulePolicy::queue;
      private:
         struct TeamData : public ScheduleTeamData
         {
            public:
               WDBestRecord               _wdExecBest;
               WDExecInfo                 _wdExecStats;
               std::set<WDExecInfoKey>    _wdExecStatsKeys;
               ResourceMap                _executionMap;

               static Lock                _bestLock;
               static Lock                _statsLock;

               WDDeque *                  _readyQueue;

               TeamData ( unsigned int size ) : ScheduleTeamData(), _wdExecBest(), _wdExecStats(), _wdExecStatsKeys(), _executionMap( size )
               {
                  unsigned int i;
                  for ( i = 0; i < size; i++ ) {
                     _executionMap[i] = NEW WorkerExecPlan();
                  }

                  _readyQueue = NEW WDDeque();
               }

               ~TeamData()
               {
                  unsigned int i;
                  for ( i = 0; i < _executionMap.size(); i++ ) {
                     delete _executionMap[i];
                  }

                  delete _readyQueue;
               }


               void initExecInfoData ( WDExecInfoData & data, WD * wd )
               {
                  unsigned int numVersions = wd->getNumDevices();

                  debug( "[versioning] First record for wd key ("
                        + toString<unsigned long>( wd->getVersionGroupId() )
                        + ", " + toString<size_t>( wd->getParamsSize() ) + ") with "
                        + toString<int>( numVersions ) + " versions" );

                  _statsLock.acquire();
                  // Reserve as much memory as we need for all the implementations
                  data.reserve( numVersions );
                  data = *NEW WDExecInfoData( numVersions );

                  bool compatible = false;
                  unsigned int i;
                  for ( i = 0; i < numVersions; i++ ) {
                     // Check there is at least one thread for each compatible device type
                     if ( sys.getNumWorkers( wd->getDevices()[i] ) == 0 ) {
                        // If not, 'disable' the implementation by making the scheduler never choose it
                        data[i]._pe = NULL;
                        data[i]._elapsedTime = std::numeric_limits<double>::max();
                        data[i]._lastElapsedTime = std::numeric_limits<double>::max();
                        data[i]._numRecords = _minRecordTrial;
                        data[i]._numAssigned = _minRecordTrial;
                     } else {
                        data[i]._pe = NULL;
                        data[i]._elapsedTime = 0.0;
                        data[i]._lastElapsedTime = 0.0;
                        data[i]._numRecords = -1;
                        data[i]._numAssigned = 0;
                        compatible = true;
                     }
                  }

                  _statsLock.release();

                  WDExecInfoKey key = std::make_pair( wd->getVersionGroupId(), wd->getParamsSize() );
                  _wdExecStatsKeys.insert( key );

                  fatal_cond( !compatible, "Error: there is no suitable device in the system to run the submitted task.");

               }


               inline WDExecInfoData& getWDExecInfo ( WD * wd )
               {
                  WDExecInfoKey key = std::make_pair( wd->getVersionGroupId(), wd->getParamsSize() );
                  WDExecInfoData &data = _wdExecStats[key];

                  return data;
               }


               inline WDBestRecordData& getWDBestRecord ( WD * wd )
               {
                  WDBestRecordKey key = std::make_pair( wd->getVersionGroupId(), wd->getParamsSize() );
                  WDBestRecordData &data = _wdExecBest[key];

                  return data;
               }


               void printStats()
               {
                  // Do not print stats if no data were recorded
                  if ( _wdExecStatsKeys.size() ) {
                     message( "VERSIONING SCHEDULER RECORDS" );
                     message( "BEST RECORDS" );

                     for ( std::set<WDExecInfoKey>::iterator it = _wdExecStatsKeys.begin(); it != _wdExecStatsKeys.end(); it++ ) {
                        WDBestRecordKey key = *it;
                        WDBestRecordData &data = _wdExecBest[key];

                        message( "    Best version found for groupId " << key.first << ", paramSize " << key.second << ":");
                        if ( data._pe != NULL ) {
                           message( "    versionId: " << data._versionId << ", PE: " << data._pe->getDeviceTypes()[0]->getName() << ", time: " << data._elapsedTime );
                        } else {
                           message( "    versionId: " << data._versionId << ", PE: null, time: " << data._elapsedTime );
                        }
                     }

                     message( "GENERAL STATISTICS" );

                     for ( std::set<WDExecInfoKey>::iterator it = _wdExecStatsKeys.begin(); it != _wdExecStatsKeys.end(); it++ ) {
                        WDExecInfoKey key = *it;
                        WDExecInfoData &data = _wdExecStats[key];

                        message( "    Statistics for groupId " << key.first << ", paramSize " << key.second << ":");
                        for ( unsigned int i = 0; i < data.size(); i++ ) {
                           WDExecRecord &record = data[i];
                           if ( record._pe == NULL ) {
                              if ( record._numAssigned.value() != _minRecordTrial ) {
                                 message( "    PE: " << "Device is NULL" << ", elapsed time: " << record._elapsedTime << " us, #records: " << record._numAssigned.value() );
                              } else {
                                 message( "    PE: " << "Device not present" << ", elapsed time: " << record._elapsedTime << " us, #records: " << record._numAssigned.value() );
                              }
                           } else {
                              message( "    PE: " << record._pe->getDeviceTypes()[0]->getName() << ", elapsed time: " << record._elapsedTime << " us, #records: " << record._numAssigned.value() );
                           }
                        }
                     }

                  }
               }
         };

      public:
         static bool       _useStack;
         static int        _minRecordTrial;

         Versioning() : SchedulePolicy( "Versioning" ) {}
         virtual ~Versioning () {}

      private:
         virtual size_t getTeamDataSize () const { return sizeof( TeamData ); }
         virtual size_t getThreadDataSize () const { return 0; }

         virtual ScheduleTeamData * createTeamData ()
         {
            TeamData *data;

            unsigned int num = sys.getNumWorkers();
            data = NEW TeamData( num );

            return data;
         }

         virtual ScheduleThreadData * createThreadData ()
         {
            return 0;
         }

         virtual void queue ( BaseThread *thread, WD &wd )
         {
            TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();
            tdata._readyQueue->push_back( &wd );
         }

         virtual WD *atSubmit ( BaseThread *thread, WD &newWD )
         {
            queue( thread, newWD );
            return 0;
         }

         virtual bool isCheckingWDExecTime ()
         {
            return true;
         }

         /*
          * Activate the device for the given WD.
          * If the current thread can run it, add it to its queue, otherwise, enqueue the task again (and it will
          * be picked by a thread compatible with WD's active device
          *
          */
         WD * setDevice ( BaseThread *thread, WD *wd, unsigned int deviceIdx, bool getTask = true, double time = 1 )
         {
            TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();

            if ( thread->runningOn()->supports( *( wd->getDevices()[deviceIdx]->getDevice() ) ) ) {

               NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_SETDEVICE_CANRUN );

               debug( "[versioning] Setting device #" + toString<unsigned int>( deviceIdx )
                     + " for WD " + toString<int>( wd->getId() ) + " (compatible with my PE)" );

               WD * next = NULL;
               setWorker( tdata, wd, deviceIdx, thread->getId(), time );

               if ( getTask ) {
                  next = tdata._executionMap[thread->getId()]->getFirstTask( thread );

#ifdef CHOOSE_ALWAYS_BEST
                  // TODO: Check this option: it seems like next should be enqueued again, and the pointer
                  // set to NULL because this task should not be returned to the current thread
                  unsigned int version = findBestVersion( thread, next );
                  if ( next->getActiveDeviceIdx() != version ) next->activateDevice( version );
#endif

                  debug( "[versioning] Getting front task of my queue: " + toString<int>( next ? next->getId() : -1 ) + " from setDevice()" );
               }

               NANOS_SCHED_VER_CLOSE_EVENT;

               return next;

            } else {

               double bestTime;
               // WD device activation needed before calling findEarliestExecutionWorkerForDevice()
               wd->activateDevice( deviceIdx );
               int worker = findEarliestExecutionWorkerForDevice( tdata, wd, bestTime, time, deviceIdx );
               setWorker( tdata, wd, deviceIdx, worker, bestTime );

               NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_SETDEVICE_CANNOTRUN );

               debug( "[versioning] Setting device #" + toString<unsigned int>( deviceIdx )
                     + " for WD " + toString<int>( wd->getId() ) + " (not compatible with my PE)" );

               WD * next = NULL;
               if ( getTask ) {
                  next = tdata._executionMap[thread->getId()]->getFirstTask( thread );

#ifdef CHOOSE_ALWAYS_BEST
                  if ( next != NULL ) {
                     unsigned int version = findBestVersion( thread, next );
                     if ( next->getActiveDeviceIdx() != version ) next->activateDevice( version );
                  }
#endif

                  debug( "[versioning] Getting front task of my queue: " + toString<int>( next ? next->getId() : -1 ) + " from setDevice()" );
               }

               NANOS_SCHED_VER_CLOSE_EVENT;

               return next;
            }
         }


         /*
          * Assign the given task to the given worker and update the stats
          *
          * In order to avoid repeated codes and keep consistency, this should be the only point in the code where tasks
          * are assigned to workers
          *
          */
         void setWorker ( TeamData &tdata, WD *wd, unsigned int deviceIdx, unsigned int workerIdx, double time = 1 )
         {
            WDExecInfoData &data = tdata.getWDExecInfo( wd );
            if ( data.empty() ) {
               tdata.initExecInfoData( data, wd );
            }
            data[deviceIdx]._numAssigned++;

            wd->activateDevice( deviceIdx );
            tdata._executionMap[workerIdx]->addTask( time, wd );

            debug( "[versioning] Setting worker #" + toString<unsigned int>( workerIdx ) + " for WD "
                  + toString<int>( wd->getId() ) + " and vId " + toString<unsigned int>( deviceIdx ) );
         }


         /*
          * TODO: Check this function to return the appropriate values
          *
          * Try to find the earliest execution worker for a WD, taking into account the estimated busy time
          * of each worker. Then, idle workers are also taken into account, even if they are not the fastest
          * executors of the given task
          *
          * It should return the information that atBeforeExit needs to schedule a bunch of tasks at a time, too
          *
          * If the WD has an active device, it will only consider compatible workers with that device
          *
          * Don't add tasks to thread queues here, this function should only be used for checking:
          * setEarliestExecutionWorker() should be used for planning next tasks
          *
          */
         int findEarliestExecutionWorker ( TeamData & tdata, WD *next, double &bestTime, double &time, unsigned int &devIdx  )
         {
            unsigned int w;
            int earliest = -1;
            double earliestTime = std::numeric_limits<double>::max();
            BaseThread * thread;

            // It should be fine to get WDExecInfoData reference once and just acquire the lock at each iteration
            WDExecInfoData &data = tdata.getWDExecInfo( next );

            for ( w = 0; w < tdata._executionMap.size(); w++ ) {
               thread = sys.getWorker( w );
               // Check the thread can run the task
               if ( thread->runningOn()->canRun( *next ) ) {
                  // Check if it would be the earliest time to run the task
                  unsigned int i;

                  tdata._statsLock.acquire();

                  tdata._executionMap[w]->_lock.acquire();
                  time = tdata._executionMap[w]->_estimatedBusyTime;

                  for ( i = 0; i < data.size(); i++ ) {
                     if ( data[i]._pe && thread->runningOn()->getDeviceTypes()[0] == data[i]._pe->getDeviceTypes()[0] ) {
                        if ( ( time + data[i]._elapsedTime ) < earliestTime ) {

                           NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_FINDEARLIESTEW_BETTERTIME );

                           earliestTime = time + data[i]._elapsedTime;
                           bestTime = data[i]._elapsedTime;
                           devIdx = i;
                           earliest = w;

                           NANOS_SCHED_VER_CLOSE_EVENT;

                        } else if ( tdata._executionMap[w]->_estimatedBusyTime == 0 && sys.getSchedulerStats().getTotalTasks() > 100 ) {
                           // compute a more accurate threshold?

                           NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_FINDEARLIESTEW_IDLEWORKER );

                           tdata._executionMap[w]->_lock.release();
                           tdata._statsLock.release();

                           earliest = w;
                           devIdx = findBestVersion ( thread, next, bestTime );
                           earliestTime = time + bestTime;

                           NANOS_SCHED_VER_CLOSE_EVENT;

                           return earliest;
                        }
                     }
                  }

                  tdata._executionMap[w]->_lock.release();
                  tdata._statsLock.release();
               }
            }

            //ensure( earliest != -1, "Could not find a suitable thread to run the task." );

            return earliest;
         }

         /*
          * Try to find the earliest execution worker for a WD that has already set an active device
          *
          * It only takes into account the estimated busy time of each compatible worker
          *
          * Don't add tasks to thread queues here, this function should only be used for checking:
          * setWorker() should be used for adding next tasks to workers
          *
          */
         int findEarliestExecutionWorkerForDevice ( TeamData & tdata, WD *next, double &bestTime, double &time, unsigned int &devIdx )
         {
            unsigned int w;
            int earliest = -1;
            double earliestTime = std::numeric_limits<double>::max();
            double busyTime;
            BaseThread * thread;

            for ( w = 0; w < tdata._executionMap.size(); w++ ) {
               thread = sys.getWorker( w );
               // Check the thread can run the task
               if ( thread->runningOn()->canRun( *next ) ) {
                  // Find the least busy thread to run the task

                  tdata._executionMap[w]->_lock.acquire();
                  busyTime = tdata._executionMap[w]->_estimatedBusyTime;

                  if ( busyTime < earliestTime ) {

                     NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_FINDEARLIESTEW_BETTERTIME );

                     earliestTime = busyTime;
                     earliest = w;

                     NANOS_SCHED_VER_CLOSE_EVENT;

                  }

                  tdata._executionMap[w]->_lock.release();
               }
            }

            // Lock not needed
            //tdata._statsLock.acquire();

            WDExecInfoData &data = tdata.getWDExecInfo( next );

            if ( data[devIdx]._numRecords > 0 ) {
               bestTime = data[devIdx]._elapsedTime;
            } else {
               bestTime = time;
            }

            //tdata._statsLock.release();

            ensure( earliest != -1, "Could not find a suitable thread to run the task." );

            return earliest;
         }


         /*
          * Try to find the earliest execution worker for the given WD.
          * If found, set it. Otherwise, enqueue the WD again
          *
          */
         int setEarliestExecutionWorker ( TeamData & tdata, WD *next )
         {
            double bestTime = 0, totalTime = 0;
            unsigned int devIdx = next->getNumDevices();
            int earliest = findEarliestExecutionWorker ( tdata, next, bestTime, totalTime, devIdx );
            if ( earliest != -1 ) {

               NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_SETEARLIESTEW_FOUND );

               setWorker( tdata, next, devIdx, earliest, bestTime );

               NANOS_SCHED_VER_CLOSE_EVENT;

            } else {
               // Could not find a good worker to run the task: queue it again to the ready queue

               NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_SETEARLIESTEW_NOTFOUND );

               std::cout << "WARNING: uncontrolled queuing path!!" << std::endl;

               queue( myThread, *next );

               NANOS_SCHED_VER_CLOSE_EVENT;

            }

            return earliest;
         }


         /*
          * Given a WD and a device (from PE), choose the best versionId to run this task
          * We assume that wd has at least 1 implementation that the given device can run
          *
          */
         unsigned int findBestVersion ( BaseThread * thread, WD * wd, double &bestTime )
         {
            TeamData & tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();
            WDExecInfoData &data = tdata.getWDExecInfo( wd );
            ProcessingElement *pe = thread->runningOn();
            unsigned int numVersions = wd->getNumDevices();
            DeviceData **devices = wd->getDevices();
            unsigned int bestIdx = numVersions;

            bestTime = std::numeric_limits<double>::max();

            unsigned int i;

            // It is not likely for 'data' to be empty, but it can happen, so we have to
            // make sure that it is initialized correctly
            if ( data.empty() ) {
               tdata.initExecInfoData( data, wd );
            }

            tdata._statsLock.acquire();

            for ( i = 0; i < numVersions; i++ ) {
               if ( pe->supports( *devices[i]->getDevice() ) && data[i]._numRecords > 0 && bestTime > data[i]._elapsedTime ) {
                  bestIdx = i;
                  bestTime = data[i]._elapsedTime;
               }
            }

            if ( bestIdx == numVersions ) {
               for ( i = 0; i < numVersions; i++ ) {
                  if ( pe->supports( *devices[i]->getDevice() ) && data[i]._numRecords < _minRecordTrial ) {
                     bestIdx = i;
                     bestTime = 1;
                     break;
                  }
               }
            }

            tdata._statsLock.release();

            ensure( bestIdx < numVersions, "Couldn't find the best version for this device" );

            return ( unsigned int ) bestIdx;
         }


         inline WD * selectWD ( BaseThread *thread, WD *next )
         {
            return selectWD( ( TeamData & ) *thread->getTeam()->getScheduleData(), thread, next );
         }


         /*
          * Choose the device where the task will be executed
          * Check we have recorded good (and reliable) enough results for each versionId and
          * if not, force running the appropriate versionIds to finally get statistics from
          * all WD's versionIds.
          *
          */
         WD * selectWD ( TeamData & tdata, BaseThread *thread, WD *next )
         {
            WDExecInfoData &data = tdata.getWDExecInfo( next );
            ProcessingElement *pe = thread->runningOn();
            unsigned int numVersions = next->getNumDevices();
            DeviceData **devices = next->getDevices();

            // First record for the given { wdId, paramsSize }
            if ( data.empty() ) {

               tdata.initExecInfoData( data, next );

               tdata._statsLock.acquire();

               if ( pe->canRun( *next ) ) {
                  // If the thread can run the task, activate its device and return the WD
                  unsigned int i;
                  for ( i = 0; i < numVersions; i++ ) {
                     if ( pe->supports( *devices[i]->getDevice() ) ) {
                        data[i]._numAssigned++;
                        tdata._statsLock.release();

                        NANOS_SCHED_VER_POINT_EVENT( NANOS_SCHED_VER_SELECTWD_FIRSTCANRUN );

                        return setDevice( thread, next, i );
                     }
                  }
               }

               tdata._statsLock.release();

               NANOS_SCHED_VER_POINT_EVENT( NANOS_SCHED_VER_SELECTWD_FIRSTCANNOTRUN );

               // Otherwise, return NULL
               return NULL;
            }

            // Reaching this point means we have already recorded some data for this wdType

            double timeLimit = std::numeric_limits<double>::max();
            //if ( tdata._wdExecBest.find( key ) )
            //   timeLimit = tdata._wdExecBest.find( key )->_elapsedTime * MAX_DIFFERENCE;

            unsigned int i;

            // First, check if the thread can run and, in fact, has to run the task
            if ( pe->canRun( *next ) ) {
               unsigned int bestCandidateIdx = numVersions;

               tdata._statsLock.acquire();

               for ( i = 0; i < numVersions; i++ ) {
                  WDExecRecord & record = data[i];

                  // Find a version that this PE can run
//                  if ( record._versionId->isCompatible( pe ) ) {
                     if ( record._lastElapsedTime < timeLimit ) {
                        // It is worth trying this device, so go on
                        if ( record._numAssigned < _minRecordTrial ) {
                           // Not enough records to have reliable values, so go on with this versionId

                           debug("[versioning] Less than 3 records for my device ("
                                 + toString<int>( record._numAssigned.value() ) + ") for key ("
                                 + toString<unsigned long>( next->getVersionGroupId() ) + ", "
                                 + toString<size_t>( next->getParamsSize() ) + ") vId "
                                 + toString<unsigned int>( i ) + " device "
                                 + next->getDevices()[i]->getDevice()->getName() );

                           // If this PE can run the task, run it
                           if ( pe->supports( *next->getDevices()[i]->getDevice() ) ) {

                              NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_SELECTWD_BELOWMINRECCANRUN );

                              memoryFence();
                              tdata._statsLock.release();

                              NANOS_SCHED_VER_CLOSE_EVENT;

                              return setDevice( thread, next, i );
                           }

                           bestCandidateIdx = ( bestCandidateIdx == numVersions ) ? i : bestCandidateIdx;
                        }

#ifdef CHECK_STDDEV
                        double sqDev = record._elapsedTime - record._lastElapsedTime;
                        sqDev *= sqDev;
                        if ( sqrt( sqDev ) > MAX_DEVIATION ) {
                           // Time values differ too much from each other, try to run the task again

                           debug("[versioning] Too much difference in records for my device ("
                                 + toString<double>( sqrt( sqDev ) ) + " > "
                                 + toString<double>( MAX_DEVIATION ) + ") for key ("
                                 + toString<unsigned long>( key.first ) + ", "
                                 + toString<size_t>( key.second ) + ") vId "
                                 + toString<unsigned int>( i ) + " device "
                                 + next->getDevices()[i]->getDevice()->getName() );

                           // If this PE can run the task, run it
                           if ( next->getDevices()[i]->isCompatible( pe ) ) {
                              record._numAssigned++;
                              memoryFence();
                              tdata._statsLock.release();
                              return setDevice( thread, next, i );
                           }

                           bestCandidateIdx = ( bestCandidateIdx == numVersions ) ? i : bestCandidateIdx;
                           //tdata._statsLock.release();
                           //return setDevice( thread, next, record._versionId );
                        }
#endif

                     }
//                  }
               }

               if ( bestCandidateIdx != numVersions ) {

                  NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_SELECTWD_UNDEFINED );

                  double sqDev = data[bestCandidateIdx]._elapsedTime - data[bestCandidateIdx]._lastElapsedTime;
                  sqDev *= sqDev;

                  debug("[versioning] Discarding my PE, but assigning to another device ("
                        + toString<double>( sqrt( sqDev ) ) + " > "
                        + toString<double>( MAX_DEVIATION ) + ") for key ("
                        + toString<unsigned long>( next->getVersionGroupId() ) + ", "
                        + toString<size_t>( next->getParamsSize() ) + ") vId "
                        + toString<unsigned int>( bestCandidateIdx ) + " device "
                        + next->getDevices()[bestCandidateIdx]->getDevice()->getName() );

                  memoryFence();
                  tdata._statsLock.release();

                  NANOS_SCHED_VER_CLOSE_EVENT;

                  return setDevice( thread, next, bestCandidateIdx );
               }

               tdata._statsLock.release();
            }

            // Reaching this point means that we have enough records to decide
            // It may happen that not all versionIds have been run
            // Choose the best versionId we have found by now
            // There is no need to care about 'next', since setEarliestExecutionWorker()
            // will deal with it
            setEarliestExecutionWorker( tdata, next );

            WD * wd = NULL;

            NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_SELECTWD_GETFIRST );

            wd = tdata._executionMap[thread->getId()]->getFirstTask( thread );

            NANOS_SCHED_VER_CLOSE_EVENT;

#ifdef CHOOSE_ALWAYS_BEST
            if ( wd != NULL ) {
               unsigned int version = findBestVersion( thread, wd );
               if ( wd->getActiveDeviceIdx() != version ) wd->activateDevice( version );
            }
#endif

            return wd;
         }


         WD * atIdle ( BaseThread *thread, int numSteal )
         {
            TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();

            WD * next = NULL;

            next = ( _useStack ? tdata._readyQueue->pop_back( thread ) : tdata._readyQueue->pop_front( thread ) );

            if ( next ) {
               return ( !( next->hasActiveDevice() ) ) ? selectWD( tdata, thread, next ) : next ;
            } else {
               next = tdata._executionMap[thread->getId()]->getFirstTask( thread );

               if ( next ) {

                  NANOS_SCHED_VER_POINT_EVENT( NANOS_SCHED_VER_ATIDLE_GETFIRST );

#ifdef CHOOSE_ALWAYS_BEST
                  unsigned int version = findBestVersion( thread, next );
                  if ( next->getActiveDeviceIdx() != version ) next->activateDevice( version );
#endif

                  return next;
               }
            }

            NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_ATIDLE_NOFIRST );

            OS::nanosleep( 100 );

            NANOS_SCHED_VER_CLOSE_EVENT;

            return NULL;
         }


         WD * atPrefetch ( BaseThread *thread, WD &current )
         {
            WD * found = NULL;

            TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();
            found = tdata._executionMap[thread->getId()]->getFirstTask( thread );

            if ( found ) {

               NANOS_SCHED_VER_POINT_EVENT( NANOS_SCHED_VER_ATPREFETCH_GETFIRST );

#ifdef CHOOSE_ALWAYS_BEST
               unsigned int version = findBestVersion( thread, found );
               if ( found->getActiveDeviceIdx() != version ) found->activateDevice( version );
#endif

               return found;
            }

            // WARNING: We're breaking a dependency here by getting the immediate successor and we've got to
            // guarantee that the immediate successor won't be run before the current task has finished. Then,
            // the safest way is to force this thread to run the immediate successor

            found = current.getImmediateSuccessor( *thread );

            if ( found ) {

               NANOS_SCHED_VER_POINT_EVENT( NANOS_SCHED_VER_ATPREFETCH_GETIMMSUCC );

               if ( thread == found->isTiedTo() ) return found;

               if ( !( found->hasActiveDevice() ) ) {
                  //return selectWD( thread, found );
                  double time = 1;
                  unsigned int deviceIdx = findBestVersion( thread, found, time );

                  WD * next = setDevice( thread, found, deviceIdx, true, time );
                  WD * last = found;

                  // WARNING: Prefetching for slower devices may impact on application's performance!
                  // It is not checked here, but, by now, only GPU is calling scheduler's prefetching mechanism
                  int i, numPrefetch = 16;
                  for ( i = 0; i < numPrefetch; i++ && last != NULL ) {
                     // getImmediateSuccessor() will only return tasks that either have no active device
                     // or its active device is compatible with this thread
                     WD * pref = last->getImmediateSuccessor( *thread );

                     if ( pref != NULL ) {
                        time = 1;
                        // Since we can get tasks with just one implementation, we have to check if
                        // the task has already an active device or not
                        if ( !pref->hasActiveDevice() ) {
                           deviceIdx = findBestVersion( thread, pref, time );
                        } else {
                           deviceIdx = pref->getActiveDeviceIdx();
                        }
                        setDevice( thread, pref, deviceIdx, false, time );
                        last = pref;
                     } else {
                        break;
                     }
                  }

                  return next;
               }

               return found;
            }

            NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_ATPREFETCH_NOFIRST );

            OS::nanosleep( 100 );

            NANOS_SCHED_VER_CLOSE_EVENT;

            return atIdle( thread, false );
         }

         WD * atBeforeExit ( BaseThread *thread, WD &currentWD, bool schedule )
         {
            TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();

            if ( currentWD.getNumDevices() > 1 ) {
               ProcessingElement * pe = thread->runningOn();
               double executionTime = currentWD.getExecutionTime();
               unsigned int devIdx = currentWD.getActiveDeviceIdx();

               tdata._executionMap[thread->getId()]->finishTask( &currentWD );

               tdata._statsLock.acquire();
               WDExecInfoData & data = tdata.getWDExecInfo( &currentWD );

               // Record statistic values
               // Update stats
               // TODO: Choose the appropriate device

               if ( data[devIdx]._numRecords == -1 ) {
                  // As it is the first time for the given PE, we omit the results because
                  // they can be potentially worse than future executions
                  WDExecRecord & records = data[devIdx];
                  records._pe = pe;
                  records._elapsedTime = 0.0; // Should be 'executionTime'
                  records._numRecords++; // Should be '1' but in fact it is -1+1 = 0
                  records._lastElapsedTime = executionTime;

                  debug("[versioning] First recording for key ("
                        + toString<unsigned long>( currentWD.getVersionGroupId() )
                        + ", " + toString<size_t>( currentWD.getParamsSize() )
                        + ") {pe=" + toString<void *>( records._pe )
                        + ", dev=" + currentWD.getDevices()[devIdx]->getDevice()->getName()
                        + ", #=" + toString<int>( records._numRecords )
                        + ", T=" + toString<double>( records._elapsedTime )
                        + ", T2=" + toString<double>( records._lastElapsedTime )
                        + "}; exec time = " + toString<double>( executionTime ) );

               } else {
                  WDExecRecord & records  = data[devIdx];
                  double time = records._elapsedTime * records._numRecords;
                  records._numRecords++;
                  records._elapsedTime = ( time + executionTime ) / records._numRecords;
                  records._lastElapsedTime = executionTime;

                  debug("[versioning] Recording for key ("
                        + toString<unsigned long>( currentWD.getVersionGroupId() )
                        + ", " + toString<size_t>( currentWD.getParamsSize() )
                        + ") {pe=" + toString<void *>( records._pe )
                        + ", dev=" + currentWD.getDevices()[devIdx]->getDevice()->getName()
                        + ", #=" + toString<int>( records._numRecords )
                        + ", T=" + toString<double>( records._elapsedTime )
                        + ", T2=" + toString<double>( records._lastElapsedTime )
                        + "}; exec time = " + toString<double>( executionTime ) );

               }

               memoryFence();

               tdata._statsLock.release();
               tdata._bestLock.acquire();

               // Check if it is the best time
               WDBestRecordData &bestData = tdata.getWDBestRecord( &currentWD );
               bool isBestTime = ( bestData._elapsedTime > executionTime ) || ( bestData._pe == NULL );
               if ( isBestTime ) {
                  // New best value recorded
                  bestData._versionId = devIdx;
                  bestData._pe = pe;
                  bestData._elapsedTime = executionTime;

                  debug("[versioning] New best time: {pe=" + toString<void *>( bestData._pe )
                        + ", T=" + toString<double>( bestData._elapsedTime ) + "}" );
               }

               tdata._bestLock.release();
            }

            if ( schedule ) {
               // Get next WD to run
               WD * found = tdata._executionMap[thread->getId()]->getFirstTask( thread );

               if ( found ) {

                  NANOS_SCHED_VER_POINT_EVENT( NANOS_SCHED_VER_ATBEFEX_GETFIRST );

#ifdef CHOOSE_ALWAYS_BEST
                  unsigned int version = findBestVersion( thread, found );
                  if ( found->getActiveDeviceIdx() != version ) found->activateDevice( version );
#endif

                  return found;
               }
            }

            NANOS_SCHED_VER_RAISE_EVENT( NANOS_SCHED_VER_ATBEFEX_NOFIRST );

            OS::nanosleep( 100 );

            NANOS_SCHED_VER_CLOSE_EVENT;

            return NULL;
         }


         /*
          * This function should only be called for debugging purposes
          *
          */
         void printExecutionPlan ()
         {
#ifdef NANOS_DEBUG_ENABLED
            std::string s;

            // Execution plan vector (each position represents a worker thread task queue)
            ResourceMap * map = &( ( TeamData * ) myThread->getTeam()->getScheduleData() )->_executionMap;
            unsigned int i, size = map->size();

            BaseThread * worker;
            WorkerExecPlan * wq;
            WDTimingDeque * wdlist;

            unsigned int j, qsize;

            s =   "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            s +=  "+                                 EXECUTION PLAN                                 +\n";
            s +=  "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";



            for ( i = 0; i < size; i++ ) {
               // Current worker thread
               worker = sys.getWorker( i );
               // Current worker thread's execution plan
               wq = ( *map )[i];

               wq->_lock.acquire();

               // Current worker thread's task queue (inside its execution plan)
               wdlist = &wq->_assignedTasksList;
               qsize = wq->_assignedTasksList.size();

               s +=  "\n+ ";
               s +=  toString<unsigned int>( i );
               s +=  " + ";

               // Since we cannot access the queue objects directly, we will pop front and
               // push back each and every object
               for ( j = 0; j < qsize; j++ ) {
                  WD * current = wdlist->frontTask( worker );
                  s += toString<int>( current->getId() );
                  wdlist->push_back( current );

                  if ( j + 1 < qsize ) {
                     s += " | ";
                  } else {
                     s += " +";
                  }
               }
               wq->_lock.release();

               if ( i + 1 < size ) {
                  s +=  "\n+--------------------------------------------------------------------------------+";
               }

            }

            s +=  "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

            std::cout << s << std::endl;
#endif
         }
   };

   bool Versioning::_useStack = false;
   int Versioning::_minRecordTrial = MIN_RECORDS;
   Lock Versioning::TeamData::_bestLock;
   Lock Versioning::TeamData::_statsLock;

   class VersioningSchedPlugin : public Plugin
   {

      public:
         VersioningSchedPlugin() : Plugin( "Versioning scheduling Plugin", 1 ) {}

         ~VersioningSchedPlugin() {}

         virtual void config ( Config &cfg )
         {
            // Set whether to use stack to pop tasks from ready queue or not
            cfg.setOptionsSection( "Versioning module", "Versioning scheduling module" );
            cfg.registerConfigOption ( "versioning-use-stack",
                  NEW Config::FlagOption( Versioning::_useStack ),
                  "Stack usage for the versioning policy" );
            cfg.registerArgOption( "versioning-use-stack", "versioning-use-stack" );

            cfg.registerAlias ( "versioning-use-stack", "versioning-stack",
                  "Stack usage for the versioning policy" );
            cfg.registerArgOption ( "versioning-stack", "versioning-stack" );

            // Set the number of minimum trials for each task version before considering
            // statistics reliable
            cfg.registerConfigOption ( "versioning-min-trials",
                  NEW Config::IntegerVar( Versioning::_minRecordTrial ),
                  "Minimum number of task version trials for the versioning policy" );
            cfg.registerArgOption( "versioning-min-trials", "versioning-min-trials" );
         }

         virtual void init()
         {
            sys.setDefaultSchedulePolicy( NEW Versioning() );
         }
   };

}
}


DECLARE_PLUGIN( "placeholder-name", nanos::ext::VersioningSchedPlugin );
