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

#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

#include <math.h>
#include <limits>


namespace nanos {
namespace ext
{

//#define MAX_STDDEV   0.1
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
   typedef HashMap< WDBestRecordKey, WDBestRecordData > WDBestRecord;

   struct WDExecRecord {
      unsigned int            _versionId;
      ProcessingElement *     _pe;
      double                  _elapsedTime;
      double                  _lastElapsedTime;
      int                     _numRecords;
      int                     _numAssigned;
   };

   typedef std::pair< unsigned long, size_t > WDExecInfoKey;
   typedef std::vector<WDExecRecord> WDExecInfoData;
   typedef HashMap< WDExecInfoKey, WDExecInfoData > WDExecInfo;



   class Versioning : public SchedulePolicy
   {
      private:
         struct TeamData : public ScheduleTeamData
         {
            public:
               WDBestRecord      _wdExecBest;
               WDExecInfo        _wdExecStats;

               static Lock       _bestLock;
               static Lock       _statsLock;

               WDDeque           _readyQueue;

               TeamData () : ScheduleTeamData(), _wdExecBest(), _wdExecStats(), _readyQueue() {}
               ~TeamData ()
               {
                  message( "VERSIONING SCHEDULER RECORDS" );
                  message( "BEST RECORDS" );
                  for ( WDBestRecord::iterator it = _wdExecBest.begin(); it != _wdExecBest.end(); it++ ) {

                     message( "    Best version found for groupId " << it.getKey().first << ", paramSize " << it.getKey().second << ":");
                     message( "    versionId: " << it->_versionId << ", PE: " << it->_pe->getDeviceType().getName() << ", time: " << it->_elapsedTime );
                  }

                  message( "GENERAL STATISTICS" );
                  for ( WDExecInfo::iterator it = _wdExecStats.begin(); it != _wdExecStats.end(); it++ ) {
                     message( "    Statistics for groupId " << it.getKey().first << ", paramSize " << it.getKey().second << ":");
                     for ( WDExecInfoData::iterator it2 = it->begin(); it2 != it->end(); it2++ ) {
                        message( "    PE: " << it2->_pe->getDeviceType().getName() << ", elapsed time: " << it2->_elapsedTime << ", #records: " << it2->_numRecords );
                     }
                  }
               }

         };

      public:
         static bool       _useStack;

         Versioning() : SchedulePolicy( "Versioning" ) {}
         virtual ~Versioning ()
         {
            if ( myThread->getTeam()->getScheduleData() != NULL ) {
               myThread->getTeam()->getScheduleData()->~ScheduleTeamData();
            }
         }

      private:

         virtual size_t getTeamDataSize () const { return sizeof( TeamData ); }
         virtual size_t getThreadDataSize () const { return 0; }

         virtual ScheduleTeamData * createTeamData ()
         {
            return NEW TeamData();
         }

         virtual ScheduleThreadData * createThreadData ()
         {
            return 0;
         }

         virtual void queue ( BaseThread *thread, WD &wd )
         {
            TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();
            tdata._readyQueue.push_back( &wd );
         }

         virtual WD *atSubmit ( BaseThread *thread, WD &newWD )
         {
            queue( thread, newWD );
            return 0;
         }

         WD * setDevice ( BaseThread *thread, WD *wd, unsigned int deviceIdx )
         {

            wd->activateDevice( deviceIdx );
            if ( wd->getDevices()[deviceIdx]->isCompatible( thread->runningOn()->getDeviceType() ) ) {
               debug( "[versioning] Setting device #" + toString<unsigned int>( deviceIdx )
                     + " for WD " + toString<int>( wd->getId() ) + " (compatible with my PE)" );
               return wd;
            } else {
               queue( thread, *wd );
               debug( "[versioning] Setting device #" + toString<unsigned int>( deviceIdx )
                     + " for WD " + toString<int>( wd->getId() ) + " (not compatible with my PE)" );
               return NULL;
            }
         }

         inline WD * selectWD ( BaseThread *thread, WD *next )
         {
            return selectWD( ( TeamData & ) *thread->getTeam()->getScheduleData(), thread, next );
         }

         WD * selectWD ( TeamData & tdata, BaseThread *thread, WD *next )
         {
            // Choose the device where the task will be executed
            // Check we have recorded good (and reliable) enough results for each versionId
            unsigned long wdId =  next->getVersionGroupId();
            size_t paramsSize = next->getParamsSize();
            WDExecInfoKey key = std::make_pair( wdId, paramsSize );
            WDExecInfoData &data = tdata._wdExecStats[key];
            ProcessingElement *pe = thread->runningOn();
            unsigned int numVersions = next->getNumDevices();
            DeviceData **devices = next->getDevices();

            // First record for the given { wdId, paramsSize }
            if ( data.empty() ) {

               debug( "[versioning] First record for wd key (" + toString<unsigned long>( key.first )
                     + ", " + toString<size_t>( key.second ) + ") with "
                     + toString<int>( numVersions ) + " versions" );

               tdata._statsLock.acquire();
               // Reserve as much memory as we need for all the implementations
               data.reserve( numVersions );
               data = *NEW WDExecInfoData( numVersions );

               unsigned int i;
               for ( i = 0; i < numVersions; i++ ) {
                  data[i]._pe = NULL;
                  data[i]._elapsedTime = 0.0;
                  data[i]._lastElapsedTime = 0.0;
                  data[i]._numRecords = -1;
                  data[i]._numAssigned = 0;
               }
               tdata._statsLock.release();

               if ( next->canRunIn( *pe ) ) {
                  // If the thread can run the task, activate its device and return the WD
                  for ( i = 0; i < numVersions; i++ ) {
                     if ( devices[i]->isCompatible( pe->getDeviceType() ) ) {
                        return setDevice( thread, next, i );
                     }
                  }
               } //else {
                  // Else, activate the first device
                  //i = 0;
               //}

               // Otherwise, return NULL
               return NULL;
               //return setDevice( thread, next, next->getDevices()[i] );
            }

            // Reaching this point means we have already recorded some data for this wdType
            double timeLimit = std::numeric_limits<double>::max();
            if ( tdata._wdExecBest.find( key ) )
               timeLimit = tdata._wdExecBest.find( key )->_elapsedTime * MAX_DIFFERENCE;

            tdata._statsLock.acquire();
            unsigned int i;

            // First, check if the thread can run and, in fact, has to run the task
            if ( next->canRunIn( *pe ) ) {
               unsigned int bestCandidateIdx = numVersions;

               for ( i = 0; i < numVersions; i++ ) {
                  WDExecRecord & record = data[i];

                  // Find a version that this PE can run
//                  if ( record._versionId->isCompatible( pe->getDeviceType() ) ) {
                     if ( record._lastElapsedTime < timeLimit ) {
                        // It is worth trying this device, so go on
                        if ( record._numAssigned < MIN_RECORDS ) {
                           // Not enough records to have reliable values, so go on with this versionId

                           debug("[versioning] Less than 3 records for my device ("
                                 + toString<int>( record._numAssigned ) + ") for key ("
                                 + toString<unsigned long>( key.first ) + ", "
                                 + toString<size_t>( key.second ) + ") vId "
                                 + toString<unsigned int>( i ) + " device "
                                 + next->getDevices()[i]->getDevice()->getName() );

                           // If this PE can run the task, run it
                           if ( next->getDevices()[i]->isCompatible( pe->getDeviceType() ) ) {
                              record._numAssigned++;
                              memoryFence();
                              tdata._statsLock.release();
                              return setDevice( thread, next, i );
                           }

                           bestCandidateIdx = ( bestCandidateIdx == numVersions ) ? i : bestCandidateIdx;
                           //tdata._statsLock.release();
                           //return setDevice( thread, next, record._versionId );
                        }

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
                           if ( next->getDevices()[i]->isCompatible( pe->getDeviceType() ) ) {
                              record._numAssigned++;
                              memoryFence();
                              tdata._statsLock.release();
                              return setDevice( thread, next, i );
                           }

                           bestCandidateIdx = ( bestCandidateIdx == numVersions ) ? i : bestCandidateIdx;
                           //tdata._statsLock.release();
                           //return setDevice( thread, next, record._versionId );
                        }

                     }
//                  }
               }
               if ( bestCandidateIdx != numVersions ) {

                  double sqDev = data[bestCandidateIdx]._elapsedTime - data[bestCandidateIdx]._lastElapsedTime;
                  sqDev *= sqDev;

                  debug("[versioning] Discarding my PE, but assigning to another device ("
                        + toString<double>( sqrt( sqDev ) ) + " > "
                        + toString<double>( MAX_DEVIATION ) + ") for key ("
                        + toString<unsigned long>( key.first ) + ", "
                        + toString<size_t>( key.second ) + ") vId "
                        + toString<unsigned int>( bestCandidateIdx ) + " device "
                        + next->getDevices()[data[bestCandidateIdx]._versionId]->getDevice()->getName() );

                  data[bestCandidateIdx]._numAssigned++;
                  memoryFence();
                  tdata._statsLock.release();
                  return setDevice( thread, next, bestCandidateIdx );
               }
            }

            // Reaching this point means either one of these 2 situations:
            // - This PE cannot run the task
            // - Each versionId compatible with this PE has been run enough times
            //   and results are reliable
#if 0
            for ( i = 0; i < numVersions; i++ ) {
               WDExecRecord & record = data[i];
               if ( record._lastElapsedTime < timeLimit ) {
                  // It is worth trying this device, so go on

                  if ( record._numAssigned < MIN_RECORDS ) {
                     // Not enough records to have reliable values

                     debug("[versioning] Less than 3 records ("
                           + toString<int>( record._numRecords ) + ") for key ("
                           + toString<unsigned long>( key.first ) + ", "
                           + toString<size_t>( key.second ) + ") vId "
                           + toString<void *>( record._versionId ) + " device "
                           + record._versionId->getDevice()->getName() );

                     record._numAssigned++;

                     tdata._statsLock.release();
                     return setDevice( thread, next, record._versionId );
                  }

                  double sqDev = record._elapsedTime - record._lastElapsedTime;
                  sqDev *= sqDev;
                  if ( sqrt( sqDev ) > MAX_DEVIATION ) {
                     // Values differ too much from each other, compute again

                     debug("[versioning] Too much difference in records ("
                           + toString<double>( sqrt( sqDev ) ) + " > "
                           + toString<double>( MAX_DEVIATION ) + ") for key ("
                           + toString<unsigned long>( key.first ) + ", "
                           + toString<size_t>( key.second ) + ") vId "
                           + toString<void *>( record._versionId ) + " device "
                           + record._versionId->getDevice()->getName() );

                     record._numAssigned++;

                     tdata._statsLock.release();
                     return setDevice( thread, next, record._versionId );
                  }
               }
            }
#endif

            // Reaching this point means that we have enough records to decide
            // It may happen that not all versionIds have been run
            // Choose the best versionId we have found by now
            tdata._bestLock.acquire();
            WDBestRecordData * bestData = tdata._wdExecBest.find( key );

            if ( next->getDevices()[bestData->_versionId]->isCompatible( pe->getDeviceType() ) ) {
               data[bestData->_versionId]._numAssigned++;

               debug("[versioning] Autochoosing for key ("
                     + toString<unsigned long>( key.first ) + ", " + toString<size_t>( key.second )
                     + ") my device " + bestData->_pe->getDeviceType().getName()
                     + " for vId " + toString<unsigned int>( bestData->_versionId ) );

               memoryFence();
               tdata._bestLock.release();
               tdata._statsLock.release();
               return setDevice( thread, next, bestData->_versionId );
            }

            // My PE is not the best PE to run this task, but still may be able to help with the execution
            // Estimate the remaining amount of time to execute the assigned tasks
            //double remTime = 0.0;
            double remBest = 0.0;
            unsigned int idxBest = numVersions;
            unsigned int myIndex = numVersions;

            for ( i = 0; i < numVersions; i++ ) {
               WDExecRecord & record = data[i];
               if ( bestData->_versionId == i ) {
               //if ( record._versionId->isCompatible( bestData->_pe->getDeviceType() ) ) {
                  // Keep the index of the best versionId
                  idxBest = i;
                  remBest = ( record._numAssigned - record._numRecords ) * record._elapsedTime;
               } else if ( next->getDevices()[i]->isCompatible( pe->getDeviceType() ) ) {
                  // PE can run this versionId, but make sure this is the best versionId this PE can run
                  if ( myIndex == numVersions ) {
                     myIndex = i;
                  } else {
                     // Another versionId compatible with this PE has been found previously
                     // Check which versionId is better
                     myIndex = data[myIndex]._elapsedTime > record._elapsedTime ? i : myIndex ;
                  }
               } //else {
                  //remTime += ( record._numAssigned - record._numRecords ) * record._elapsedTime;
               //}
            }

            // For simplicity, assume that all PEs that can run the best version will run it in parallel
            // and estimate the total time this will take
            timeLimit = remBest / sys.getNumWorkers( next->getDevices()[bestData->_versionId] );
            timeLimit = timeLimit * 0.9;
            if ( timeLimit > data[myIndex]._elapsedTime ) {
               // This PE has enough time to execute one instance of its own version
               idxBest = myIndex;
            }

            data[idxBest]._numAssigned++;

            debug("[versioning] Autochoosing for key ("
                  + toString<unsigned long>( key.first ) + ", " + toString<size_t>( key.second )
                  + ") best device is " + bestData->_pe->getDeviceType().getName()
                  + " chosen device is " + next->getDevices()[idxBest]->getDevice()->getName()
                  + " vId " + toString<unsigned int>( idxBest ) + ": time limit is "
                  + toString<double>( timeLimit ) + " and my elapsed time is "
                  + toString<double>( data[idxBest]._elapsedTime ) );

            memoryFence();
            tdata._bestLock.release();
            tdata._statsLock.release();
            return setDevice( thread, next, idxBest );

         }

         WD * atIdle ( BaseThread *thread )
         {
            TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();

            WD * next = NULL;

            if ( _useStack ) {
               next = tdata._readyQueue.pop_back( thread );
            } else {
               next = tdata._readyQueue.pop_front( thread );
            }

            if ( next ) {
               return ( !( next->hasActiveDevice() ) ) ? selectWD( tdata, thread, next ) : next ;
            }

            struct timespec req, rem;
            req.tv_sec = 0;
            req.tv_nsec = 100;
            nanosleep( &req, &rem );

            return NULL;
         }

         WD * atPrefetch ( BaseThread *thread, WD &current )
         {
            WD * found = current.getImmediateSuccessor( *thread );

            if ( found ) {
               return ( !( found->hasActiveDevice() ) ) ? selectWD( thread, found ) : found ;
            }

            struct timespec req, rem;
            req.tv_sec = 0;
            req.tv_nsec = 100;
            nanosleep( &req, &rem );

            return NULL;
         }

         WD * atBeforeExit ( BaseThread *thread, WD &currentWD )
         {
            if ( currentWD.getNumDevices() > 1 ) {
               unsigned long wdId = currentWD.getVersionGroupId();
               size_t paramsSize = currentWD.getParamsSize();
               ProcessingElement * pe = thread->runningOn();
               double executionTime = currentWD.getExecutionTime();
               unsigned int devIdx = currentWD.getActiveDeviceIdx();

               WDExecInfoKey key = std::make_pair( wdId, paramsSize );

               TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();
               tdata._statsLock.acquire();
               WDExecInfoData & data = tdata._wdExecStats[key];

               // Record statistic values
               // Update stats
               // TODO: Choose the appropriate device

               if ( data[devIdx]._numRecords == -1 ) {
                  // Here 'i' points to the first free position to record the values for the given PE
                  // As it is the first time for the given PE, we omit the results because
                  // they can be potentially worse than future executions
                  WDExecRecord & records = data[devIdx];
                  records._pe = pe;
                  records._elapsedTime = 0.0; // Should be 'executionTime'
                  records._numRecords++; // Should be '1' but in fact it is -1+1 = 0
                  records._lastElapsedTime = executionTime;

                  debug("[versioning] First recording for key (" + toString<unsigned long>( key.first )
                        + ", " + toString<size_t>( key.second )
                        + ") {pe=" + toString<void *>( records._pe )
                        + ", dev=" + currentWD.getDevices()[devIdx]->getDevice()->getName()
                        + ", #=" + toString<int>( records._numRecords )
                        + ", T=" + toString<double>( records._elapsedTime )
                        + ", T2=" + toString<double>( records._lastElapsedTime )
                        + "}; exec time = " + toString<double>( executionTime ) );

               } else {
                  // Here 'i' points to the position associated to the given PE
                  WDExecRecord & records  = data[devIdx];
                  double time = records._elapsedTime * records._numRecords;
                  records._numRecords++;
                  records._elapsedTime = ( time + executionTime ) / records._numRecords;
                  records._lastElapsedTime = executionTime;

                  debug("[versioning] Recording for key (" + toString<unsigned long>( key.first )
                        + ", " + toString<size_t>( key.second )
                        + ") {pe=" + toString<void *>( records._pe )
                        + ", dev=" + currentWD.getDevices()[devIdx]->getDevice()->getName()
                        + ", #=" + toString<int>( records._numRecords )
                        + ", T=" + toString<double>( records._elapsedTime )
                        + ", T2=" + toString<double>( records._lastElapsedTime )
                        + "}; exec time = " + toString<double>( executionTime ) );

               }

               tdata._statsLock.release();
               tdata._bestLock.acquire();

               // Check if it is the best time
               WDBestRecordData &bestData = tdata._wdExecBest[key];
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


            WD * found = currentWD.getImmediateSuccessor( *thread );

            if ( found ) {
               return ( !( found->hasActiveDevice() ) ) ? selectWD( thread, found ) : found ;
            }

            struct timespec req, rem;
            req.tv_sec = 0;
            req.tv_nsec = 100;
            nanosleep( &req, &rem );

            return NULL;
         }
   };

   bool Versioning::_useStack = false;
   Lock Versioning::TeamData::_bestLock;
   Lock Versioning::TeamData::_statsLock;

   class VersioningSchedPlugin : public Plugin
   {

      public:
         VersioningSchedPlugin() : Plugin( "Versioning scheduling Plugin", 1 ) {}

         ~VersioningSchedPlugin()
         {
            if ( sys.getDefaultSchedulePolicy() != NULL ) {
               delete sys.getDefaultSchedulePolicy();
               sys.setDefaultSchedulePolicy( NULL );
            }
         }

         virtual void config ( Config &cfg )
         {
            cfg.setOptionsSection( "Versioning module", "Versioning scheduling module" );
            cfg.registerConfigOption ( "versioning-use-stack",
                  NEW Config::FlagOption( Versioning::_useStack ),
                  "Stack usage for the versioning policy" );
            cfg.registerArgOption( "versioning-use-stack", "versioning-use-stack" );

            cfg.registerAlias ( "versioning-use-stack", "versioning-stack",
                  "Stack usage for the versioning policy" );
            cfg.registerArgOption ( "versioning-stack", "versioning-stack" );
         }

         virtual void init()
         {
            sys.setDefaultSchedulePolicy( NEW Versioning() );
         }
   };

}
}


nanos::ext::VersioningSchedPlugin NanosXPlugin;

