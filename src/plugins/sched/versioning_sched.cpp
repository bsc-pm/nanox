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


namespace nanos {
namespace ext
{

//#define MAX_STDDEV   0.1
#define MAX_DEVIATION   0.01
#define MIN_RECORDS     3

   struct WDExecRecords {
      ProcessingElement *     _pe;
      const Device *          _device;
      double                  _elapsedTime;
      double                  _lastElapsedTime;
      int                     _numRecords;
   };

   // WDBestRecordKey { wdType, paramsSize }
   // WDBestRecordData { PE, elapsedTime }
   typedef std::pair< unsigned long, size_t > WDBestRecordKey;
   typedef std::pair< ProcessingElement *, double> WDBestRecordData;
   typedef HashMap< WDBestRecordKey, WDBestRecordData > WDBestRecord;
   // WDExecInfoKey { wdType, paramsSize }
   // WDExecInfoData { PE, elapsedTime, elapsedTime^2, numRecords }
   typedef std::pair< unsigned long, size_t > WDExecInfoKey;
   typedef std::vector<WDExecRecords> WDExecInfoData;
   typedef HashMap< WDExecInfoKey, WDExecInfoData > WDExecInfo;

   class Versioning : public SchedulePolicy
   {
      private:
         struct TeamData : public ScheduleTeamData
         {
            public:
               WDBestRecord      _wdExecBest;
               WDExecInfo        _wdExecStats;

               WDDeque           _readyQueue;

               TeamData () : ScheduleTeamData(), _wdExecBest(), _wdExecStats(), _readyQueue() {}
               ~TeamData () {}

         };

      public:
         static bool       _useStack;
         static Lock       _lock;

         Versioning() : SchedulePolicy( "Versioning" ) {}
         virtual ~Versioning () {}

      private:

         virtual size_t getTeamDataSize () const { return sizeof( TeamData ); }
         virtual size_t getThreadDataSize () const { return 0; }

         virtual ScheduleTeamData * createTeamData ( ScheduleTeamData *preAlloc )
         {
            TeamData *data;

            if ( preAlloc ) data = new ( preAlloc ) TeamData();
            else data = NEW TeamData();

            return data;
         }

         virtual ScheduleThreadData * createThreadData ( ScheduleThreadData *preAlloc )
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

         WD * setDevice ( BaseThread *thread, WD *wd, const Device * device )
         {
            wd->activateDevice( *device );
            if ( device == &( thread->runningOn()->getDeviceType() ) ) {
               return wd;
            } else {
               queue( thread, *wd );
               return NULL;
            }
         }

         WD * setDevice ( BaseThread *thread, WD *wd, const DeviceData * dd )
         {
            return setDevice( thread, wd, dd->getDevice() );
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

            // Choose the device where the task will be executed
            if ( next && !( next->hasActiveDevice() ) ) {
               // Check we have recorded good (and reliable) enough results for each PE
               unsigned long wdId =  next->getVersionGroupId();
               size_t paramsSize = next->getParamsSize();
               WDExecInfoKey key = std::make_pair( wdId, paramsSize );
               WDExecInfoData &data = tdata._wdExecStats[key];
               ProcessingElement *pe = thread->runningOn();

               // First record for the given { wdId, paramsSize }
               if ( data.size() == 0 ) {

                  debug( "[versioning] First record for wd key (" + toString<unsigned long>( key.first )
                        + ", " + toString<size_t>( key.second ) + ") with "
                        + toString<int>( next->getNumDevices() ) + " versions" );

                  _lock.acquire();
                  // Reserve as much memory as we need for all the implementations
                  data.reserve( next->getNumDevices() );
                  data = *NEW WDExecInfoData( next->getNumDevices() );

                  unsigned int i;
                  for ( i = 0; i < data.size(); i++ ) {
                     data[i]._elapsedTime = 0.0;
                     data[i]._lastElapsedTime = 0.0;
                     data[i]._numRecords = -1;
                     data[i]._pe = NULL;
                     data[i]._device = NULL;
                  }
                  _lock.release();

                  if ( next->canRunIn( *pe ) ) {
                     // If the thread can run the task, activate its device

                     for ( i = 0; i < next->getNumDevices(); i++ ) {
                        if ( next->getDevices()[i]->isCompatible( pe->getDeviceType() ) ) break;
                     }
                  } else {
                     // Else, activate the first device
                     i = 0;
                  }

                  return setDevice( thread, next, next->getDevices()[i] );
               }

               _lock.acquire();
               unsigned int i;

               // First, check if the thread can run and, in fact, has to run the task
               if ( next->canRunIn( *pe ) ) {
                  for ( i = 0; i < data.size(); i++ ) {
                     WDExecRecords & records = data[i];

                     if ( records._device == NULL ) records._device = &pe->getDeviceType();
                     if ( records._device->getName() == pe->getDeviceType().getName() ) {
                        if ( records._numRecords < MIN_RECORDS ) {
                           // Not enough records to have reliable values

                           debug("[versioning] Less than 3 records for my device ("
                                 + toString<int>( records._numRecords ) + ") for key ("
                                 + toString<unsigned long>( key.first ) + ", "
                                 + toString<size_t>( key.second ) + ") device "
                                 + records._device->getName() );

                           _lock.release();
                           return setDevice( thread, next, records._device );
                        }

                        double sqDev = records._elapsedTime - records._lastElapsedTime;
                        sqDev *= sqDev;
                        if ( sqrt( sqDev ) > MAX_DEVIATION ) {
                           // Values differ too much from each other, compute again

                           debug("[versioning] Too much difference in records for my device ("
                                 + toString<double>( sqrt( sqDev ) ) + " > "
                                 + toString<double>( MAX_DEVIATION ) + ") for key ("
                                 + toString<unsigned long>( key.first ) + ", "
                                 + toString<size_t>( key.second ) + ") device "
                                 + records._device->getName() );

                           _lock.release();
                           return setDevice( thread, next, records._device );
                        }
                     }
                  }
               }

               for ( i = 0; i < data.size(); i++ ) {
                  WDExecRecords & records = data[i];

                  if ( records._numRecords < MIN_RECORDS ) {
                     // Not enough records to have reliable values

                     debug("[versioning] Less than 3 records ("
                           + toString<int>( records._numRecords ) + ") for key ("
                           + toString<unsigned long>( key.first ) + ", "
                           + toString<size_t>( key.second ) + ") device "
                           + records._device->getName() );

                     _lock.release();
                     return setDevice( thread, next, records._device );
                  }

                  double sqDev = records._elapsedTime - records._lastElapsedTime;
                  sqDev *= sqDev;
                  if ( sqrt( sqDev ) > MAX_DEVIATION ) {
                     // Values differ too much from each other, compute again

                     debug("[versioning] Too much difference in records ("
                           + toString<double>( sqrt( sqDev ) ) + " > "
                           + toString<double>( MAX_DEVIATION ) + ") for key ("
                           + toString<unsigned long>( key.first ) + ", "
                           + toString<size_t>( key.second ) + ") device "
                           + records._device->getName() );

                     _lock.release();
                     return setDevice( thread, next, records._device );
                  }
               }

               // Reaching this point means that we have enough records to decide
               ProcessingElement * bestPE = tdata._wdExecBest.find( key )->first;

               debug("[versioning] Autochoosing for key ("
                     + toString<unsigned long>( key.first ) + ", "
                     + toString<size_t>( key.second ) + ") device "
                     + bestPE->getDeviceType().getName() );

               _lock.release();
               return setDevice( thread, next, &( bestPE->getDeviceType() ) );
            }

            return next;
         }

         WD * atBeforeExit ( BaseThread *thread, WD &currentWD )
         {
            if ( currentWD.getNumDevices() > 1 ) {
               TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();
               unsigned long wdId = currentWD.getVersionGroupId();
               size_t paramsSize = currentWD.getParamsSize();
               ProcessingElement * pe = thread->runningOn();
               double executionTime = currentWD.getExecutionTime();

               WDExecInfoKey key = std::make_pair( wdId, paramsSize );

               _lock.acquire();
               WDExecInfoData & data = tdata._wdExecStats[key];

               // Record statistic values
               // Update stats
               // TODO: Choose the appropriate device
               bool found = false;
               unsigned int i;
               for ( i = 0; i < data.size(); i++ ) {
                  if ( data[i]._device == NULL ) break;
                  if ( data[i]._device->getName() == pe->getDeviceType().getName() ) {
                     if ( data[i]._numRecords >= 0 ) found = true;
                     break;
                  }
               }

               if ( !found ) {
                  // Here 'i' points to the first free position to record the values for the given PE
                  // As it is the first time for the given PE, we omit the results because
                  // they can be potentially worse than future executions
                  WDExecRecords & records = data[i];
                  records._pe = pe;
                  records._device = &pe->getDeviceType();
                  records._elapsedTime = 0.0; // Should be 'executionTime'
                  records._numRecords++; // Should be '1' but in fact it is -1+1 = 0
                  records._lastElapsedTime = 0.0; // Should be 'executionTime'

                  debug("[versioning] First recording for key (" + toString<unsigned long>( key.first )
                        + ", " + toString<size_t>( key.second )
                        + ") {pe=" + toString<void *>( records._pe )
                        + ", dev=" + records._device->getName()
                        + ", #=" + toString<int>( records._numRecords )
                        + ", T=" + toString<double>( records._elapsedTime )
                        + ", T2=" + toString<double>( records._lastElapsedTime )
                        + "}; exec time = " + toString<double>( executionTime ) );

               } else {
                  // Here 'i' points to the position associated to the given PE
                  WDExecRecords & records  = data[i];
                  double time = records._elapsedTime * records._numRecords;
                  records._numRecords++;
                  records._elapsedTime = ( time + executionTime ) / records._numRecords;
                  records._lastElapsedTime = executionTime;

                  debug("[versioning] Recording for key (" + toString<unsigned long>( key.first )
                        + ", " + toString<size_t>( key.second )
                        + ") {pe=" + toString<void *>( records._pe )
                        + ", dev=" + records._device->getName()
                        + ", #=" + toString<int>( records._numRecords )
                        + ", T=" + toString<double>( records._elapsedTime )
                        + ", T2=" + toString<double>( records._lastElapsedTime )
                        + "}; exec time = " + toString<double>( executionTime ) );

               }

               // Check if it is the best time
               WDBestRecordData &bestData = tdata._wdExecBest[key];
               bool isBestTime = ( bestData.second > executionTime ) || ( bestData.first == NULL );
               if ( isBestTime ) {
                  // New best value recorded
                  bestData.first = pe;
                  bestData.second = executionTime;

                  debug("[versioning] New best time: {pe=" + toString<void *>( bestData.first )
                        + ", T=" + toString<double>( bestData.second ) + "}" );

               }
               _lock.release();
            }

            return NULL;
         }
   };

   bool Versioning::_useStack = false;
   Lock Versioning::_lock;

   class VersioningSchedPlugin : public Plugin
   {

      public:
         VersioningSchedPlugin() : Plugin( "Versioning scheduling Plugin", 1 ) {}

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

