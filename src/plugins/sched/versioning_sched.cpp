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

namespace nanos {
   namespace ext {

      class Versioning : public SchedulePolicy
      {

         private:
            struct TeamData : public ScheduleTeamData
            {
               public:
                  // WDBestRecord { wdType, elapsedTime }
                  typedef HashMap< int, double > WDBestRecord;
                  // WDExecInfoKey { wdType, paramsSize, PE }
                  // WDExecInfoData { elapsedTime, numRecords }
                  typedef std::pair< std::pair<int, size_t >, ProcessingElement *> WDExecInfoKey;
                  typedef std::pair< double, unsigned int > WDExecInfoData;
                  typedef HashMap< WDExecInfoKey, WDExecInfoData > WDExecInfo;

                  WDBestRecord      _wdExecBest;
                  WDExecInfo        _wdExecStats;

                  WDDeque           _readyQueue;

                  TeamData () : ScheduleTeamData(), _wdExecBest(), _wdExecStats(), _readyQueue() {}
                  ~TeamData () {}

                  inline double getExecutionTime( WDExecInfoData &info ) { return info.first; }
                  inline int getNumRecords( WDExecInfoData &info ) { return info.second; }
                  WDExecInfoData * getExecStats ( int wdType, size_t paramsSize, ProcessingElement *pe )
                  {
                     if ( _wdExecStats.find(std::pair< std::pair<wdType, paramsSize >, pe>) ) {
                        return _wdExecStats.find(std::pair< std::pair<wdType, paramsSize >, pe>);
                     } else {
                        return NULL;
                     }
                  }
            };

         public:
            static bool       _useStack;

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
                  ProcessingElement *pe = getFastestPE( next );
                  if ( pe ) {
                     next->activateDevice( pe->getDeviceType() );
                     if ( !next->canRunIn( *thread->runningOn() ) ) {
                        next = NULL;
                     }
                  } else {
                     next->activateDevice( thread->runningOn()->getDeviceType() );
                  }
               }
            }

            WD *atBeforeExit ( BaseThread *thread, WD &currentWD )
            {
               TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();

               if ( tdata._wdExecStats.find( WDExecInfoKey( currentWD.getId(), currentWD.getParamsSize(),
                     currentWD.getExecutionTime() ) ) ) {
                  // Record statistic values
                  WDExecInfo &info = tdata._wdExecStats.find( WDExecInfoKey( currentWD.getId(),
                        currentWD.getParamsSize(), currentWD.getExecutionTime() ) );

                  //info.second.first
                  // Check if it is the best time


               } else {
                  // First recorded time
                  tdata._wdExecStats.insert( WDExecInfoKey( currentWD.getId(), currentWD.getParamsSize(),
                        currentWD.getExecutionTime() ) );
               }

               return 0;
            }

            ProcessingElement * getFastestPE( WD * wd );

            double getBestElapsedTime( int wdId, size_t paramSize );
            void setBestElapsedTime( int wdId, size_t paramSize, double time, ProcessingElement * pe );
      };

      bool Versioning::_useStack = false;

      class VersioningSchedPlugin : public Plugin
      {

         public:
         VersioningSchedPlugin() : Plugin( "Versioning scheduling Plugin", 1 ) {}

            virtual void config ( Config &config )
            {
               config.setOptionsSection( "Versioning module", "Versioning scheduling module" );
               config.registerConfigOption ( "versioning-use-stack",
                     NEW Config::FlagOption( Versioning::_useStack ),
                     "Stack usage for the versioning policy" );
               config.registerArgOption( "versioning-use-stack", "versioning-use-stack" );

               config.registerAlias ( "versioning-use-stack", "versioning-stack",
                     "Stack usage for the versioning policy" );
               config.registerArgOption ( "versioning-stack", "versioning-stack" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW Versioning());
            }
      };

   }
}

/*******************
ProcessingElement * ScheduleWDVersion::getFastestPE( WD * wd )
{
   WDExecInfo::iterator it = _wdExecInfo.find( WDExecInfoKey( wd->getId(), wd->getDataSize() ) );
   return ( it != _wdExecInfo.end() ) ? it->second.second : NULL;
}

double ScheduleWDVersion::getBestElapsedTime( int wdId, size_t paramSize )
{
   WDExecInfo::iterator it = _wdExecInfo.find( WDExecInfoKey( wdId, paramSize ) );
   return ( it != _wdExecInfo.end() ) ? it->second.first : std::numeric_limits<double>::max();
}

void ScheduleWDVersion::setBestElapsedTime( int wdId, size_t paramSize, double time, ProcessingElement * pe)
{
   _lock.acquire();
   _wdExecInfo[WDExecInfoKey( wdId, paramSize )] = WDExecInfoData( time, pe );
   _lock.release();
}
 */


nanos::ext::VersioningSchedPlugin NanosXPlugin;

