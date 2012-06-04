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
#include "memtracker.hpp"

namespace nanos {
   namespace ext {

      class CacheSchedPolicy : public SchedulePolicy
      {
         private:

            struct TeamData : public ScheduleTeamData
            {
               WDDeque            _globalReadyQueue;
               WDDeque*           _readyQueues;
               WDDeque*           _bufferQueues;
               std::size_t*       _createdData;
               Atomic<bool>       _holdTasks;
               unsigned int       _numNodes;
 
               TeamData ( unsigned int size ) : ScheduleTeamData()
               {
                  unsigned int nodes = sys.getNetwork()->getNumNodes();
                  unsigned int numqueues = nodes; //(nodes > 0) ? nodes + 1 : nodes;
                  _numNodes = ( sys.getNetwork()->getNodeNum() == 0 ) ? nodes : 1;

                  if ( _numNodes > 1 ) {
		     _readyQueues = NEW WDDeque[numqueues];
		     _bufferQueues = NEW WDDeque[numqueues];
		     _createdData = NEW std::size_t[numqueues];
		     for (unsigned int i = 0; i < numqueues; i += 1 ) _createdData[i] = 0;
		     _holdTasks = false;
                  }
               }

               ~TeamData ()
               {
                  delete[] _readyQueues;
                  delete[] _bufferQueues;
                  delete[] _createdData;
               }
            };

            /** \brief Cache Scheduler data associated to each thread
              *
              */
            struct ThreadData : public ScheduleThreadData
            {
               /*! queue of ready tasks to be executed */
               unsigned int _cacheId;
               bool _init;

               ThreadData () : _cacheId(0), _init(false) {}
               virtual ~ThreadData () {
               }
            };

            /* disable copy and assigment */
            explicit CacheSchedPolicy ( const CacheSchedPolicy & );
            const CacheSchedPolicy & operator= ( const CacheSchedPolicy & );

         public:
            static bool _noSteal;
            // constructor
            CacheSchedPolicy() : SchedulePolicy ( "Cache" ) {}

            // destructor
            virtual ~CacheSchedPolicy() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               /* Queue 0 will be the global one */
               unsigned int numQueues = 0; // will be computed later
               
               return NEW TeamData( numQueues );
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return NEW ThreadData();
            }

            /*!
            *  \brief Enqueue a work descriptor in the readyQueue of the passed thread
            *  \param thread pointer to the thread to which readyQueue the task must be appended
            *  \param wd a reference to the work descriptor to be enqueued
            *  \sa ThreadData, WD and BaseThread
            */
            virtual void queue ( BaseThread *thread, WD &wd )
            {
#if 0
               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               if ( !data._init ) {
                  //data._cacheId = thread->runningOn()->getMemorySpaceId();
                  data._cacheId = thread->runningOn()->getMyNodeNumber();
                  data._init = true;
               }
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

            //message("in queue node " << sys.getNetwork()->getNodeNum()<< " wd os " << wd.getId());
               if ( wd.isTied() ) {
                  //unsigned int index = wd.isTiedTo()->runningOn()->getMemorySpaceId();
                  unsigned int index = wd.isTiedTo()->runningOn()->getMyNodeNumber();
                  tdata._readyQueues[index].push_front ( &wd );
                  return;
               }
               if ( wd.getNumCopies() > 0 ){
                  CopyData * copies = wd.getCopies();
                  unsigned int wo_copies = 0, ro_copies = 0, rw_copies = 0;
                  std::size_t createdDataSize = 0;
                  for (unsigned int idx = 0; idx < wd.getNumCopies(); idx += 1)
                  {
                     if ( !copies[idx].isPrivate() ) {
                        rw_copies += (  copies[idx].isInput() &&  copies[idx].isOutput() );
                        ro_copies += (  copies[idx].isInput() && !copies[idx].isOutput() );
                        wo_copies += ( !copies[idx].isInput() &&  copies[idx].isOutput() );
                        createdDataSize += ( !copies[idx].isInput() && copies[idx].isOutput() ) * copies[idx].getSize();
                     }
                  }

                  if ( wo_copies == wd.getNumCopies() ) /* init task */
                  {
                     //unsigned int numCaches = sys.getCacheMap().getSize();
                     //message("numcaches is " << numCaches);
                     if ( _numNodes > 1 ) {
                        //int winner = numCaches - 1;
                        int winner = _numNodes - 1;
                        for ( int i = winner - 1; i >= 0; i -= 1 )
                        {
                           winner = ( tdata._createdData[ winner ] < tdata._createdData[ i ] ) ? winner : i ;
                        }
                        tdata._createdData[ winner ] += createdDataSize;
                        tdata._bufferQueues[ winner ].push_back( &wd );
                        //message("init: queue " << (winner) << " for wd " << wd.getId() );
                     } else {
                        tdata._globalReadyQueue.push_back( &wd );
                     }
                     //tdata._readyQueues[winner + 1].push_back( &wd );
                     tdata._holdTasks = true;
                  }
                  else
                  {
                     unsigned int ranks[ _numNodes + 1];
                     if ( tdata._holdTasks.value() )
                     {
                        if ( tdata._holdTasks.cswap( true, false ) )
                        {
                           for ( unsigned int idx = 0; idx <= _numNodes; idx += 1) 
                           {
                              tdata._readyQueues[ idx ].transferElemsFrom( tdata._bufferQueues[ idx ] );
                           }
                        }
                     }
                     for (unsigned int i = 0; i < _numNodes; i++ ) {
                        ranks[i] = 0;
                     }
                     for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                        if ( !copies[i].isPrivate() && copies[i].isOutput() ) {
                           WorkDescriptor* parent = wd.getParent();
                           if ( parent != NULL ) {
                              Directory *dir = parent->getDirectory();
                              if ( dir != NULL ) {
                                 DirectoryEntry *de = dir->findEntry(copies[i].getAddress());
                                 if ( de != NULL ) {
                                    for ( unsigned int j = 0; j < numCaches; j++ ) {
                                       ranks[j]+=((unsigned int)(de->getAccess( j+1 ) > 0))*copies[i].getSize();
                                    }
                                 }
                              }
                           }
		     //message("check wd " << wd.getId() << " tag " << (void*)copies[i].getAddress()  << " ranks " << ranks[0] << "," << ranks[1] << "," << ranks[2] << "," << ranks[3] );
                        }
                     }
                     unsigned int winner = 1;
                     unsigned int maxRank = 0;
                     for ( unsigned int i = 0; i < _numNodes+1; i++ ) {
                        if ( ranks[i] > maxRank ) {
                           winner = i+1;
                           maxRank = ranks[i];
                        }
                     }
		     message("queued wd " << wd.getId() << " to queue " << winner << " ranks " << ranks[0] << "," << ranks[1] << "," << ranks[2] << "," << ranks[3] );
                     tdata._readyQueues[winner].push_back( &wd );
                  }
               } else {
                  tdata._readyQueues[0].push_front ( &wd );
               }
#endif
            }

            /*!
            *  \brief Function called when a new task must be created: the new created task
            *          is directly queued (Breadth-First policy)
            *  \param thread pointer to the thread to which belongs the new task
            *  \param wd a reference to the work descriptor of the new task
            *  \sa WD and BaseThread
            */
            virtual WD * atSubmit ( BaseThread *thread, WD &newWD )
            {
               queue(thread,newWD);

               return 0;
            }

            virtual WD *atIdle ( BaseThread *thread );
            virtual WD *atBlock ( BaseThread *thread, WD *current );

            virtual WD *atAfterExit ( BaseThread *thread, WD *current )
            {
               return atBlock(thread, current );
            }

            WD * atPrefetch ( BaseThread *thread, WD &current )
            {
               WD * found = current.getImmediateSuccessor(*thread);
         
               return found != NULL ? found : atIdle(thread);
            }
         
            WD * atBeforeExit ( BaseThread *thread, WD &current )
            {
               return current.getImmediateSuccessor(*thread);
            }

      };

      WD *CacheSchedPolicy::atBlock ( BaseThread *thread, WD *current )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         if ( !data._init ) {
            //data._cacheId = thread->runningOn()->getMemorySpaceId();
            data._cacheId = thread->runningOn()->getMyNodeNumber() + 1;
            data._init = true;
         }
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         if ( tdata._holdTasks.value() ) 
         {
            if ( tdata._holdTasks.cswap( true, false ) )
            {
               unsigned int numCaches = sys.getCacheMap().getSize();
               for ( unsigned int idx = 1; idx <= numCaches+1; idx += 1) 
               {
                  tdata._readyQueues[ idx ].transferElemsFrom( tdata._bufferQueues[ idx] );
               }
            }
         }
         /*
          *  First try to schedule the thread with a task from its queue
          */
         if ( ( wd = tdata._readyQueues[data._cacheId].pop_front ( thread ) ) != NULL ) {
            message("Block:: Ive got a wd, Im at node " << data._cacheId );
            return wd;
         } else {
            /*
             * Then try to get it from the global queue
             */
             wd = tdata._readyQueues[0].pop_front ( thread );
         }
         if ( !_noSteal )
         {
            if ( wd == NULL ) {
               for ( unsigned int i = data._cacheId; i < sys.getCacheMap().getSize(); i++ ) {
                  if ( !tdata._readyQueues[i+1].empty() ) {
                     wd = tdata._readyQueues[i+1].pop_front( thread );
                     return wd;
                  } 
               }
               for ( unsigned int i = 0; i < data._cacheId; i++ ) {
                  if ( !tdata._readyQueues[i+1].empty() ) {
                     wd = tdata._readyQueues[i+1].pop_front( thread );
                     return wd;
                  } 
               }
            }
         }
         return wd;
      }

      /*! 
       */
      WD * CacheSchedPolicy::atIdle ( BaseThread *thread )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         if ( !data._init ) {
            //data._cacheId = thread->runningOn()->getMemorySpaceId();
            data._cacheId = thread->runningOn()->getMyNodeNumber() + 1;
            data._init = true;
         }
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         /*
          *  First try to schedule the thread with a task from its queue
          */
         if ( ( wd = tdata._readyQueues[data._cacheId].pop_front ( thread ) ) != NULL ) {
           message("Ive got a wd, Im at node " << data._cacheId );
            return wd;
         } else {
            /*
             * Then try to get it from the global queue
             */
             //message("getting from global... im " << data._cacheId);
             wd = tdata._readyQueues[0].pop_front ( thread );
         }
         if ( !_noSteal )
         {
            if ( wd == NULL ) {
               for ( unsigned int i = data._cacheId; i < sys.getCacheMap().getSize(); i++ ) {
                  if ( tdata._readyQueues[i+1].size() > 1 ) {
                     wd = tdata._readyQueues[i+1].pop_front( thread );
                     return wd;
                  } 
               }
               for ( unsigned int i = 0; i < data._cacheId; i++ ) {
                  if ( tdata._readyQueues[i+1].size() > 1 ) {
                     wd = tdata._readyQueues[i+1].pop_front( thread );
                     return wd;
                  } 
               }
            }
         }
         return wd;
      }

      bool CacheSchedPolicy::_noSteal = false;

      class CacheSchedPlugin : public Plugin
      {
         public:
            CacheSchedPlugin() : Plugin( "Cache-guided scheduling Plugin",1 ) {}

            virtual void config( Config& cfg )
            {
               cfg.setOptionsSection( "Affinity module", "Data Affinity scheduling module" );
               cfg.registerConfigOption ( "affinity-no-steal", NEW Config::FlagOption( CacheSchedPolicy::_noSteal ), "Steal tasks from other threads");
               cfg.registerArgOption( "affinity-no-steal", "affinity-no-steal" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW CacheSchedPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-affinity",nanos::ext::CacheSchedPlugin);
