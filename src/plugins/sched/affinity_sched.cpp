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
               std::set<int>      _freeNodes;
               std::set<int>      _nodeSet;
               Lock               _freeNodesLock;
               Atomic<int>       *_feeding;
 
               TeamData ( unsigned int size ) : ScheduleTeamData()
               {
                  unsigned int nodes = sys.getNetwork()->getNumNodes();
                  unsigned int numqueues = nodes; //(nodes > 0) ? nodes + 1 : nodes;
                  _numNodes = ( sys.getNetwork()->getNodeNum() == 0 ) ? nodes : 1;

                  _nodeSet.insert( 0 );
                  if ( _numNodes > 1 ) {
                     _readyQueues = NEW WDDeque[numqueues];
                     _bufferQueues = NEW WDDeque[numqueues];
                     _createdData = NEW std::size_t[numqueues];
                     for (unsigned int i = 0; i < numqueues; i += 1 ) _createdData[i] = 0;
                     _holdTasks = false;
                     for( unsigned int i = 1; i < sys.getNumMemorySpaces(); i += 1 ) {
                        if ( sys.getCaches()[ i ]->getNodeNumber() != 0 ) _nodeSet.insert( i );
                     }
                  }
                  _feeding = NEW Atomic<int>[_numNodes];
                  for (unsigned int i = 0; i < _numNodes; i += 1) _feeding[ i ] = 0;
                  _freeNodes.insert(_nodeSet.begin(), _nodeSet.end() );
               }

               ~TeamData ()
               {
                  delete[] _readyQueues;
                  delete[] _bufferQueues;
                  delete[] _createdData;
               }
            };

            struct NoCopy
            {
               static inline bool check( WD &wd, BaseThread &thread )
               {
                  //TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                        NewDirectory::LocationInfoList &locs = wd._ccontrol._cacheCopies[ i ]._locations;
                        for ( NewDirectory::LocationInfoList::iterator it = locs.begin(); it != locs.end(); it++ ) {
                           if ( !it->second.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) ) {
                              return false;
                           }
                        }
                     }
                  }
                  return true;
               }
            };

            struct NetworkSched
            {
               static void notifyCopy( WD &wd, BaseThread &thread ) {
                  TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  while( tdata._feeding[ data._cacheId ].cswap( 1, 0 ) != 1 );
                         //  std::cerr <<"cleared copy by wd " << wd.getId() << std::endl;
               }
               static inline bool check( WD &wd, BaseThread &thread )
               {
                  TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  tdata._freeNodesLock.acquire();
                  std::set<int> toBeRemovedElems;
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                        NewDirectory::LocationInfoList &locs = wd._ccontrol._cacheCopies[ i ]._locations;
                        for ( NewDirectory::LocationInfoList::iterator it = locs.begin(); it != locs.end(); it++ ) {
                           if ( !it->second.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) ) {
                              int loc = it->second.getFirstLocation();
                              if ( tdata._freeNodes.count( loc ) == 0 ) {
                                 tdata._freeNodesLock.release();
                                 return false;
                              } else {
                                 toBeRemovedElems.insert( loc );
                              }
                           }
                        }
                     }
                  }
                  if ( !toBeRemovedElems.empty( ) ) {
                     //std::cerr << "feed val is " << tdata._feedingThisNode.value() << std::endl;
                     if ( tdata._feeding[ data._cacheId ].cswap( 0, 1 ) == 1 ) {
                       // std::cerr <<"allow copy to wd " << wd.getId() << std::endl;
                        wd.setNotifyCopyFunc( notifyCopy );
                     } else {
                        tdata._freeNodesLock.release();
                        return false;
                     }
                     
                     //std::cerr <<"thread " << thread.getId() <<" node "<< data._cacheId <<" selected WD " << wd.getId() << " sources of data are ";
                     //for (std::set<int>::iterator it = toBeRemovedElems.begin(); it != toBeRemovedElems.end(); it++ ) {
                     //   std::cerr << *it << " ";
                     //}
                     //std::cerr << std::endl;
                     for (std::set<int>::iterator it = toBeRemovedElems.begin(); it != toBeRemovedElems.end(); it++ ) {
                        tdata._freeNodes.erase( *it );
                     }
                     //std::cerr << " remaining set of free nodes ";
                     //for (std::set<int>::iterator it = tdata._freeNodes.begin(); it != tdata._freeNodes.end(); it++ ) {
                     //   std::cerr << *it << " ";
                     //}
                     //std::cerr << std::endl;
                     if ( tdata._freeNodes.empty() ) {
                        tdata._freeNodes.insert( tdata._nodeSet.begin(), tdata._nodeSet.end() );
                     }
                  } //else { std::cerr <<"thread " << thread.getId() <<" selected WD " << wd.getId() << " empty sources "<<std::endl; }
                  tdata._freeNodesLock.release();
                  return true;
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
#if 1
               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               if ( !data._init ) {
                  //data._cacheId = thread->runningOn()->getMemorySpaceId();
                  data._cacheId = thread->runningOn()->getMyNodeNumber();
                  data._init = true;
               }
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
              
               if ( tdata._numNodes == 1 ) {
                  tdata._globalReadyQueue.push_front( &wd );
                  return;
               }

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
                     if ( tdata._numNodes > 1 ) {
                        //int winner = numCaches - 1;
                        int winner = tdata._numNodes - 1;
                        for ( int i = winner - 1; i >= 0; i -= 1 )
                        {
                           winner = ( tdata._createdData[ winner ] < tdata._createdData[ i ] ) ? winner : i ;
                        }
                        tdata._createdData[ winner ] += createdDataSize;
                        tdata._bufferQueues[ winner ].push_back( &wd );
                        message("init: queue " << (winner) << " for wd " << wd.getId() );
                     } else {
                        message("global queue for wd " << wd.getId() );
                        tdata._globalReadyQueue.push_back( &wd );
                     }
                     //tdata._readyQueues[winner + 1].push_back( &wd );
                     tdata._holdTasks = true;
                  }
                  else
                  {
                     unsigned int ranks[ tdata._numNodes ];
                     if ( tdata._holdTasks.value() )
                     {
                        if ( tdata._holdTasks.cswap( true, false ) )
                        {
                           for ( unsigned int idx = 0; idx < tdata._numNodes; idx += 1) 
                           {
                              tdata._readyQueues[ idx ].transferElemsFrom( tdata._bufferQueues[ idx ] );
                           }
                        }
                     }
                     for (unsigned int i = 0; i < tdata._numNodes; i++ ) {
                        ranks[i] = 0;
                     }
                     for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                        if ( !copies[i].isPrivate() && copies[i].isOutput() ) {
                           NewDirectory::LocationInfoList &locs = wd._ccontrol._cacheCopies[ i ]._locations;
                           for ( NewDirectory::LocationInfoList::iterator it = locs.begin(); it != locs.end(); it++ ) {
                              int loc = it->second.getFirstLocation();
                              ranks[ ( loc != 0 ? sys.getCaches()[ loc ]->getNodeNumber() : 0 ) ] += it->first.getBreadth();
                               if (sys.getNetwork()->getNodeNum() == 0 ) {message("wd " << wd.getId() << " selected queue " << ( loc != 0 ? sys.getCaches()[ loc ]->getNodeNumber() : 0 ) << " loc is " << loc ); }
                           }
		     if (sys.getNetwork()->getNodeNum() == 0 ) { message("check wd " << wd.getId() << " tag " << (void*)copies[i].getAddress()  << " ranks " << ranks[0] << "," << ranks[1] << "," << ranks[2] << "," << ranks[3] << " num readers for this wd: " << wd.getNumReaders() << " all readers: " << wd.getNumAllReaders() ); }
                        }
                     }
                     int winner = -1;
                     unsigned int maxRank = 0;
                     for ( int i = 0; i < ( (int) tdata._numNodes ); i++ ) {
                        if ( ranks[i] > maxRank ) {
                           winner = i;
                           maxRank = ranks[i];
                        }
                     }
		     //message("queued wd " << wd.getId() << " to queue " << winner << " ranks " << ranks[0] << "," << ranks[1] << "," << ranks[2] << "," << ranks[3] );
                     tdata._readyQueues[winner].push_back( &wd );
                  }
               } else {
                  tdata._globalReadyQueue.push_front ( &wd );
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
            data._cacheId = thread->runningOn()->getMyNodeNumber();
            data._init = true;
         }
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         if ( tdata._holdTasks.value() ) 
         {
            if ( tdata._holdTasks.cswap( true, false ) )
            {
               for ( unsigned int idx = 0; idx < tdata._numNodes; idx += 1) 
               {
                  tdata._readyQueues[ idx ].transferElemsFrom( tdata._bufferQueues[ idx] );
               }
            }
         }
         /*
          *  First try to schedule the thread with a task from its queue
          */
         //if ( ( wd = tdata._readyQueues[data._cacheId].pop_front ( thread ) ) != NULL ) {
         
         if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<NetworkSched> ( thread ) ) != NULL ) {
            return wd;
         } else 
         if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<NoCopy> ( thread ) ) != NULL ) {
            //message("Block:: Ive got a wd, Im at node " << data._cacheId );
            return wd;
         } else
         if ( ( wd = tdata._readyQueues[data._cacheId].pop_front( thread ) ) != NULL ) {
            return wd;
         } else {
            /*
             * Then try to get it from the global queue
             */
             wd = tdata._globalReadyQueue.pop_front ( thread );
         }
         if ( !_noSteal )
         {
            if ( wd == NULL ) {
               for ( unsigned int i = data._cacheId + 1; i < tdata._numNodes; i++ ) {
                  if ( !tdata._readyQueues[i].empty() ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
               for ( unsigned int i = 0; i < data._cacheId; i++ ) {
                  if ( !tdata._readyQueues[i].empty() ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
            }
         }

         if ( wd == NULL ) {
            struct timespec req, rem;
            req.tv_sec = 0;
            req.tv_nsec = 100;
            nanosleep( &req, &rem );
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
            data._cacheId = thread->runningOn()->getMyNodeNumber();
            data._init = true;
         }
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         /*
          *  First try to schedule the thread with a task from its queue
          */
         //if ( ( wd = tdata._readyQueues[data._cacheId].pop_front ( thread ) ) != NULL ) {
         if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<NetworkSched> ( thread ) ) != NULL ) {
            return wd;
         } else
         if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<NoCopy> ( thread ) ) != NULL ) {
           //message("Ive got a wd, Im at node " << data._cacheId );
            return wd;
         } else
         if ( ( wd = tdata._readyQueues[data._cacheId].pop_front( thread ) ) != NULL ) {
            return wd;
         } else {
            /*
             * Then try to get it from the global queue
             */
             //message("getting from global... im " << data._cacheId);
             wd = tdata._globalReadyQueue.pop_front ( thread );
         }
         if ( !_noSteal )
         {
            if ( wd == NULL ) {
               for ( unsigned int i = data._cacheId + 1; i < tdata._numNodes; i++ ) {
                  if ( tdata._readyQueues[i].size() > 1 ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
               for ( unsigned int i = 0; i < data._cacheId; i++ ) {
                  if ( tdata._readyQueues[i].size() > 1 ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
            }
         }
         if ( wd == NULL ) {
            struct timespec req, rem;
            req.tv_sec = 0;
            req.tv_nsec = 100;
            nanosleep( &req, &rem );
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
