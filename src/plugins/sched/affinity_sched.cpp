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
#include "clusterthread_decl.hpp"
#include "regioncache.hpp"
#include "newregiondirectory.hpp"

namespace nanos {
   namespace ext {

      class CacheSchedPolicy : public SchedulePolicy
      {
         private:

            struct TeamData : public ScheduleTeamData
            {
               WDDeque            _globalReadyQueue;
               WDDeque            _unrankedQueue;
               WDDeque*           _readyQueues;
               WDDeque*           _readyQueuesAlreadyInit;
               WDDeque*           _bufferQueues;
               std::size_t*       _createdData;
               Atomic<bool>       _holdTasks;
               unsigned int       _numNodes;
               std::set<int>      _freeNodes;
               std::set<int>      _nodeSet;
               Lock               _freeNodesLock;
               Atomic<int>       *_feeding;
               int               *_feedingVector;
               unsigned int      *_load;
               Atomic<unsigned int> _getting;
               unsigned int       _lastNodeScheduled;
               std::vector< memory_space_id_t > *_nodeToMemSpace;
 
               TeamData ( unsigned int size ) : ScheduleTeamData(), _globalReadyQueue(), _unrankedQueue()
               {
                  unsigned int nodes = sys.getNetwork()->getNumNodes();
                  unsigned int numqueues = nodes; //(nodes > 0) ? nodes + 1 : nodes;
                  _numNodes = ( sys.getNetwork()->getNodeNum() == 0 ) ? nodes : 1;
                  _getting = 0;

                  _holdTasks = false;
                  _nodeSet.insert( 0 );
                  if ( _numNodes > 1 ) {
                     _readyQueues = NEW WDDeque[numqueues];
                     _readyQueuesAlreadyInit = NEW WDDeque[numqueues];
                     _bufferQueues = NEW WDDeque[numqueues];
                     _createdData = NEW std::size_t[numqueues];
                     for (unsigned int i = 0; i < numqueues; i += 1 ) {
                        _createdData[i] = 0;
                     }
                     _nodeToMemSpace = NEW std::vector< memory_space_id_t >( _numNodes );
                     (*_nodeToMemSpace)[ 0 ] = 0;
                     for( unsigned int i = 1; i <= sys.getSeparateMemoryAddressSpacesCount(); i += 1 ) {
                        //if ( sys.getCaches()[ i ]->getNodeNumber() != 0 ) _nodeSet.insert( i );
                        if ( sys.getSeparateMemory( i ).getNodeNumber() != 0 ) {
                           _nodeSet.insert( i );
                           std::cerr << " location " << i << " is node " << sys.getSeparateMemory( i ).getNodeNumber() << std::endl;
                           (*_nodeToMemSpace)[ sys.getSeparateMemory( i ).getNodeNumber() ] = i;
                        }
                     }
                  }
                  _load = NEW unsigned int[_numNodes];
                  _feeding = NEW Atomic<int>[_numNodes];
                  _feedingVector = NEW int[ _numNodes * 2 ];
                  for (unsigned int i = 0; i < _numNodes; i += 1) {
                     _feeding[ i ] = 0;
                     _feedingVector[ i * 2 ] = 0;
                     _feedingVector[ i * 2 + 1 ] = 0;
                     _load[ i ] = 0;
                  }
                  _freeNodes.insert(_nodeSet.begin(), _nodeSet.end() );
                  _lastNodeScheduled = 1;
             
               }

               ~TeamData ()
               {
                  delete[] _readyQueues;
                  delete[] _readyQueuesAlreadyInit;
                  delete[] _bufferQueues;
                  delete[] _createdData;
                  /* TODO add delete for new members */
               }
            };

            template <class CondA, class CondB>
            struct Or {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return CondA::check( wd, thread ) || CondB::check( wd, thread );
               }
            };

            template <class CondA, class CondB>
            struct And {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return CondA::check( wd, thread ) && CondB::check( wd, thread );
               }
            };

            template <class Cond>
            struct Not {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return !Cond::check( wd, thread );
               }
            };

            struct AlreadyInit {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  if ( wd.initialized() ) return true;
                  else return false;
               }
            };

            struct AlreadyDataInit {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  if ( wd.initialized() && wd._mcontrol.isDataReady() ) return true;
                  else return false;
               }
            };

            struct WouldNotTriggerInvalidation
            {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return wd.resourceCheck( thread, false );
               }
            };

            struct WouldNotRunOutOfMemory
            {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return wd.resourceCheck( thread, true );
               }
            };

            struct NoCopy
            {
               static inline bool check( WD &wd, BaseThread const &thread )
               {
                  //TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                           NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                           if ( !locs.empty() ) {
                              for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn()->getMemorySpaceId() ) ) {
                                    return false;
                                 }
                              }
                           } else {
                              if ( ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) )
                                 return false;
                           }
                     }
                  }
                  return true;
               }
            };
            //struct SiCopy
            //{
            //   static inline bool check( WD &wd, BaseThread const &thread )
            //   {
            //      //TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
            //      //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
            //      CopyData * copies = wd.getCopies();
            //      for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
            //         if ( !copies[i].isPrivate() && copies[i].isInput() ) {
            //               //NewLocationInfoList const &locs = wd._mcontrol.getCacheCopies()[ i ].getNewLocations();
            //               NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
            //               if ( !locs.empty() ) {
            //                  for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
            //                     if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn()->getMemorySpaceId() ) ) {
            //                        return true;
            //                     }
            //                  }
            //               } else {
            //                  if ( ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) ) 
            //                     return true;
            //               }
            //         }
            //      }
            //      return false;
            //   }
            //};
            struct SiCopySiMaster
            {
               static inline bool check( WD &wd, BaseThread const &thread )
               {
                  //TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                           ///NewLocationInfoList const &locs = wd._mcontrol.getCacheCopies()[ i ].getNewLocations();
                           NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                           if ( ! locs.empty() ) {
                              for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn()->getMemorySpaceId() ) && NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, 0) ) {
                                    return true;
                                 }
                              }
                           } else {
                              if ( ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) && wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( 0 ) )
                                 return true;
                           }
                     }
                  }
                  return false;
               }
            };
            struct SiCopySiMasterInit
            {
               static inline bool check( WD &wd, BaseThread const &thread )
               {
                  //TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  if ( wd.initialized() ) return false;
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                           //NewLocationInfoList const &locs = wd._mcontrol.getCacheCopies()[ i ].getNewLocations();
                           NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                           if ( !locs.empty() ) {
                              for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn()->getMemorySpaceId() )
                                 && NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, 0 )  ) {
                                    return true;
                                }
                              }
                           } else {
                              if ( ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) &&
                                 wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( 0 ) )
                                 return true;
                           }
                     }
                  }
                  return false;
               }
            };
            struct SiCopyNoMasterInit
            {
               static inline bool check( WD &wd, BaseThread const &thread )
               {
                  //TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  if ( wd.initialized() ) return false;
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                           //NewLocationInfoList const &locs = wd._mcontrol.getCacheCopies()[ i ].getNewLocations();
                           NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                           if ( !locs.empty() ) {
                              for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn()->getMemorySpaceId() ) && ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, 0 ) ) {
                                    return true;
                                 }
                              }
                           } else {
                                 if ( ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) && 
                                      ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( 0 ) ) 
                                    return true;
                           }
                     }
                  }
                  return false;
               }
            };
            struct SiCopyNoMaster
            {
               static inline bool check( WD &wd, BaseThread const &thread )
               {
                  //TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                           //NewLocationInfoList const &locs = wd._mcontrol.getCacheCopies()[ i ].getNewLocations();
                           NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                           if ( !locs.empty() ) {
                              for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn()->getMemorySpaceId() ) && ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, 0 ) ) {
                                    return true;
                                 }
                              }
                           } else {
                              if ( ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) &&
                                    ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( 0 ) )
                                    return true;
                           }
                     }
                  }
                  return false;
               }
            };

#if 0
            struct NetworkSched
            {
               struct NotifyData {
                  NotifyData( unsigned int numNodes, unsigned int dst ) : _dstNode( dst ) {
                     _srcNode = NEW unsigned int [numNodes];
                  }
                  unsigned int *_srcNode;
                  unsigned int _dstNode;
                  //std::size_t *_len;
               }
               static void notifyCopy( WD &wd, BaseThread &thread, void *ndata ) {
                  TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  NotifyData * notifyData = ( NotifyData * ) ndata;
                  
                  //while( tdata._feeding[ data._cacheId ].cswap( 1, 0 ) != 1 );
                  tdata._freeNodesLock.acquire();
                  //tdata._freeNodes.insert( data._cacheId );
                  tdata._feedingVector[ srcNode ]
                  tdata._freeNodesLock.release();
                  //       std::cerr <<"cleared copy by wd " << wd.getId() << " id is " << data._cacheId  <<std::endl;
               }

               static inline bool check( WD &wd, BaseThread const &thread )
               {
                  TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
                  //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
                  tdata._freeNodesLock.acquire();
                  std::set<int> toBeRemovedElems;
                  CopyData * copies = wd.getCopies();
                  unsigned int allLocs = 0, gotLocs = 0;
                  int tmpFeedingVector[ tdata._numNodes * 2 ];
                  int srcNodes[ tdata._numNodes ];
                  for (unsigned int i = 0; i < tdata._numNodes; i +=1 ) srcNodes[i] = 0;
                  ::memcpy( tmpFeedingVector, tdata._feedingVector, sizeof( int ) * tdata._numNodes * 2 );

                  // check destination (its this thread)
                  if ( tmpFeedingVector[ data._cacheId * 2 ] + 1 > 2 ) {
                  tdata._freeNodesLock.release();
                                 return false;
                  }

                  if (sys.getNetwork()->getNodeNum() == 0)std::cerr <<"check wd " << wd.getId() << std::endl;
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                        NewDirectory::LocationInfoList const &locs = wd._mcontrol._cacheCopies[ i ].getLocations();
                        for ( NewDirectory::LocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                           allLocs+= 1;
                           if ( !it->second.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) ) {
                              int loc = it->second.getFirstLocation();
                              srcNodes[ loc ] = 1;
                              //tmpFeedingVector[ loc * 2 + 1 ] += 1;       //src
                              if ( tmpFeedingVector[ loc * 2 + 1 ] + 1 > 2 /* XXX THRESHOLD */ ) { 
                                 tdata._freeNodesLock.release();
                                 return false;
                              }
                              //if ( tdata._freeNodes.count( loc ) == 0 ) {
                              //   if (sys.getNetwork()->getNodeNum() == 0)std::cerr <<"loc busy... " << loc << std::endl;
                              //   tdata._freeNodesLock.release();
                              //   return false;
                              //} else {
                              //   toBeRemovedElems.insert( loc );
                              //}
                           } else {
                              gotLocs += 1; 
                           }
                        }
                     }
                  }
                  if ( allLocs == gotLocs ) {
                  tdata._freeNodesLock.release();
                     return false;
                  } else {
                     NotifyData *ndata = NEW ( allLocs - gotLocs, data._cacheId );
                     for ( unsigned int i = 0; i < tdata._numNodes; i += 1 ) {
                        tmpFeedingVector[ i ] += srcNodes[ i ];
                        ndata->_srcNodes
                     }
                     //for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     //   if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                     //      NewDirectory::LocationInfoList const &locs = wd._mcontrol._cacheCopies[ i ].getLocations();
                     //      for ( NewDirectory::LocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                     //         if ( !it->second.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) ) {
                     //            int loc = it->second.getFirstLocation();
                     //            tmpFeedingVector[ data._cacheId * 2 ] += 1;
                     //            tmpFeedingVector[ loc * 2 + 1 ] += 1;
                     //         }
                     //      }
                     //   }
                     //}
                     // XXX commit_changes_to_FeedingVector();
                     wd.setNotifyCopyFunc( notifyCopy, (void *) NEW NotifyData( data._cacheId, loc )  );
                     for ( unsigned int i = 0; i < tdata._numNodes; i += 1 ) {
                        tdata._feedingVector[ i * 2 ] = tmpFeedingVector[ i * 2 ];
                        tdata._feedingVector[ i * 2 + 1 ] = tmpFeedingVector[ i * 2 + 1 ];
                     }
                  tdata._freeNodesLock.release();
                     return true;
                  }
                  //if ( !toBeRemovedElems.empty( ) ) {
                  //   //std::cerr << "feed val is " << tdata._feedingThisNode.value() << std::endl;
                  //   if ( tdata._feeding[ data._cacheId ].cswap( 0, 1 ) == 1 ) {
                  //      if (sys.getNetwork()->getNodeNum() == 0)std::cerr <<"allow copy to wd " << wd.getId() << " dest " << data._cacheId << std::endl;
                  //      wd.setNotifyCopyFunc( notifyCopy );
                  //   } else {
                  // if (sys.getNetwork()->getNodeNum() == 0)std::cerr <<"stolen... " << std::endl;
                  //      tdata._freeNodesLock.release();
                  //      return false;
                  //   }
                  //   
                  //   //std::cerr <<"thread " << thread.getId() <<" node "<< data._cacheId <<" selected WD " << wd.getId() << " sources of data are ";
                  //   //for (std::set<int>::iterator it = toBeRemovedElems.begin(); it != toBeRemovedElems.end(); it++ ) {
                  //   //   std::cerr << *it << " ";
                  //   //}
                  //   //std::cerr << std::endl;
                  //   for (std::set<int>::iterator it = toBeRemovedElems.begin(); it != toBeRemovedElems.end(); it++ ) {
                  //      tdata._freeNodes.erase( *it );
                  //   }
                  //   //std::cerr << " remaining set of free nodes ";
                  //   //for (std::set<int>::iterator it = tdata._freeNodes.begin(); it != tdata._freeNodes.end(); it++ ) {
                  //   //   std::cerr << *it << " ";
                  //   //}
                  //   //std::cerr << std::endl;
                  //   //if ( tdata._freeNodes.empty() ) {
                  //   //   tdata._freeNodes.insert( tdata._nodeSet.begin(), tdata._nodeSet.end() );
                  //   //}
                  //} else { 
                  //   if (sys.getNetwork()->getNodeNum() == 0)std::cerr <<"thread " << thread.getId() <<" selected WD " << wd.getId() << " empty sources, allLocs  " << allLocs << " gotLocs "<< gotLocs <<std::endl;
                  //   if ( allLocs == gotLocs )
                  //   tdata._freeNodesLock.release();
                  //   return false;
                  //}
                  //tdata._freeNodesLock.release();
                  return true;
               }
            };
#endif

            static void getComplete( WD &wd, BaseThread &thread ) {
               TeamData &tdata = (TeamData &) *thread.getTeam()->getScheduleData();
               //ThreadData &data = ( ThreadData & ) *thread.getTeamData()->getScheduleData();
               //std::cerr << " CLEAR GETTING " << std::endl;
               //while ( !tdata._getting.cswap( true, false ) ) { }
               tdata._getting--;
               
            }
            /** \brief Cache Scheduler data associated to each thread
              *
              */
            struct ThreadData : public ScheduleThreadData
            {
               /*! queue of ready tasks to be executed */
               unsigned int _cacheId;
               bool _init;
               unsigned int _helped;
               unsigned int _fetch;
               WDDeque      _locaQueue;

               ThreadData () : _cacheId(0), _init(false), _helped(0), _fetch( 0 )  {}
               virtual ~ThreadData () {
               }
            };

            /* disable copy and assigment */
            explicit CacheSchedPolicy ( const CacheSchedPolicy & );
            const CacheSchedPolicy & operator= ( const CacheSchedPolicy & );


            enum DecisionType { /* 0 */ NOCONSTRAINT,
                                /* 1 */ NOCOPY,
                                /* 2 */ SICOPYSIMASTER,
                                /* 3 */ SICOPYNOMASTER,
                                /* 4 */ SICOPY,
                                /* 5 */ SICOPYSIMASTERINIT,
                                /* 6 */ SICOPYNOMASTERINIT,
                                /* 7 */ SICOPYNOMASTERINIT_SELF,
                                /* 8 */ ALREADYINIT };

         public:
            static bool _noSteal;
            static bool _noMaster;
            static bool _noSupport;
            static bool _noInvalAware;
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

            void rankWD( BaseThread *thread, WD &wd );
            void tryGetLocationData( BaseThread *thread ); 
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

               if ( wd.getDepth() > 1 ) {
            //message("enqueue because of depth > 1 at queue 0 (this node) " << sys.getNetwork()->getNodeNum()<< " wd os " << wd.getId());
                  tdata._readyQueues[0].push_front( &wd );
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

                  if ( wo_copies + ro_copies == wd.getNumCopies() ) /* init task */
                  {
                     //unsigned int numCaches = sys.getCacheMap().getSize();
                     //message("numcaches is " << numCaches);
                     if ( tdata._numNodes > 1 ) {
                        //int winner = numCaches - 1;
                        int winner = tdata._numNodes - 1;
                        int start = ( _noMaster ) ? 1 : 0 ;
                        //for ( int i = winner - 1; i >= start; i -= 1 )
                        for ( int i = winner - 1; i >= start; i -= 1 )
                        {
                           winner = ( tdata._createdData[ winner ] < tdata._createdData[ i ] ) ? winner : i ;
                        }
                        tdata._createdData[ winner ] += createdDataSize;
                        tdata._bufferQueues[ winner ].push_back( &wd );
                        //if (sys.getNetwork()->getNodeNum() == 0) { message("init: queue " << (winner) << " for wd " << wd.getId() ); }
                     } else {
                        //if (sys.getNetwork()->getNodeNum() == 0) { message("global queue for wd " << wd.getId() ); }
                        tdata._globalReadyQueue.push_back( &wd );
                     }
                     //tdata._readyQueues[winner + 1].push_back( &wd );
                     tdata._holdTasks = true;
                 //    std::cerr << "END case, regular init wd " << wd.getId() << std::endl;
                  }
                  else
                  {
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
                     
                     bool locationDataIsAvailable = true;
                        for ( unsigned int i = 0; i < wd.getNumCopies() && locationDataIsAvailable; i++ ) {
                           locationDataIsAvailable = ( wd._mcontrol._memCacheCopies[ i ]._locationDataReady );
                        }

                     if ( locationDataIsAvailable ) {
                     //std::cerr <<"all data is available, ranking... wd "<< wd.getId() << std::endl;
                        rankWD(thread, wd);
                        
                   //  std::cerr <<"all data is available, ranked" << wd.getId() << std::endl;
                     } else { //no location data available, set as unranked
                     //std::cerr <<"not all data is available, pushing..." << wd.getId() <<std::endl;
                        tdata._unrankedQueue.push_back( &wd );
               //      std::cerr <<"not all data is available, pushed" << wd.getId() << std::endl;
                     }
                     
             //        std::cerr << "END case, regular wd " << wd.getId() << std::endl;
                  }
               } else {
                  if ( tdata._numNodes > 1  && _noMaster && sys.getNetwork()->getNodeNum() == 0 ) {
                     tdata._readyQueues[ 1 ].push_front( &wd );
                     
                  } else {
                     tdata._globalReadyQueue.push_front ( &wd );
                  }
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
//               return NULL;
               WD * found = current.getImmediateSuccessor(*thread);
               if ( found ) {
                  found->_mcontrol.preInit();
               }
         
               return found != NULL ? found : atIdle(thread);
            }
         
            WD * atBeforeExit ( BaseThread *thread, WD &current )
            {
//               return NULL;
               WD * found = current.getImmediateSuccessor(*thread);
               if ( found ) {
                  found->_mcontrol.preInit();
               }
               return found;
            }

            WD *fetchWD ( BaseThread *thread, WD *current );  
            virtual void atSupport ( BaseThread *thread );
            void pickWDtoInitialize ( BaseThread *thread );  

      };

#if 1
      inline void CacheSchedPolicy::pickWDtoInitialize( BaseThread *thread )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         if ( !data._init ) {
            //data._cacheId = thread->runningOn()->getMemorySpaceId();
            data._cacheId = thread->runningOn()->getMyNodeNumber();
            data._init = true;
         }
         if ( data._helped >= 16 ) return;
         NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         if ( thread->getId() >= sys.getNumPEs() )  {  // CLUSTER THREAD (master or slave)
            //ERROR
            std::cerr << "Error at " << __FUNCTION__ << std::endl;
         } else { // SMP Thread 
            if ( data._locaQueue.size() < (unsigned int) 1 ) 
            { //first try to schedule a task of my queue
               if ( ( wd = tdata._readyQueues[ 0 ].popFrontWithConstraints< And< WouldNotTriggerInvalidation, SiCopyNoMasterInit > > ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTERINIT_SELF;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)

                  wd->_mcontrol.initialize( *(thread->runningOn()) );
                  bool result;
                  do {
                     result = wd->_mcontrol.allocateInputMemory();
                  } while( result == false );
                  wd->init();

                  data._locaQueue.push_back( wd );
                  data._helped++;
                  return;
               }
            }


            // then attempt a task of a remote node
            unsigned int selectedNode = tdata._lastNodeScheduled;
            selectedNode = (selectedNode == 1) ? tdata._numNodes - 1 : selectedNode - 1;

            tdata._lastNodeScheduled = selectedNode;
            
            BaseThread const *actualThread = sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE().getFirstThread();
            BaseThread *actualThreadNC = sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE().getFirstThread();

            if ( sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getNodeNumber() != selectedNode ) {
               std::cerr <<"ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" <<std::endl;
            }
            
            ext::ClusterThread *actualClusterThread = dynamic_cast< ext::ClusterThread * >( actualThreadNC );

            if ( actualClusterThread->acceptsWDsSMP() ) {
               if ( actualClusterThread->tryLock() ) {

                  if ( data._fetch < 1 ) {
                     if ( ( wd = tdata._readyQueues[selectedNode].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopySiMasterInit > > ( actualThread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTERINIT ;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        wd->initWithPE( sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE() );
                        tdata._readyQueuesAlreadyInit[selectedNode].push_back( wd );

                        data._helped++;
                        data._fetch++;
                        actualClusterThread->unlock();
                        return;
                     }
                  }

                  if ( ( wd = tdata._readyQueuesAlreadyInit[selectedNode].pop_front( actualThreadNC ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTERINIT;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     actualClusterThread->addRunningWDSMP( wd );
                     Scheduler::preOutlineWorkWithThread( actualClusterThread, wd );
                     actualClusterThread->outlineWorkDependent(*wd);

                     data._helped++;
                     actualClusterThread->unlock();
                     return;
                  }

                  if ( ( wd = tdata._readyQueues[selectedNode].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopyNoMasterInit > > ( actualThread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTERINIT;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     wd->initWithPE( sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE() );

                     //std::cerr << "add running wd "<<std::endl;
                     actualClusterThread->addRunningWDSMP( wd );
                     //std::cerr << "ore outline with thd "<<std::endl;
                     Scheduler::preOutlineWorkWithThread( actualClusterThread, wd );
                     //std::cerr << "start wd at "<< selectedNode <<std::endl;
                     actualClusterThread->outlineWorkDependent(*wd);
                     //std::cerr << "done start wd at "<< selectedNode <<std::endl;

                     data._helped++;
                     actualClusterThread->unlock();
                     return;
                  }
                  actualClusterThread->unlock();
               }
            }
         }
      }
#endif

      inline WD *CacheSchedPolicy::fetchWD( BaseThread *thread, WD *current )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         if ( !data._init ) {
            //data._cacheId = thread->runningOn()->getMemorySpaceId();
            data._cacheId = thread->runningOn()->getMyNodeNumber();
            data._init = true;
         }
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         if ( data._cacheId != 0 ) {
            tdata._lastNodeScheduled = data._cacheId;
         }
         NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
         data._helped = 0;
         data._fetch = 0;
         if ( thread->getId() >= sys.getNumPEs() ) {  // CLUSTER THREAD (master or slave)
            if ( ( wd = tdata._readyQueuesAlreadyInit[data._cacheId].pop_front( thread ) ) != NULL ) {
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< NoCopy > ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPY;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            } 
            if ( !_noInvalAware ) {
               if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopyNoMaster > > ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
               }
               if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopySiMaster > > ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTER;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
               }
               if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, Not< NoCopy > > > ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
               }
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< SiCopyNoMaster >( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< SiCopySiMaster >( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTER;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< Not< NoCopy > >( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].pop_front( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            wd = tdata._globalReadyQueue.pop_front( thread );
            //if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, SiCopyNoMaster> >( thread ) ) != NULL ) {
            //   NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
            //   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
            //   return wd;
            //}
            //if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, SiCopySiMaster > >( thread ) ) != NULL ) {
            //   NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTER;)
            //   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
            //   return wd;
            //}
            //if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, Not< NoCopy > > >( thread ) ) != NULL ) {
            //   NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
            //   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
            //   return wd;
            //}
            //if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< WouldNotRunOutOfMemory >( thread ) ) != NULL ) {
            //   NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
            //   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
            //   return wd;
            //}
            //wd = tdata._globalReadyQueue.popFrontWithConstraints< WouldNotRunOutOfMemory >( thread );
         } else { // SMP Thread 
            if ( ( wd = data._locaQueue.pop_front( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = ALREADYINIT;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            } 
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<NoCopy> ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPY;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               { 
                  WD *helpWD;
                  if ( !_noInvalAware ) {
                     if ( ( helpWD = tdata._readyQueues[ 0 ].popFrontWithConstraints< And< WouldNotTriggerInvalidation, SiCopyNoMasterInit > > ( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val2 = SICOPYNOMASTERINIT_SELF;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val2 );)

                        helpWD->_mcontrol.initialize( *(thread->runningOn()) );
                        bool result;
                        do {
                           result = helpWD->_mcontrol.allocateInputMemory();
                        } while( result == false );
                        helpWD->init(); //WithPE( myThread->runningOn() );

                        data._locaQueue.push_back( helpWD );
                        data._helped++;
                     }
                  } else {
                     if ( ( helpWD = tdata._readyQueues[ 0 ].popFrontWithConstraints< SiCopyNoMasterInit > ( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val2 = SICOPYNOMASTERINIT_SELF;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val2 );)

                        helpWD->_mcontrol.initialize( *(thread->runningOn()) );
                        bool result;
                        do {
                           result = helpWD->_mcontrol.allocateInputMemory();
                        } while( result == false );
                        helpWD->init(); //WithPE( myThread->runningOn() );

                        data._locaQueue.push_back( helpWD );
                        data._helped++;
                     }
                  }
               }
               return wd;
            } 
            if ( !_noInvalAware ) {
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And< WouldNotTriggerInvalidation, SiCopyNoMaster > >( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< And< WouldNotTriggerInvalidation, Not< NoCopy > > >( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<SiCopyNoMaster> ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints< Not< NoCopy > > ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].pop_front( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               return wd;
            }
            wd = tdata._globalReadyQueue.pop_front ( thread );
         }
         if ( wd == NULL && thread->getId() < sys.getNumPEs() ) {
            atSupport( thread );
         }
         return wd;
      }

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
               if ( tdata._numNodes == 1 ) {
                  wd = tdata._globalReadyQueue.pop_front( thread );
                  return wd;
               }
         if ( thread->getId() == 0 ) {
            while ( !tdata._unrankedQueue.empty() ) {
               tryGetLocationData( thread );
            }
         }

         wd = fetchWD( thread, current ) ;

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
         } /*else {
            if ( !wd->resourceCheck( *thread, false ) ) {
               std::cerr << "Running wd " << wd->getId() << " will trigger an invalidation."<< std::endl;
            }
         }*/

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

               if ( tdata._numNodes == 1 ) {
                  wd = tdata._globalReadyQueue.pop_front( thread );
                  return wd;
               }

         if ( thread->getId() == 0 ) {
            while ( !tdata._unrankedQueue.empty() ) {
               tryGetLocationData( thread );
            }
         }
         //tryGetLocationData( thread );
         /*
          *  First try to schedule the thread with a task from its queue
          */
         //if ( ( wd = tdata._readyQueues[data._cacheId].pop_front ( thread ) ) != NULL ) {
         //if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<NetworkSched> ( thread ) ) != NULL ) {
         //   if (sys.getNetwork()->getNodeNum() == 0) std::cerr << "wd got by network constraint " << std::endl;
         //   return wd;
         //} else

         //unsigned int sum = tdata._feedingVector[ data._cacheId ];
         //if ( sum % 5 == 0 ) {
         //
#if 0
         if ( thread->getId() >= sys.getNumPEs() )  {  // CLUSTER THREAD (master or slave)
            if ( dynamic_cast<ext::ClusterThread*>( thread )->numRunningWDsSMP() <= ((unsigned int)sys.getNumPEs()) ) {
               if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<NoCopy> ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) NOCOPY );)
                  //tdata._feedingVector[ data._cacheId ]++;
                  return wd;
               } 
            } 
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<SiCopySiMaster> ( thread ) ) != NULL ) {
                  //if (sys.getNetwork()->getNodeNum() == 0) std::cerr << "atIdle: SiCopySiMaster "<< wd->getId() << std::endl;
   NANOS_   INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
   NANOS_   INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) SICOPYSIMASTER );)
               //if (sys.getNetwork()->getNodeNum() == 0) std::cerr << data._cacheId << ": wd got by si copy constraint " << std::endl;
               //   tdata._feedingVector[ data._cacheId ]++;
              //message("Ive got a wd, Im at node " << data._cacheId );
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<SiCopyNoMaster> ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) SICOPYNOMASTER );)
               //   tdata._feedingVector[ data._cacheId ]++;
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].popFrontWithConstraints<SiCopy> ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) SICOPY );)
               //   tdata._feedingVector[ data._cacheId ]++;
               return wd;
            }
            if ( ( wd = tdata._readyQueues[data._cacheId].pop_front( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) NOCONSTRAINT );)
               return wd;
            } else {
               /*
                * Then try to get it from the global queue
                */
                wd = tdata._globalReadyQueue.pop_front ( thread );
            }
         } else {
         }
#endif
         wd = fetchWD( thread, NULL );




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
         } /*else {
            if ( !wd->resourceCheck( *thread, false ) ) {
               std::cerr << "Running wd " << wd->getId() << " will trigger an invalidation."<< std::endl;
            }
         }*/
         return wd;
      }

      void CacheSchedPolicy::tryGetLocationData( BaseThread *thread ) {
         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         if ( !data._init ) {
            //data._cacheId = thread->runningOn()->getMemorySpaceId();
            data._cacheId = thread->runningOn()->getMyNodeNumber();
            data._init = true;
         }
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         if ( !tdata._unrankedQueue.empty() ) {
            WD *wd = tdata._unrankedQueue.pop_front( thread );
           
            if ( wd != NULL ) {
               //bool succeeded = true;
               for ( unsigned int i = 0; i < wd->getNumCopies(); i++ ) {
                  //if ( wd->_mcontrol.getCacheCopies()[ i ]._reg.id == 0 ) {
                  if ( !wd->_mcontrol._memCacheCopies[ i ]._locationDataReady ) {
                //     std::cerr << "trygetLoc at "<< __FUNCTION__<<std::endl;
                     //succeeded = succeeded && wd->_mcontrol.getCacheCopies()[ i ].tryGetLocation( *wd, i );
                     wd->_mcontrol._memCacheCopies[ i ].getVersionInfo();
                  }
               }
               //if ( succeeded ) {
              //    std::cerr << "got a wd delayed using "<< __FUNCTION__<<std::endl;
                  rankWD( thread, *wd );
               //} else {
               //   std::cerr << "readd a wd delayed using "<< __FUNCTION__<<std::endl;
               //   tdata._unrankedQueue.push_front( wd );
               //}
            }
         }
      }

      void CacheSchedPolicy::rankWD( BaseThread *thread, WD &wd ) {
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         CopyData * copies = wd.getCopies();
         unsigned int ranks[ tdata._numNodes ];
         for (unsigned int i = 0; i < tdata._numNodes; i++ ) {
            ranks[i] = 0;
         }
         //std::cerr << "RANKING WD " << wd.getId() << " numCopies " << wd.getNumCopies() << std::endl;
         for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
            if ( !copies[i].isPrivate() && copies[i].isInput() && copies[i].isOutput() ) {
                  //NewLocationInfoList const &locs = wd._mcontrol.getCacheCopies()[ i ].getNewLocations();
                  NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                  if ( locs.empty() ) {
                     //std::cerr << "empty list, version "<<  wd._mcontrol._memCacheCopies[ i ]._version << std::endl;
                     int loc = wd._mcontrol._memCacheCopies[ i ]._reg.getFirstLocation();
                     ranks[ ( loc != 0 ? sys.getSeparateMemory( loc ).getNodeNumber() : 0 ) ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                  } else {
                     for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                        int loc = ( NewNewRegionDirectory::hasWriteLocation( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first ) ) ? NewNewRegionDirectory::getWriteLocation( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first )  : NewNewRegionDirectory::getFirstLocation( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first );
                        if ( NewNewRegionDirectory::hasWriteLocation( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first ) ) {
                           //std::cerr << " wd " << wd.getId() << " has write loc " << NewNewRegionDirectory::getWriteLocation( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first ) << " locToNode-> " <<  ( loc != 0 ? sys.getSeparateMemory( loc ).getNodeNumber() : 0 ) << std::endl;
                        } else {
                           //std::cerr << " wd " << wd.getId() << " DOES NOT have write loc " << NewNewRegionDirectory::getFirstLocation( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first ) << " locToNode-> " <<  ( loc != 0 ? sys.getSeparateMemory( loc ).getNodeNumber() : 0 ) << std::endl;
                        }
                        ranks[ ( loc != 0 ? sys.getSeparateMemory( loc ).getNodeNumber() : 0 ) ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                     }
                  }
            } //else { std::cerr << "ignored copy "<< std::endl; }
         }
         //if (wd.getId() > 55 ) { tdata._readyQueues[ 0 ].push_back( &wd ); return; }
         int winner = -1;
         unsigned int start = ( _noMaster ) ? 1 : 0 ;
         unsigned int maxRank = 0;
         for ( int i = start; i < ( (int) tdata._numNodes ); i++ ) {
            if ( ranks[i] > maxRank ) {
               winner = i;
               maxRank = ranks[i];
            }
         }
         if ( winner == -1 )
            winner = start;
         unsigned int usage[ tdata._numNodes ];
         unsigned int ties=0;
         for ( int i = start; i < ( (int) tdata._numNodes ); i++ ) {
         //std::cerr << "winner is "<< winner << " ties "<< ties << " " << maxRank<< " this rank "<< ranks[i] << std::endl;
            if ( ranks[i] == maxRank ) {
               usage[ ties ] = i;
               ties += 1;
            }
         }
         //std::cerr << "winner is "<< winner << " ties "<< ties << " " << maxRank<< std::endl;
            if ( ties > 1 ) {
           //    std::cerr << "I have to chose between :";
               //for ( unsigned int ii = 0; ii < ties; ii += 1 ) fprintf(stderr, " %d", usage[ ii ] );
               //std::cerr << std::endl;
               unsigned int minLoad = usage[0];
               for ( unsigned int ii = 1; ii < ties; ii += 1 ) {
             //     std::cerr << "load of (min) " << minLoad << " is " << tdata._load[ minLoad ] <<std::endl;
               //   std::cerr << "load of (itr) " << usage[ ii ]  << " is " << tdata._load[ usage[ ii ] ] << std::endl;
                  if ( tdata._load[ usage[ ii ] ] < tdata._load[ minLoad ] ) {
                     minLoad = usage[ ii ];
                  }
               }
               //std::cerr << "Well winner is gonna be "<< minLoad << std::endl;
               tdata._load[ minLoad ]++;
               winner = minLoad;
            }
            //if (sys.getNetwork()->getNodeNum() == 0 ) { 
            //   std::cerr << "WD: " << wd.getId() << " ROcopies: "<<ro_copies << " WOcopies: " << wo_copies << " RWcopies: " << rw_copies << " Locality results: [ ";
            //   for (unsigned int i = 0; i < tdata._numNodes ; i += 1) std::cerr << i << ": " << (ranks[i] / (16*512*512)) << " "; 
            //   std::cerr <<"] ties " << ties << " winner " << winner << std::endl;
            //}
         //if (winner == -1) winner = start;
         //message("queued wd " << wd.getId() << " to queue " << winner << " ranks " << ranks[0] << "," << ranks[1] << "," << ranks[2] << "," << ranks[3] );
         //fprintf(stderr, "queued wd %d to queue %d ranks %x %x %x %x \n", wd.getId(), winner, ranks[0], ranks[1], ranks[2], ranks[3] );
         //std::cerr << "the winner is " << winner << std::endl;
         tdata._readyQueues[winner].push_back( &wd );
      }

      void CacheSchedPolicy::atSupport ( BaseThread *thread ) {
         //tryGetLocationData( thread );
         if ( !_noSupport ) {
            pickWDtoInitialize( thread );
         }
      }

      bool CacheSchedPolicy::_noSteal = false;
      bool CacheSchedPolicy::_noMaster = false;
      bool CacheSchedPolicy::_noSupport = false;
      bool CacheSchedPolicy::_noInvalAware = false;

      class CacheSchedPlugin : public Plugin
      {
         public:
            CacheSchedPlugin() : Plugin( "Cache-guided scheduling Plugin",1 ) {}

            virtual void config( Config& cfg )
            {
               cfg.setOptionsSection( "Affinity module", "Data Affinity scheduling module" );
               cfg.registerConfigOption ( "affinity-no-steal", NEW Config::FlagOption( CacheSchedPolicy::_noSteal ), "Steal tasks from other threads");
               cfg.registerArgOption( "affinity-no-steal", "affinity-no-steal" );

               cfg.registerConfigOption ( "affinity-no-master", NEW Config::FlagOption( CacheSchedPolicy::_noMaster ), "Do not execute tasks on master node");
               cfg.registerArgOption( "affinity-no-master", "affinity-no-master" );

               cfg.registerConfigOption ( "affinity-no-support", NEW Config::FlagOption( CacheSchedPolicy::_noSupport ), "Do not execute tasks on master node");
               cfg.registerArgOption( "affinity-no-support", "affinity-no-support" );

               cfg.registerConfigOption ( "affinity-no-inval-aware", NEW Config::FlagOption( CacheSchedPolicy::_noInvalAware ), "Do not execute tasks on master node");
               cfg.registerArgOption( "affinity-no-inval-aware", "affinity-no-inval-aware" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW CacheSchedPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-affinity",nanos::ext::CacheSchedPlugin);
