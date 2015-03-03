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
#include "os.hpp"
#include "memtracker.hpp"
#include "clusterthread_decl.hpp"
#include "regioncache.hpp"
#include "newregiondirectory.hpp"
#include "smpdd.hpp"

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
               unsigned int       _numLocalAccelerators;
               unsigned int       _numQueues;
               //Network sched std::set<int>      _freeNodes;
               //Network sched std::set<int>      _nodeSet;
               Lock               _freeNodesLock;
               Atomic<int>       *_feeding;
               int               *_feedingVector;
               unsigned int      *_load;
               Atomic<unsigned int> _getting;
               unsigned int       _lastNodeScheduled;
               std::vector< memory_space_id_t > *_nodeToMemSpace;
               std::vector< memory_space_id_t > *_queueToMemSpace;
 
               TeamData ( unsigned int size ) : ScheduleTeamData(), _globalReadyQueue(), _unrankedQueue()
               {
                  _numNodes = sys.getNumClusterNodes();
                  _numLocalAccelerators = sys.getNumAccelerators();
                  _getting = 0;
                  _numQueues = _numNodes + _numLocalAccelerators;

                  _queueToMemSpace = NEW std::vector< memory_space_id_t >( _numQueues );
                  unsigned int queue_counter = 0;

                  _holdTasks = false;
                  if ( _numQueues > 1 ) {
                     _readyQueues = NEW WDDeque[_numQueues];
                     _readyQueuesAlreadyInit = NEW WDDeque[_numQueues];
                     _bufferQueues = NEW WDDeque[_numQueues];
                     _createdData = NEW std::size_t[_numQueues];
                     for (unsigned int i = 0; i < _numQueues; i += 1 ) {
                        _createdData[i] = 0;
                     }
                     _nodeToMemSpace = NEW std::vector< memory_space_id_t >( sys.getClusterNodeSet().size() );

                     for( std::set<unsigned int>::iterator it = sys.getClusterNodeSet().begin();
                           it != sys.getClusterNodeSet().end();
                           it++ ) {
                        memory_space_id_t node_memspace = sys.getMemorySpaceIdOfClusterNode( *it );
                        (*_nodeToMemSpace)[ *it ] = node_memspace;
                        (*_queueToMemSpace)[ queue_counter ] = node_memspace;
                        queue_counter += 1;
                     }

                     for ( unsigned int accel_id = 0; accel_id < _numLocalAccelerators; accel_id += 1) {
                        memory_space_id_t accel_memspace = sys.getMemorySpaceIdOfAccelerator( accel_id );
                        if ( accel_memspace != (memory_space_id_t) -1 ) {
                           (*_queueToMemSpace)[ queue_counter ] = accel_memspace;
                           queue_counter += 1;
                        } else {
                           fatal("Unable to properly initialize the scheduler.");
                        }
                     }

                     // (*myThread->_file) << "_queueToMemSpace [ ";
                     // for ( std::vector<memory_space_id_t>::iterator it = _queueToMemSpace->begin(); it != _queueToMemSpace->end(); it++) {
                     //    (*myThread->_file) << *it << " ";
                     // }
                     // (*myThread->_file) << "]" << std::endl;
                  }
                  _load = NEW unsigned int[_numQueues];
                  _feeding = NEW Atomic<int>[_numQueues];
                  _feedingVector = NEW int[ _numQueues * 2 ];
                  for (unsigned int i = 0; i < _numQueues; i += 1) {
                     _feeding[ i ] = 0;
                     _feedingVector[ i * 2 ] = 0;
                     _feedingVector[ i * 2 + 1 ] = 0;
                     _load[ i ] = 0;
                  }
                  //_freeNodes.insert(_nodeSet.begin(), _nodeSet.end() );
                  _lastNodeScheduled = 1;
             
               }

               ~TeamData ()
               {
                  if (_numQueues > 1 ) {
                     delete[] _readyQueues;
                     delete[] _readyQueuesAlreadyInit;
                     delete[] _bufferQueues;
                     delete[] _createdData;
                  }
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
                  if ( wd.initialized() && wd._mcontrol.isDataReady( wd ) ) return true;
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
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn() ) && NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, (memory_space_id_t) 0) ) {
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
                                 && NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, (memory_space_id_t) 0 )  ) {
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
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn() ) && ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, (memory_space_id_t) 0 ) ) {
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
                                 if ( ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn() ) && ! NewNewRegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, (memory_space_id_t) 0 ) ) {
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
                  
                  //while( tdata._feeding[ queueId ].cswap( 1, 0 ) != 1 );
                  tdata._freeNodesLock.acquire();
                  //tdata._freeNodes.insert( queueId );
                  tdata._feedingVector[ srcNode ]
                  tdata._freeNodesLock.release();
                  //       (*myThread->_file) <<"cleared copy by wd " << wd.getId() << " id is " << queueId  <<std::endl;
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
                  if ( tmpFeedingVector[ queueId * 2 ] + 1 > 2 ) {
                  tdata._freeNodesLock.release();
                                 return false;
                  }

                  if (sys.getNetwork()->getNodeNum() == 0)(*myThread->_file) <<"check wd " << wd.getId() << std::endl;
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
                              //   if (sys.getNetwork()->getNodeNum() == 0)(*myThread->_file) <<"loc busy... " << loc << std::endl;
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
                     NotifyData *ndata = NEW ( allLocs - gotLocs, queueId );
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
                     //            tmpFeedingVector[ queueId * 2 ] += 1;
                     //            tmpFeedingVector[ loc * 2 + 1 ] += 1;
                     //         }
                     //      }
                     //   }
                     //}
                     // XXX commit_changes_to_FeedingVector();
                     wd.setNotifyCopyFunc( notifyCopy, (void *) NEW NotifyData( queueId, loc )  );
                     for ( unsigned int i = 0; i < tdata._numNodes; i += 1 ) {
                        tdata._feedingVector[ i * 2 ] = tmpFeedingVector[ i * 2 ];
                        tdata._feedingVector[ i * 2 + 1 ] = tmpFeedingVector[ i * 2 + 1 ];
                     }
                  tdata._freeNodesLock.release();
                     return true;
                  }
                  //if ( !toBeRemovedElems.empty( ) ) {
                  //   //(*myThread->_file) << "feed val is " << tdata._feedingThisNode.value() << std::endl;
                  //   if ( tdata._feeding[ queueId ].cswap( 0, 1 ) == 1 ) {
                  //      if (sys.getNetwork()->getNodeNum() == 0)(*myThread->_file) <<"allow copy to wd " << wd.getId() << " dest " << queueId << std::endl;
                  //      wd.setNotifyCopyFunc( notifyCopy );
                  //   } else {
                  // if (sys.getNetwork()->getNodeNum() == 0)(*myThread->_file) <<"stolen... " << std::endl;
                  //      tdata._freeNodesLock.release();
                  //      return false;
                  //   }
                  //   
                  //   //(*myThread->_file) <<"thread " << thread.getId() <<" node "<< queueId <<" selected WD " << wd.getId() << " sources of data are ";
                  //   //for (std::set<int>::iterator it = toBeRemovedElems.begin(); it != toBeRemovedElems.end(); it++ ) {
                  //   //   (*myThread->_file) << *it << " ";
                  //   //}
                  //   //(*myThread->_file) << std::endl;
                  //   for (std::set<int>::iterator it = toBeRemovedElems.begin(); it != toBeRemovedElems.end(); it++ ) {
                  //      tdata._freeNodes.erase( *it );
                  //   }
                  //   //(*myThread->_file) << " remaining set of free nodes ";
                  //   //for (std::set<int>::iterator it = tdata._freeNodes.begin(); it != tdata._freeNodes.end(); it++ ) {
                  //   //   (*myThread->_file) << *it << " ";
                  //   //}
                  //   //(*myThread->_file) << std::endl;
                  //   //if ( tdata._freeNodes.empty() ) {
                  //   //   tdata._freeNodes.insert( tdata._nodeSet.begin(), tdata._nodeSet.end() );
                  //   //}
                  //} else { 
                  //   if (sys.getNetwork()->getNodeNum() == 0)(*myThread->_file) <<"thread " << thread.getId() <<" selected WD " << wd.getId() << " empty sources, allLocs  " << allLocs << " gotLocs "<< gotLocs <<std::endl;
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
               //(*myThread->_file) << " CLEAR GETTING " << std::endl;
               //while ( !tdata._getting.cswap( true, false ) ) { }
               tdata._getting--;
               
            }
            /** \brief Cache Scheduler data associated to each thread
              *
              */
            struct ThreadData : public ScheduleThreadData
            {
               /*! queue of ready tasks to be executed */
               unsigned int _nodeId;
               bool _isLocalAccelerator;
               unsigned int _acceleratorId;
               bool _init;
               unsigned int _helped;
               unsigned int _fetch;
               WDDeque      _locaQueue;

               ThreadData () : _nodeId(0), _isLocalAccelerator( false ), _acceleratorId(0), _init(false), _helped(0), _fetch( 0 )  {}
               virtual ~ThreadData () {
               }

               void initialize( BaseThread *thd ) {
                  if ( !_init ) {
                     memory_space_id_t mem_id = thd->runningOn()->getMemorySpaceId();
                     _nodeId =  mem_id == 0 ? 0 : sys.getSeparateMemory( mem_id ).getNodeNumber();
                     _isLocalAccelerator = (_nodeId == 0 && mem_id != 0);
                     _acceleratorId = mem_id == 0 ? 0 : sys.getSeparateMemory( mem_id ).getAcceleratorNumber();
                     _init = true;
                  }
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
            static bool _steal;
            static bool _noMaster;
#ifdef CLUSTER_DEV
            static bool _support;
#endif
            static bool _invalAware;
            static bool _affinityInout;
            static bool _constraints;
            static bool _immediateSuccessor;
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

            int computeAffinityScore( WD &wd, unsigned int numQueues, unsigned int numNodes, std::size_t *scores, std::size_t &maxPossibleScore );
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
               //(*myThread->_file) << " queue wd " << wd.getId() << std::endl;
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t deb = ID->getEventKey("debug"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( deb, (nanos_event_value_t) 2 ); )
#if 1
               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               data.initialize( thread );
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
              
               if ( tdata._numQueues == 1 ) {
                  //(*myThread->_file) << " 1 queue, goto global" << std::endl;
                  tdata._globalReadyQueue.push_front( &wd );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( deb, 0 ); )
                  return;
               }

               if ( wd.getDepth() > 1 ) {
            //message("enqueue because of depth > 1 at queue 0 (this node) " << sys.getNetwork()->getNodeNum()<< " wd os " << wd.getId());
                  tdata._readyQueues[0].push_front( &wd );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( deb, 0 ); )
                  return;
               }
            //message("in queue node " << sys.getNetwork()->getNodeNum()<< " wd os " << wd.getId());
               if ( wd.isTied() ) {
                  unsigned int index = wd.isTiedTo()->runningOn()->getClusterNode();
                  tdata._readyQueues[index].push_front ( &wd );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( deb, 0 ); )
                  return;
               }
               memory_space_id_t rootedLocation;
               bool rooted = wd._mcontrol.isRooted( rootedLocation );

               if ( rooted ) { //it has to be executed on a given node
                  unsigned int tied_node = rootedLocation != 0 ? sys.getSeparateMemory( wd.isTiedToLocation() ).getNodeNumber() : 0; 
                  //FIXME take into account local accelerators
                  if ( tied_node == 0 ) {
                     bool locationDataIsAvailable = true;
                     for ( unsigned int i = 0; i < wd.getNumCopies() && locationDataIsAvailable; i++ ) {
                        locationDataIsAvailable = ( wd._mcontrol._memCacheCopies[ i ]._locationDataReady );
                     }

                     if ( locationDataIsAvailable ) {
                        //(*myThread->_file) <<"all data is available, ranking... wd "<< wd.getId() << std::endl;
                        rankWD(thread, wd);

                        //  (*myThread->_file) <<"all data is available, ranked" << wd.getId() << std::endl;
                     } else { //no location data available, set as unranked
                        (*myThread->_file) <<"not all data is available, pushing..." << wd.getId() <<std::endl;
                        tdata._unrankedQueue.push_back( &wd );
                        //      (*myThread->_file) <<"not all data is available, pushed" << wd.getId() << std::endl;
                     }
                  } else {
                     tdata._readyQueues[ tied_node ].push_back( &wd );
                  }
               } else {

               if ( wd.getNumCopies() > 0 ) {
                  CopyData * copies = wd.getCopies();
                  unsigned int wo_copies = 0, ro_copies = 0, rw_copies = 0, new_data_copies = 0;
                  std::size_t createdDataSize = 0;
                  for (unsigned int idx = 0; idx < wd.getNumCopies(); idx += 1)
                  {
                     if ( !copies[idx].isPrivate() ) {
                        new_data_copies += ( wd._mcontrol._memCacheCopies[ idx ].getVersion() == 1 && copies[idx].isOutput() );
                        createdDataSize += ( wd._mcontrol._memCacheCopies[ idx ].getVersion() == 1 && copies[idx].isOutput() ) * copies[idx].getSize();
                        //createdDataSize += ( !copies[idx].isInput() && copies[idx].isOutput() ) * copies[idx].getSize();
                        rw_copies += (  copies[idx].isInput() &&  copies[idx].isOutput() );
                        ro_copies += (  copies[idx].isInput() && !copies[idx].isOutput() );
                        wo_copies += ( !copies[idx].isInput() &&  copies[idx].isOutput() );
                        createdDataSize += ( !copies[idx].isInput() && copies[idx].isOutput() ) * copies[idx].getSize();
                     }
                  }

                  //(*myThread->_file) << " wtfffffffffffffffff " << wd.getId() << " ndc " << new_data_copies << std::endl;
                  //if ( wo_copies + ro_copies == wd.getNumCopies() ) /* init task */
                  if ( new_data_copies ) /* init task: distribute only among nodes, because we can't  */
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
//          *thread->_file << "Ranked init wd (id=" << wd.getId() << ", desc=" <<
//             (wd.getDescription() != NULL ? wd.getDescription() : "n/a") <<
//             ") to queue " << winner << std::endl;
                        //tdata._bufferQueues[ winner ].push_back( &wd );

                        //if (sys.getNetwork()->getNodeNum() == 0) { message("init: queue " << (winner) << " for wd " << wd.getId() ); }

                        if ( winner == 0 && tdata._numLocalAccelerators > 0 ) {
                           if ( wd.canRunIn( SMP ) ) {
                              tdata._readyQueues[ winner ].push_back( &wd );
                              //(*myThread->_file) << "[SC:aff] wd " << wd.getId() << " Multiple nodes, node 0 won / multiple acc, SMP winner is " << winner << std::endl;
                           } else {
                              for ( unsigned int queue_idx = tdata._numNodes; queue_idx < tdata._numQueues; queue_idx += 1) {
                                 if ( wd.canRunIn( sys.getSeparateMemory( (*tdata._queueToMemSpace)[ queue_idx ] ).getDevice() ) ) {
                                    if (winner == 0) {
                                       winner = queue_idx;
                                    }
                                    winner = ( tdata._createdData[ winner ] < tdata._createdData[ queue_idx ] ) ? winner : queue_idx ;
                                 }
                              }
                              tdata._createdData[ winner ] += createdDataSize;
                              //(*myThread->_file) << "[SC:aff] wd " << wd.getId() << " Multiple nodes, node 0 won / multiple acc, ACC winner is " << winner << std::endl;
                              tdata._readyQueues[ winner ].push_back( &wd );
                           }
                        } else {
                           //(*myThread->_file) << "[SC:aff] wd " << wd.getId() << " Multiple nodes, remote node won or no acc available, winner is " << winner << std::endl;
                           tdata._readyQueues[ winner ].push_back( &wd );
                        }
                     } else {
                        //if (sys.getNetwork()->getNodeNum() == 0) { message("global queue for wd " << wd.getId() ); }
                        if ( tdata._numQueues > 1 ) {
                           //Single node with accelerators
                           //wd.canRunIn( sys.getSeparateMemory( loc ).getDevice() )
                           if ( wd.canRunIn( SMP ) ) {
                              //(*myThread->_file) << "[SC:aff] wd " << wd.getId() << " Single node, acc avail, SMP winner global queue" << std::endl;
                              tdata._globalReadyQueue.push_back( &wd );
                           } else {
                              int winner = 0;
                              for ( unsigned int queue_idx = tdata._numNodes; queue_idx < tdata._numQueues; queue_idx += 1) {
                                 if ( wd.canRunIn( sys.getSeparateMemory( (*tdata._queueToMemSpace)[ queue_idx ] ).getDevice() ) ) {
                                    if (winner == 0) {
                                       winner = queue_idx;
                                    }
                                    winner = ( tdata._createdData[ winner ] < tdata._createdData[ queue_idx ] ) ? winner : queue_idx ;
                                 }
                              }
                              //(*myThread->_file) << "[SC:aff] wd " << wd.getId() << " Single node, acc avail, acc winner is " << winner << std::endl;
                              tdata._createdData[ winner ] += createdDataSize;
                              //(*myThread->_file) << "winner is " << winner << std::endl;
                              tdata._readyQueues[ winner ].push_back( &wd );
                           }
                        } else {
                           //Single node, no accelerators
                           //(*myThread->_file) << "[SC:aff] wd " << wd.getId() << " Single node, no acc avail, global queue" << std::endl;
                           tdata._globalReadyQueue.push_back( &wd );
                        }
                     }
                     //tdata._readyQueues[winner + 1].push_back( &wd );
                     tdata._holdTasks = true;
                     //(*myThread->_file) << "END case, regular init wd " << wd.getId() << std::endl;
                  }
                  else
                  {
                     //if ( tdata._holdTasks.value() )
                     //{
                     //   if ( tdata._holdTasks.cswap( true, false ) )
                     //   {
                     //      for ( unsigned int idx = 0; idx < tdata._numQueues; idx += 1) 
                     //      {
                     //         tdata._readyQueues[ idx ].transferElemsFrom( tdata._bufferQueues[ idx ] );
                     //      }
                     //   }
                     //}
                     
                     bool locationDataIsAvailable = true;
                        for ( unsigned int i = 0; i < wd.getNumCopies() && locationDataIsAvailable; i++ ) {
                           locationDataIsAvailable = ( wd._mcontrol._memCacheCopies[ i ]._locationDataReady );
                        }

                     if ( locationDataIsAvailable ) {
                     //(*myThread->_file) <<"all data is available, ranking... wd "<< wd.getId() << std::endl;
                        rankWD(thread, wd);
                        
                   //  (*myThread->_file) <<"all data is available, ranked" << wd.getId() << std::endl;
                     } else { //no location data available, set as unranked
                     (*myThread->_file) <<"!!!!!! not all data is available, pushing..." << wd.getId() <<std::endl;
                     //   tdata._unrankedQueue.push_back( &wd );
               //      (*myThread->_file) <<"not all data is available, pushed" << wd.getId() << std::endl;
                     }
                     
             //        (*myThread->_file) << "END case, regular wd " << wd.getId() << std::endl;
                  }
               } else {
                  if ( tdata._numNodes > 1  && _noMaster && sys.getNetwork()->getNodeNum() == 0 ) {
                     tdata._readyQueues[ 1 ].push_front( &wd );
                     
                  } else {
                     tdata._globalReadyQueue.push_front ( &wd );
                  }
               }

               } //not rooted wd
#endif
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( deb, 0 ); )
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
               WD * found = NULL;
               if ( _immediateSuccessor ) {
                  found = current.getImmediateSuccessor(*thread);
                  //if ( found ) {
                  //   (*myThread->_file) << " atPrefetch (getImmediateSuccessor) returns wd " << found->getId() << std::endl;
                  //}
               }
               return found != NULL ? found : atIdle(thread);
            }
         
            WD * atBeforeExit ( BaseThread *thread, WD &current, bool schedule )
            {
               WD * found = NULL;
               if ( schedule && _immediateSuccessor ) {
                  found = current.getImmediateSuccessor(*thread);
                  //if ( found ) {
                  //   (*myThread->_file) << " atBeforeExit (getImmediateSuccessor) returns wd " << found->getId() << std::endl;
                  //}
               }
               return found;
            }

            WD *fetchWD ( BaseThread *thread, WD *current );  
            virtual void atSupport ( BaseThread *thread );
#ifdef CLUSTER_DEV
            void pickWDtoInitialize ( BaseThread *thread );  
#endif

      };

#ifdef CLUSTER_DEV //disabled for now
      inline void CacheSchedPolicy::pickWDtoInitialize( BaseThread *thread )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         data.initialize( thread );
         if ( data._helped >= 16 ) return;
         NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         if ( tdata._numNodes == 1 ) {
            return;
         }
         if ( thread->runningOn()->getClusterNode() > 0 )  {  // CLUSTER THREAD (master or slave)
            //ERROR
            (*myThread->_file) << "Error at " << __FUNCTION__ << std::endl;
         } else { // Non cluster Thread 
            if ( data._locaQueue.size() < (unsigned int) 1 && _constraints ) 
            { //first try to schedule a task of my queue
               if ( ( wd = tdata._readyQueues[ 0 ].popFrontWithConstraints< And< WouldNotTriggerInvalidation, SiCopyNoMasterInit > > ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTERINIT_SELF;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)

                        //(*myThread->_file) << myThread->getId() << " helped SiCopyNoMasterInit with wd " << wd->getId() << " to self node" << std::endl;
                  wd->_mcontrol.initialize( *(thread->runningOn()) );
                  bool result;
                  do {
                     result = wd->_mcontrol.allocateTaskMemory();
                  } while( result == false );
                  wd->init();

                        //(*myThread->_file) << myThread->getId() << " helped WDONE SiCopyNoMasterInit with wd " << wd->getId() << " to self node" << std::endl;
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
               (*myThread->_file) <<"ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" <<std::endl;
            }
            
            ext::ClusterThread *actualClusterThread = dynamic_cast< ext::ClusterThread * >( actualThreadNC );

            if ( actualClusterThread->acceptsWDsSMP() ) {
               if ( actualClusterThread->tryLock() ) {

                  if ( data._fetch < 1 ) {
                     if ( ( wd = tdata._readyQueues[selectedNode].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopySiMasterInit > > ( actualThread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTERINIT ;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)


                        //(*myThread->_file) << myThread->getId() << " helped COPY FROM MASTER with wd " << wd->getId() << " to node " << selectedNode << std::endl;
                        wd->_mcontrol.initialize( sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE() );
                        bool result;
                        do {
                           result = wd->_mcontrol.allocateTaskMemory();
                        } while( result == false );

                        wd->initWithPE( sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE() );
                        //(*myThread->_file) << myThread->getId() << " helped WDONE COPY FROM MASTER with wd " << wd->getId() << " to node " << selectedNode << std::endl;
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

                     actualClusterThread->preOutlineWorkDependent( *wd );
                     //(*myThread->_file) << myThread->getId() << " helped SICOPYNOMASTERINIT ai LAUNCH with wd " << wd->getId() << " to node " << selectedNode << std::endl;
                     //actualClusterThread->runningOn()->waitInputs( *wd );
                     while( !wd->isInputDataReady() ) {
                     }
                     actualClusterThread->outlineWorkDependent(*wd);
                     //(*myThread->_file) << myThread->getId() << " helped SICOPYNOMASTERINIT ai WDONE LAUNCH with wd " << wd->getId() << " to node " << selectedNode << std::endl;

                     data._helped++;
                     actualClusterThread->unlock();
                     return;
                  }

                  if ( ( wd = tdata._readyQueues[selectedNode].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopyNoMasterInit > > ( actualThread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTERINIT;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)


                     wd->_mcontrol.initialize( sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE() );
                     bool result;
                     do {
                        result = wd->_mcontrol.allocateTaskMemory();
                     } while( result == false );
                     wd->initWithPE( sys.getSeparateMemory( (*tdata._nodeToMemSpace)[ selectedNode ] ).getPE() );

                     //(*myThread->_file) << "add running wd "<<std::endl;
                     actualClusterThread->addRunningWDSMP( wd );
                     //(*myThread->_file) << "ore outline with thd "<<std::endl;
                     Scheduler::preOutlineWorkWithThread( actualClusterThread, wd );
                     //(*myThread->_file) << "start wd at "<< selectedNode <<std::endl;
                     actualClusterThread->preOutlineWorkDependent( *wd );
                     //(*myThread->_file) << myThread->getId() << " helped SICOPYNOMASTERINIT REMOTE COPY AND LAUNCH with wd " << wd->getId() << " to node " << selectedNode << std::endl;
                     //actualClusterThread->runningOn()->waitInputs( *wd );
                     while( !wd->isInputDataReady() ) {
                     }
                     actualClusterThread->outlineWorkDependent(*wd);
                     //(*myThread->_file) << "done start wd at "<< selectedNode <<std::endl;
                     //(*myThread->_file) << myThread->getId() << " helped SICOPYNOMASTERINIT WDONE REMOTE COPY AND LAUNCH with wd " << wd->getId() << " to node " << selectedNode << std::endl;

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
         data.initialize( thread );
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         NANOS_INSTRUMENT(static nanos_event_value_t val_dbg;)

         unsigned int queueId = data._isLocalAccelerator ? tdata._numNodes + data._acceleratorId : data._nodeId;

         if ( queueId != 0 ) {
            tdata._lastNodeScheduled = queueId;
         }
         NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
         data._helped = 0;
         data._fetch = 0;


         if ( !_constraints ) {
            if ( ( wd = tdata._readyQueues[queueId].pop_front( thread ) ) != NULL ) {
               return wd;
            }
            if ( data._isLocalAccelerator ) { //check local node queue
               if ( ( wd = tdata._readyQueues[0].pop_front( thread ) ) != NULL ) {
                  return wd;
               }
            }
            wd = tdata._globalReadyQueue.pop_front ( thread );
         } else {

            if ( thread->runningOn()->getClusterNode() > 0 ) {  // CLUSTER THREAD (master or slave)
               if ( ( wd = tdata._readyQueuesAlreadyInit[queueId].pop_front( thread ) ) != NULL ) {
                  return wd;
               }
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< NoCopy > ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPY;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               } 
               if ( _invalAware ) {
                  if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopyNoMaster > > ( thread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
                  }
                  if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, SiCopySiMaster > > ( thread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTER;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
                  }
                  if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, Not< NoCopy > > > ( thread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
                  }
               }
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< SiCopyNoMaster >( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               }
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< SiCopySiMaster >( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTER;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               }
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< Not< NoCopy > >( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               }
               if ( ( wd = tdata._readyQueues[queueId].pop_front( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               }
               wd = tdata._globalReadyQueue.pop_front( thread );
               //if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, SiCopyNoMaster> >( thread ) ) != NULL ) {
               //   NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
               //   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               //   return wd;
               //}
               //if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, SiCopySiMaster > >( thread ) ) != NULL ) {
               //   NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYSIMASTER;)
               //   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               //   return wd;
               //}
               //if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, Not< NoCopy > > >( thread ) ) != NULL ) {
               //   NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
               //   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
               //   return wd;
               //}
               //if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< WouldNotRunOutOfMemory >( thread ) ) != NULL ) {
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
               NANOS_INSTRUMENT(val_dbg = 122; sys.getInstrumentation()->raisePointEvents( 1, &key, &val_dbg );)
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints<NoCopy> ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPY;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  { 
                     WD *helpWD;
                     if ( _invalAware ) {
                        if ( ( helpWD = tdata._readyQueues[ 0 ].popFrontWithConstraints< And< WouldNotTriggerInvalidation, SiCopyNoMasterInit > > ( thread ) ) != NULL ) {
                           NANOS_INSTRUMENT(static nanos_event_value_t val2 = SICOPYNOMASTERINIT_SELF;)
                           NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val2 );)

                           helpWD->_mcontrol.initialize( *(thread->runningOn()) );
                           bool result;
                           do {
                              result = helpWD->_mcontrol.allocateTaskMemory();
                           } while( result == false );
                           helpWD->init(); //WithPE( myThread->runningOn() );

                           data._locaQueue.push_back( helpWD );
                           data._helped++;
                        }
                     } else {
                        NANOS_INSTRUMENT(val_dbg = 123; sys.getInstrumentation()->raisePointEvents( 1, &key, &val_dbg );)
                        if ( ( helpWD = tdata._readyQueues[ 0 ].popFrontWithConstraints< SiCopyNoMasterInit > ( thread ) ) != NULL ) {
                           NANOS_INSTRUMENT(static nanos_event_value_t val2 = SICOPYNOMASTERINIT_SELF;)
                           NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val2 );)

                           helpWD->_mcontrol.initialize( *(thread->runningOn()) );
                           bool result;
                           do {
                              result = helpWD->_mcontrol.allocateTaskMemory();
                           } while( result == false );
                           helpWD->init(); //WithPE( myThread->runningOn() );

                           data._locaQueue.push_back( helpWD );
                           data._helped++;
                        }
                     }
                  }
                  return wd;
               } 
               if ( _invalAware ) {
                  if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And< WouldNotTriggerInvalidation, SiCopyNoMaster > >( thread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
                  }
                  if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< And< WouldNotTriggerInvalidation, Not< NoCopy > > >( thread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
                  }
               }
               NANOS_INSTRUMENT(val_dbg = 124; sys.getInstrumentation()->raisePointEvents( 1, &key, &val_dbg );)
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints<SiCopyNoMaster> ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOMASTER;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               }
               NANOS_INSTRUMENT(val_dbg = 125; sys.getInstrumentation()->raisePointEvents( 1, &key, &val_dbg );)
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints< Not< NoCopy > > ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               }
               //if ( thread->getId() > 0 ) *thread->_file << "lol smp thread " << thread->getId() << " check queue " << queueId << " is acc? " << data._isLocalAccelerator << " acc " << data._acceleratorId << " nodeId " << data._nodeId << " elems in queue "  << tdata._readyQueues[data._nodeId].size() << std::endl;
               NANOS_INSTRUMENT(val_dbg = 126; sys.getInstrumentation()->raisePointEvents( 1, &key, &val_dbg );)
               if ( ( wd = tdata._readyQueues[queueId].pop_front( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                  return wd;
               }

               if ( data._isLocalAccelerator ) { //check local node queue
                  NANOS_INSTRUMENT(val_dbg = 127; sys.getInstrumentation()->raisePointEvents( 1, &key, &val_dbg );)
                  if ( ( wd = tdata._readyQueues[0].pop_front( thread ) ) != NULL ) {
                     NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                     return wd;
                  }
               }
               NANOS_INSTRUMENT(val_dbg = 128; sys.getInstrumentation()->raisePointEvents( 1, &key, &val_dbg );)
               wd = tdata._globalReadyQueue.pop_front ( thread );
            }
         } // constraints


         if ( wd == NULL && thread->runningOn()->getClusterNode() == 0 ) {
            atSupport( thread );
         }
         return wd;
      }

      WD *CacheSchedPolicy::atBlock ( BaseThread *thread, WD *current )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         data.initialize( thread );
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         unsigned int queueId = data._isLocalAccelerator ? tdata._numNodes + data._acceleratorId : data._nodeId;

         // if ( tdata._holdTasks.value() ) 
         // {
         //    if ( tdata._holdTasks.cswap( true, false ) )
         //    {
         //       for ( unsigned int idx = 0; idx < tdata._numQueues; idx += 1) 
         //       {
         //          tdata._readyQueues[ idx ].transferElemsFrom( tdata._bufferQueues[ idx] );
         //       }
         //    }
         // }
               if ( tdata._numQueues == 1 ) {
                  wd = tdata._globalReadyQueue.pop_front( thread );
                  return wd;
               }
         if ( thread->getId() == 0 ) {
            while ( !tdata._unrankedQueue.empty() ) {
               tryGetLocationData( thread );
            }
         }

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t deb = ID->getEventKey("debug"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( deb, (nanos_event_value_t) 1 ); )
         wd = fetchWD( thread, current ) ;
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( deb, 0 ); )

         if ( _steal )
         {
            if ( wd == NULL ) {
               for ( unsigned int i = queueId + 1; i < tdata._numNodes; i++ ) {
                  if ( !tdata._readyQueues[i].empty() ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
               for ( unsigned int i = 0; i < queueId; i++ ) {
                  if ( !tdata._readyQueues[i].empty() ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
            }
         }

         if ( wd == NULL ) {
            OS::nanosleep( 100 );
         } /*else {
            if ( !wd->resourceCheck( *thread, false ) ) {
               (*myThread->_file) << "Running wd " << wd->getId() << " will trigger an invalidation."<< std::endl;
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
         data.initialize( thread );
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         unsigned int queueId = data._isLocalAccelerator ? tdata._numNodes + data._acceleratorId : data._nodeId;

         if ( tdata._numQueues == 1 ) {
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
         //if ( ( wd = tdata._readyQueues[queueId].pop_front ( thread ) ) != NULL ) {
         //if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints<NetworkSched> ( thread ) ) != NULL ) {
         //   if (sys.getNetwork()->getNodeNum() == 0) (*myThread->_file) << "wd got by network constraint " << std::endl;
         //   return wd;
         //} else

         //unsigned int sum = tdata._feedingVector[ queueId ];
         //if ( sum % 5 == 0 ) {
         //
#if 0
         if ( thread->getId() >= sys.getNumPEs() )  {  // CLUSTER THREAD (master or slave)
            if ( dynamic_cast<ext::ClusterThread*>( thread )->numRunningWDsSMP() <= ((unsigned int)sys.getNumPEs()) ) {
               if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints<NoCopy> ( thread ) ) != NULL ) {
                  NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
                  NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) NOCOPY );)
                  //tdata._feedingVector[ queueId ]++;
                  return wd;
               } 
            } 
            if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints<SiCopySiMaster> ( thread ) ) != NULL ) {
                  //if (sys.getNetwork()->getNodeNum() == 0) (*myThread->_file) << "atIdle: SiCopySiMaster "<< wd->getId() << std::endl;
   NANOS_   INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
   NANOS_   INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) SICOPYSIMASTER );)
               //if (sys.getNetwork()->getNodeNum() == 0) (*myThread->_file) << queueId << ": wd got by si copy constraint " << std::endl;
               //   tdata._feedingVector[ queueId ]++;
              //message("Ive got a wd, Im at node " << queueId );
               return wd;
            }
            if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints<SiCopyNoMaster> ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) SICOPYNOMASTER );)
               //   tdata._feedingVector[ queueId ]++;
               return wd;
            }
            if ( ( wd = tdata._readyQueues[queueId].popFrontWithConstraints<SiCopy> ( thread ) ) != NULL ) {
               NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) SICOPY );)
               //   tdata._feedingVector[ queueId ]++;
               return wd;
            }
            if ( ( wd = tdata._readyQueues[queueId].pop_front( thread ) ) != NULL ) {
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
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t deb = ID->getEventKey("debug"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( deb, (nanos_event_value_t) 1 ); )
         wd = fetchWD( thread, NULL );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( deb, 0 ); )
         // *thread->_file << "thread " << thread->getId() << " got wd " << wd << std::endl;


         // if ( wd && sys.getNetwork()->getNodeNum() == 0 ) {
         //    std::size_t scores[ tdata._numQueues ];
         //    std::size_t maxScore = 0;
         //    int winner = computeAffinityScore( *wd, tdata._numQueues, tdata._numNodes, scores, maxScore );
         //    if ( scores[winner] != scores[queueId] ) {
         //       sys.increaseAffinityFailureCount();
         //    }
         //    //(*myThread->_file) << "This wd should run in " << winner << " (score of " << scores[winner] << ", will do it on " << queueId << ( ( winner == data._nodeId || scores[winner] == scores[data._nodeId] ) ? " gr8" : " f4il" ) << std::endl;
         // }


         if ( _steal )
         {
            if ( wd == NULL ) {
               for ( unsigned int i = queueId + 1; i < tdata._numNodes; i++ ) {
                  if ( tdata._readyQueues[i].size() > 1 ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
               for ( unsigned int i = 0; i < queueId; i++ ) {
                  if ( tdata._readyQueues[i].size() > 1 ) {
                     wd = tdata._readyQueues[i].pop_front( thread );
                     return wd;
                  } 
               }
            }
         }
         if ( wd == NULL ) {
            OS::nanosleep( 100 );
         } /*else {
            if ( !wd->resourceCheck( *thread, false ) ) {
               (*myThread->_file) << "Running wd " << wd->getId() << " will trigger an invalidation."<< std::endl;
            }
         }*/
         return wd;
      }

      void CacheSchedPolicy::tryGetLocationData( BaseThread *thread ) {
         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         data.initialize( thread );
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
         if ( !tdata._unrankedQueue.empty() ) {
            WD *wd = tdata._unrankedQueue.pop_front( thread );
           
            if ( wd != NULL ) {
               //bool succeeded = true;
               for ( unsigned int i = 0; i < wd->getNumCopies(); i++ ) {
                  //if ( wd->_mcontrol.getCacheCopies()[ i ]._reg.id == 0 ) {
                  if ( !wd->_mcontrol._memCacheCopies[ i ]._locationDataReady ) {
                //     (*myThread->_file) << "trygetLoc at "<< __FUNCTION__<<std::endl;
                     //succeeded = succeeded && wd->_mcontrol.getCacheCopies()[ i ].tryGetLocation( *wd, i );
                     wd->_mcontrol._memCacheCopies[ i ].getVersionInfo();
                  }
               }
               //if ( succeeded ) {
              //    (*myThread->_file) << "got a wd delayed using "<< __FUNCTION__<<std::endl;
                  rankWD( thread, *wd );
               //} else {
               //   (*myThread->_file) << "readd a wd delayed using "<< __FUNCTION__<<std::endl;
               //   tdata._unrankedQueue.push_front( wd );
               //}
            }
         }
      }

      int CacheSchedPolicy::computeAffinityScore( WD &wd, unsigned int numQueues, unsigned int numNodes, std::size_t *scores, std::size_t &maxPossibleScore ) {
         CopyData * copies = wd.getCopies();
         for (unsigned int i = 0; i < numQueues; i++ ) {
            scores[i] = 0;
         }
         maxPossibleScore = 0;
         for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
            if ( !copies[i].isPrivate() && (
                     ( copies[i].isInput() && copies[i].isOutput() && _affinityInout ) || ( !_affinityInout )
                     ) ) {
               NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
               maxPossibleScore += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
               //*myThread->_file << "Affinity score for region "; wd._mcontrol._memCacheCopies[ i ]._reg.key->printRegion( *myThread->_file, wd._mcontrol._memCacheCopies[ i ]._reg.id );
               //{
               //   NewNewDirectoryEntryData *entry = NewNewRegionDirectory::getDirectoryEntry( *wd._mcontrol._memCacheCopies[ i ]._reg.key, wd._mcontrol._memCacheCopies[ i ]._reg.id );
               //   *myThread->_file << " " << *entry << std::endl;
               //}
               if ( locs.empty() ) {
                  //(*myThread->_file) << "empty list, version "<<  wd._mcontrol._memCacheCopies[ i ]._version << std::endl;
                  for ( std::set< memory_space_id_t >::const_iterator locIt = wd._mcontrol._memCacheCopies[ i ]._reg.getLocations().begin();
                        locIt != wd._mcontrol._memCacheCopies[ i ]._reg.getLocations().end(); locIt++ ) {
                     memory_space_id_t loc = *locIt;
                     unsigned int score_idx = ( loc != 0 ? sys.getSeparateMemory( loc ).getNodeNumber() : 0 );
                     if (wd._mcontrol._memCacheCopies[ i ]._reg.isRooted()) {
                        scores[ score_idx ] = (std::size_t) -1;
                     } else if ( scores[ score_idx ] != (std::size_t) -1 ) {
                        scores[ score_idx ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                     }
                     if ( loc != 0 && sys.getSeparateMemory( loc ).isAccelerator()
                           && wd.canRunIn( sys.getSeparateMemory( loc ).getDevice() ) ) {
                        int accelerator_id = sys.getSeparateMemory( loc ).getAcceleratorNumber();
                        scores[ numNodes + accelerator_id ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                     }
                     if ( score_idx == 0 && !wd.canRunIn( SMP ) ) {
                        scores[ 0 ] = 0;
                     }

                  }
               } else {
                  for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {

                     for ( std::set< memory_space_id_t >::const_iterator locIt = wd._mcontrol._memCacheCopies[ i ]._reg.getLocations().begin();
                           locIt != wd._mcontrol._memCacheCopies[ i ]._reg.getLocations().end(); locIt++ ) {
                        memory_space_id_t loc = *locIt;

                        unsigned int score_idx = ( loc != 0 ? sys.getSeparateMemory( loc ).getNodeNumber() : 0 );
                        if (wd._mcontrol._memCacheCopies[ i ]._reg.isRooted()) {
                           scores[ score_idx ] = (std::size_t) -1;
                        } else if ( scores[ score_idx ] != (std::size_t) -1 ) {
                           scores[ score_idx ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                        }
                        if ( loc != 0 && sys.getSeparateMemory( loc ).isAccelerator()
                              && wd.canRunIn( sys.getSeparateMemory( loc ).getDevice() ) ) {
                           int accelerator_id = sys.getSeparateMemory( loc ).getAcceleratorNumber();
                           scores[ numNodes + accelerator_id ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                           if ( score_idx == 0 && !wd.canRunIn( SMP ) ) {
                              scores[ 0 ] = 0;
                           }
                        }
                     }
                  }
               }
            } //else { (*myThread->_file) << "ignored copy "<< std::endl; }
         }

         //(*myThread->_file) << "scores for wd " << wd.getId() << " :";
         //for (unsigned int idx = 0; idx < numQueues; idx += 1) {
         //   (*myThread->_file) << " [ " << idx << ", " << scores[ idx ] << " ]";
         //}
         //(*myThread->_file) << std::endl;
         int winner = -1;
         unsigned int start = ( _noMaster ) ? 1 : 0 ;
         std::size_t maxRank = 0;
         for ( unsigned int i = start; i < numQueues; i++ ) {
            if ( scores[i] > maxRank ) {
               winner = i;
               maxRank = scores[i];
            }
         }
         if ( winner == -1 )
            winner = start;
         return winner;
      }

      void CacheSchedPolicy::rankWD( BaseThread *thread, WD &wd ) {
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         /* Rank by cluster node */ 
         //(*myThread->_file) << "Ranking wd " << wd.getId() << std::endl;

         std::size_t scores[ tdata._numQueues ];
         //(*myThread->_file) << "RANKING WD " << wd.getId() << " numCopies " << wd.getNumCopies() << std::endl;
         std::size_t max_possible_score = 0;
         int winner = computeAffinityScore( wd, tdata._numQueues, tdata._numNodes, scores, max_possible_score );
         unsigned int usage[ tdata._numQueues ];
         unsigned int ties=0;
         std::size_t maxRank = scores[ winner ];
         unsigned int start = ( _noMaster ) ? 1 : 0 ;
         for ( int i = start; i < ( (int) tdata._numQueues ); i++ ) {
         //(*myThread->_file) << "winner is "<< winner << " ties "<< ties << " " << maxRank<< " this score "<< scores[i] << std::endl;
            if ( scores[i] == maxRank ) {
               usage[ ties ] = i;
               ties += 1;
            }
         }
         //(*myThread->_file) << "winner is "<< winner << " ties "<< ties << " " << maxRank<< std::endl;
         if ( ties > 1 ) {
            //    (*myThread->_file) << "I have to chose between :";
            //for ( unsigned int ii = 0; ii < ties; ii += 1 ) fprintf(stderr, " %d", usage[ ii ] );
            //(*myThread->_file) << std::endl;
            unsigned int minLoad = usage[0];
            for ( unsigned int ii = 1; ii < ties; ii += 1 ) {
               //     (*myThread->_file) << "load of (min) " << minLoad << " is " << tdata._load[ minLoad ] <<std::endl;
               //   (*myThread->_file) << "load of (itr) " << usage[ ii ]  << " is " << tdata._load[ usage[ ii ] ] << std::endl;
               if ( tdata._load[ usage[ ii ] ] < tdata._load[ minLoad ] ) {
                  minLoad = usage[ ii ];
               }
            }
            //(*myThread->_file) << "Well winner is gonna be "<< minLoad << std::endl;
            tdata._load[ minLoad ]++;
            winner = minLoad;
         }
         //if (sys.getNetwork()->getNodeNum() == 0 ) { 
         //   (*myThread->_file) << "WD: " << wd.getId() << " ROcopies: "<<ro_copies << " WOcopies: " << wo_copies << " RWcopies: " << rw_copies << " Locality results: [ ";
         //   for (unsigned int i = 0; i < tdata._numNodes ; i += 1) (*myThread->_file) << i << ": " << (scores[i] / (16*512*512)) << " "; 
         //   (*myThread->_file) <<"] ties " << ties << " winner " << winner << std::endl;
         //}
         //if (winner == -1) winner = start;
         //message("queued wd " << wd.getId() << " to queue " << winner << " scores " << scores[0] << "," << scores[1] << "," << scores[2] << "," << scores[3] );
         //fprintf(stderr, "queued wd %d to queue %d scores %x %x %x %x \n", wd.getId(), winner, scores[0], scores[1], scores[2], scores[3] );
         //(*myThread->_file) << "the winner is " << winner << std::endl;
         wd._mcontrol.setAffinityScore( scores[ winner ] );
         wd._mcontrol.setMaxAffinityScore( max_possible_score );

         /* end of rank by cluster node */

         //(*myThread->_file) << "Winner is " << winner << std::endl;
         if ( winner != 0 ) {
//          *thread->_file << "Ranked wd (id=" << wd.getId() << ", desc=" <<
//             (wd.getDescription() != NULL ? wd.getDescription() : "n/a") <<
//             ") to queue " << winner << std::endl;

            tdata._readyQueues[winner].push_back( &wd );
         } else {
            tdata._readyQueues[winner].push_back( &wd );
#if 0 /* WIP */
            /* rank for accelerators in node 0 */
            std::size_t local_scores[ tdata._numLocalAccelerators ];
            std::size_t max_possible_local_score = 0;
            int local_winner = computeAffinityScore( wd, tdata._numLocalAccelerators, local_scores, max_possible_local_score );

            /* end of rank for accelerators in node 0 */

            if ( local_winner != 0 ) {
               tdata._readyQueuesLocalAccelerators[winner].push_back( &wd );
            } else {
               /* rank smp by NUMA node ? */
               /* end of rank smp by NUMA node ? */
               tdata._readyQueues[winner].push_back( &wd );
            }
#endif
         }
      }

      void CacheSchedPolicy::atSupport ( BaseThread *thread ) {
         //tryGetLocationData( thread );
#ifdef CLUSTER_DEV
         if ( _support ) {
            pickWDtoInitialize( thread );
         }
#endif
      }

      bool CacheSchedPolicy::_steal = false;
      bool CacheSchedPolicy::_noMaster = false;
#ifdef CLUSTER_DEV
      bool CacheSchedPolicy::_support = false;
#endif
      bool CacheSchedPolicy::_invalAware = false;
      bool CacheSchedPolicy::_affinityInout = false;
      bool CacheSchedPolicy::_constraints = false;
      bool CacheSchedPolicy::_immediateSuccessor = false;

      class CacheSchedPlugin : public Plugin
      {
         public:
            CacheSchedPlugin() : Plugin( "Cache-guided scheduling Plugin",1 ) {}

            virtual void config( Config& cfg )
            {
               cfg.setOptionsSection( "Affinity module", "Data Affinity scheduling module" );
               cfg.registerConfigOption ( "affinity-steal", NEW Config::FlagOption( CacheSchedPolicy::_steal ), "Steal tasks from other threads");
               cfg.registerArgOption( "affinity-steal", "affinity-steal" );

               cfg.registerConfigOption ( "affinity-no-master", NEW Config::FlagOption( CacheSchedPolicy::_noMaster ), "Do not execute tasks on master node");
               cfg.registerArgOption( "affinity-no-master", "affinity-no-master" );

#ifdef CLUSTER_DEV
               cfg.registerConfigOption ( "affinity-support", NEW Config::FlagOption( CacheSchedPolicy::_support ), "Use worker threads to help scheduling tasks.");
               cfg.registerArgOption( "affinity-support", "affinity-support" );
#endif

               cfg.registerConfigOption ( "affinity-inval-aware", NEW Config::FlagOption( CacheSchedPolicy::_invalAware ), "Try to execute tasks avoiding invalidations.");
               cfg.registerArgOption( "affinity-inval-aware", "affinity-inval-aware" );

               cfg.registerConfigOption ( "affinity-inout", NEW Config::FlagOption( CacheSchedPolicy::_affinityInout ), "Check affinity for inout data only");
               cfg.registerArgOption( "affinity-inout", "affinity-inout" );

               cfg.registerConfigOption ( "affinity-constraints", NEW Config::FlagOption( CacheSchedPolicy::_constraints ), "Use constrained WD fetching.");
               cfg.registerArgOption( "affinity-constraints", "affinity-constraints" );

               cfg.registerConfigOption ( "affinity-use-immediate-successor", NEW Config::FlagOption( CacheSchedPolicy::_immediateSuccessor ), "Use 'getImmediateSuccessor' to prefetch WDs.");
               cfg.registerArgOption( "affinity-use-immediate-successor", "affinity-use-immediate-successor" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW CacheSchedPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-affinity",nanos::ext::CacheSchedPlugin);
