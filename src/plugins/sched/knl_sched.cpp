/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
#include "regiondict.hpp"
#include "memcachecopy.hpp"
#include "globalregt.hpp"
#include "smpprocessor.hpp"

#define PUSH_BACK_TO_READY_QUEUE( idx, wd ) \
   do {                                     \
      /* *myThread->_file << "Push wd " << (wd)->getId() << " to tdata._readyQueues " << idx << std::endl; */\
      tdata._readyQueues[ idx ].push_back( wd );\
   } while (0)

#define READY_QUEUE( idx, tid ) ( tdata._readyQueues[ idx ]  )

namespace nanos {
   namespace ext {


      class CacheSchedPolicy : public SchedulePolicy
      {
         private:



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

      class WDMap {
         typedef std::set< WD * > _wd_set_t;
         typedef std::map< reg_t, _wd_set_t > _reg_map_t;
         typedef std::map< GlobalRegionDictionary *, _reg_map_t > _wd_map_t;
         _wd_map_t _wdMap;
         std::size_t _wdCount;
         Lock _lock;
         WDDeque _noCopiesWDs;
         WDDeque _fakeQueue;
         WDDeque _topWDsQueue;
         WDDeque _invWDsQueue;
         _wd_set_t _invSet;
         Atomic<unsigned int> _nextComputed;
         Lock _nextComputedLock;
         WDDeque _preQueue;
         Atomic<int> _nextSetReady;
         Atomic<int> _nextSetComputing;
         WD *selected_wd;
         WD *last_selected_wd;

         public:
         WDMap() : _wdMap(), _wdCount( 0 ), _lock(), _noCopiesWDs(), _fakeQueue(),
                   _topWDsQueue(), _invWDsQueue(), _invSet(), _nextComputed(0),
                   _nextComputedLock(),
                   _preQueue(), _nextSetReady(0), _nextSetComputing(0), /*_wdCount(),*/
                   /*_topWDs(),*/ selected_wd(NULL), last_selected_wd(NULL) {
         }

         ~WDMap() {
         }

         WDDeque *_selectLocalQueue( WD *wd ) {
            std::vector<ProcessingElement * > *accessors[wd->getNumCopies()];
            std::map< ProcessingElement *, std::size_t > scores;
            std::size_t max_score = 0;
            ProcessingElement *max_score_pe = NULL;
            WDDeque *target_queue = NULL;
            for ( unsigned int idx = 0; idx < wd->getNumCopies(); idx += 1 ) {
               accessors[idx] = wd->_mcontrol._memCacheCopies[idx]._reg.getAccessedPEs();
               if ( accessors[idx] != NULL ) {
                  std::size_t reg_size = wd->_mcontrol._memCacheCopies[idx]._reg.getDataSize();
                  for ( unsigned int vidx = 0; vidx < accessors[idx]->size(); vidx += 1 )
                  {
                     ProcessingElement *pe = (*accessors[idx])[vidx];
                     scores[ pe ] += reg_size;
                     std::size_t this_score = scores[ pe ];
                     if ( this_score > max_score ) {
                        max_score = this_score;
                        max_score_pe = pe;
                     }
                  }
               }
            }

            //std::cerr << "Selected PE is " << max_score_pe << std::endl;
            if ( max_score_pe == NULL ) {

            } else {
               BaseThread *thd = max_score_pe->getFirstThread();
               ThreadData &data = ( ThreadData & ) *thd->getTeamData()->getScheduleData();
               target_queue = &data._locaQueue;
            }

            for ( unsigned int idx = 0; idx < wd->getNumCopies(); idx += 1 ) {
               delete accessors[idx];
            }

            return target_queue;
         }

         void _prepareToPush(BaseThread *thread, WD *wd) {
            WD *next = NULL;
            WDPool * wd_queue = wd->getMyQueue();
            if ( wd_queue != NULL ) {
               if ( wd_queue != &_fakeQueue ) {
                  *myThread->_file << "queue error, != _fakeQueue for wd " << wd->getId() << std::endl;
               }
               if ( wd_queue->removeWD( thread, wd, &next ) ) {
                  if ( next != wd ) {
                     *myThread->_file << "2 removeWD failure! " << wd->getId() << std::endl;
                  }
               } else {
                  *myThread->_file << "1 removeWD failure! " << wd->getId() << std::endl;
               }
            } else {
               //*myThread->_file << "null queue for wd " << wd->getId() << std::endl;
            }
         }

         void _insert( WD *wd ) {
            //std::ostream &o = *myThread->_file;
            //o << "insert wd " << wd->getId() << " copies " << wd->getNumCopies() << " wd is tied? " << (wd->isTied() ? wd->isTiedTo()->getId() : -1) << std::endl;
            //ensure (wd->getNumCopies() > 0, "invalid wd, no copies");
            if ( wd->getNumCopies() > 0 ) {
               memory_space_id_t mem_id = myThread->runningOn()->getMemorySpaceId();
               for ( unsigned int idx = 0; idx < wd->getNumCopies(); idx += 1 ) {
                  global_reg_t alloc_reg;
                  sys.getSeparateMemory(mem_id).getCache().getAllocatableRegion(wd->_mcontrol._memCacheCopies[idx]._reg, alloc_reg);
                  GlobalRegionDictionary *key = alloc_reg.key;
                  reg_t id = alloc_reg.id;
                  if ( id == 0  ) {
                     *myThread->_file << "reg 0 in wd " << wd->getId() << " copy idx "<< idx << std::endl;
                     fatal("error");
                  }
                  _wdMap[ key ][ id ].insert( wd );
               }
               _wdCount += 1;
               _fakeQueue.push_back(wd);
               //*myThread->_file << myThread->getId() << " Inserted  wd " << wd->getId() << std::endl;
            } else {
               _noCopiesWDs.push_back(wd);
            }
         }

         void _checkCopiesAndInsert( BaseThread *thread, WD *wd ) {
            memory_space_id_t mem_id = thread->runningOn()->getMemorySpaceId();
            std::map<GlobalRegionDictionary *, std::set<reg_t> > const &map = sys.getSeparateMemory(mem_id).getCache().getAllocatedRegionMap();
            //*myThread->_file << "insert wd " << wd->getId() << " copies " << wd->getNumCopies() << " wd is tied? " << (wd->isTied() ? wd->isTiedTo()->getId() : -1) << std::endl;
            //ensure (wd->getNumCopies() > 0, "invalid wd, no copies");
            //std::ostream &o = *myThread->_file;
            if ( wd->getNumCopies() > 0 ) {
               bool all_copies_allocated = true;
               unsigned int num_copies_found = 0;
               std::size_t not_allocated_bytes = 0;
               for ( unsigned int idx = 0; idx < wd->getNumCopies(); idx += 1 ) {
                  global_reg_t alloc_reg;
                  sys.getSeparateMemory(mem_id).getCache().getAllocatableRegion(wd->_mcontrol._memCacheCopies[idx]._reg, alloc_reg);

                  GlobalRegionDictionary *key = alloc_reg.key;
                  reg_t id = alloc_reg.id;
                  std::size_t allocated_reg_size = alloc_reg.getDataSize();
                  if ( id == 0 ) {
                     *myThread->_file << "reg 0 in wd " << wd->getId() << " copy idx "<< idx << std::endl;
                     fatal("error");
                  }
                  std::map<GlobalRegionDictionary *, std::set<reg_t> >::const_iterator object = map.find( key );
                  if ( object != map.end() ) {
                     std::set< reg_t > const &regions = object->second;
                     bool found = false;
                     for ( std::set< reg_t >:: const_iterator sit = regions.begin(); sit != regions.end() && !found; sit++) {
                        reg_t allocated_reg = *sit;

                        if ( allocated_reg == 1 || allocated_reg == id 
                          // || ( key->checkIntersect( allocated_reg, id ) && key->computeIntersect( allocated_reg, id ) == id ) // condition not needed as we work with allocatedreg
                              ) {
                           found = true;
                           num_copies_found += 1;
                        }
                     }
                     all_copies_allocated = all_copies_allocated && found;
                     if ( !found ) {
                        not_allocated_bytes += allocated_reg_size;
                     }
                  } else {
                     all_copies_allocated = false;
                     not_allocated_bytes += allocated_reg_size;
                  }
               }
               if ( all_copies_allocated ) {
                  //o << " all copies ok for wd " << wd->getId() << std::endl;
                  WDDeque *queue = _selectLocalQueue( wd );
                  if ( queue != NULL ) {
                     queue->push_back( wd );
                  } else {
                     _topWDsQueue.push_back( wd );
                  }
               } else if ( not_allocated_bytes < sys.getSeparateMemory(mem_id).getCache().getUnallocatedBytes() ) {
                  //not all copies are allocated, but there is space in the cache
                  WDDeque *queue = _selectLocalQueue( wd );
                  if ( queue != NULL ) {
                     queue->push_back( wd );
                  } else {
                     _topWDsQueue.push_back( wd );
                  }
               } else {
                  this->_insert(wd);
               }
            } else {
               _noCopiesWDs.push_back(wd);
            }
         }

         void _getWithWD( WD *wd, BaseThread *thread ) {
            NANOS_INSTRUMENT(static nanos_event_key_t ikey = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
            memory_space_id_t mem_id = thread->runningOn()->getMemorySpaceId();
            //*myThread->_file << "computing WDs with base wd " << wd->getId() << " has numCopies " << wd->getNumCopies() << std::endl;
            //std::ostream &o = *thread->_file;
            if ( wd->getNumCopies() > 0 ) {
               int count=0;
               std::map< WD *, unsigned int > wd_count;
               std::map< GlobalRegionDictionary *, std::set< reg_t > > wd_regions;
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 137 );)
               for ( unsigned int idx = 0; idx < wd->getNumCopies(); idx += 1 ) {
                  global_reg_t allocated_reg;
                  sys.getSeparateMemory(mem_id).getCache().getAllocatableRegion(wd->_mcontrol._memCacheCopies[idx]._reg, allocated_reg);
                  GlobalRegionDictionary *key = wd->_mcontrol._memCacheCopies[idx]._reg.key;
                  if ( wd_regions[key].count( allocated_reg.id ) == 0 ) {
                     wd_regions[key].insert( allocated_reg.id );
                  } else {
                     continue;
                  }
                  reg_t id = wd->_mcontrol._memCacheCopies[idx]._reg.id;
                  //*myThread->_file << "allocated reg is " << allocated_reg.id << " orig reg is " << id << std::endl;
                  if ( id == 0 ) {
                     *myThread->_file << "reg 0 in wd " << wd->getId() << " copy idx "<< idx << std::endl;
                     fatal("error");
                  }
                  _wd_map_t::iterator wd_map_it = _wdMap.find( key );
                  if ( wd_map_it != _wdMap.end() ) {
                     _reg_map_t &reg_map = wd_map_it->second;
                     //*myThread->_file << "copy " << idx << " found " << std::endl;
                     for ( _reg_map_t::iterator reg_map_it = reg_map.begin(); reg_map_it != reg_map.end(); ) {
                        reg_t registered_reg = reg_map_it->first;
                        _wd_set_t &wd_set = reg_map_it->second;
                        reg_map_it++;
                        if ( registered_reg == 1 || registered_reg == allocated_reg.id
                            /* ||  ( key->checkIntersect( allocated_reg.id, registered_reg ) && key->computeIntersect( allocated_reg.id, registered_reg ) == registered_reg ) */
                           ) {
                           //_invSet.insert(wd_set.begin(), wd_set.end());
                           for ( _wd_set_t::iterator sit = wd_set.begin(); sit != wd_set.end(); ) { 
                              WD *this_wd = *sit;
                              if ( this_wd != wd ) {
                                 unsigned int current_count = wd_count[ this_wd ] + 1;
                                 if ( current_count == this_wd->getNumCopies() ) {
                                    wd_count.erase( this_wd );
                     //o << "\t insert in inv set wd: " << this_wd->getId() << std::endl;
                                    _invSet.insert( this_wd );
                                    count++;
                                 } else {
                                    wd_count[ this_wd ] = current_count;
                                 }
                              }
                              sit++;
                           //   this->_prepareToPush(thread, this_wd);
                           //   this->_remove(this_wd);
                           //   _invWDsQueue.push_back( this_wd );
                           }
                        }
                     }
                  }
               }
               NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 0 );)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 9000000 + count );)
               NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 0 );)
            }
         }

         void _tryComputeNextSet( BaseThread *thread ) {
            if ( selected_wd != NULL ) {
               //if ( _nextComputed.cswap( 0, 1 ) )
               if ( _nextComputedLock.tryAcquire() )
               {
                  if ( _invSet.empty() ) {
                     //std::ostream &o = *thread->_file;
                     NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 134 );)
                     this->_getWithWD( selected_wd, thread );
                     for ( _wd_set_t::const_iterator invIt = _invSet.begin(); invIt != _invSet.end(); invIt++ ) {
                        //this->_prepareToPush( thread, *invIt );
                        this->_remove(*invIt);
                     }
                     //o << "computed the set of inv wds: " << _invSet.size() << " with wd " << selected_wd->getId() << std::endl;
                     _invSet.insert(selected_wd);
                     //this->_prepareToPush( thread, selected_wd );
                     this->_remove(selected_wd);
                     NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 0 );)
                  }
                  last_selected_wd = selected_wd;
                  selected_wd = NULL;
                  _nextComputedLock.release();
               }// else {
                //  *myThread->_file << "not computing next set: val is " << _nextComputed.value() << std::endl;
                // }
            }
         }

         void insert( WD *wd ) {
            while ( !_lock.tryAcquire() ) { myThread->processTransfers(); }
            this->_insert(wd);
            _lock.release();
         }

         void preQueue( WD *wd ) {
            if ( wd->getNumCopies() > 0 ) {
               _preQueue.push_back( wd );
            } else {
               _noCopiesWDs.push_back(wd);
            }
         }

         WD *_getRandomWD() {
            WD *wd = NULL;
            for ( _wd_map_t::iterator it = _wdMap.begin(); it != _wdMap.end() && wd == NULL; it++ ) {
               _reg_map_t &reg_map = it->second;
               for( _reg_map_t::iterator rit = reg_map.begin(); rit != reg_map.end() && wd == NULL; rit++ ) {
                  _wd_set_t &wd_set = rit->second;
                  if ( !wd_set.empty() ) {
                     wd = *(wd_set.begin());
                  }
               }
            }
            return wd;
         }

         WD *fetch( BaseThread *thread ) {
            NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
            //std::ostream &o = *myThread->_file;
            memory_space_id_t id = thread->runningOn()->getMemorySpaceId();
            //o << "fetch for " << id << std::endl;
            WD *top_wd = NULL;
            //WD *inv_wd = NULL;
            if ( _topWDsQueue.empty() ) {
               if ( _lock.tryAcquire() ) {
                  //std::map<GlobalRegionDictionary *, std::set<reg_t> > const &map = sys.getSeparateMemory(id).getCache().getAllocatedRegionMap();
                  //NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 131 );)
                  while ( !_preQueue.empty() ) {
                     WD *preWD = _preQueue.pop_front( thread );
                     if ( preWD != NULL ) {
                        this->_checkCopiesAndInsert(thread, preWD);
                     } else {
                        break;
                     }
                  }
                  //NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
                  _tryComputeNextSet( thread );

                   // *myThread->_file << "topWDs queues: " << _topWDsQueue.size() << " invSet: " << _invSet.size() << std::endl;
                  if ( _topWDsQueue.empty() && !_invSet.empty() ) {
                     //if ( _nextComputed.cswap( 1, 0 ) ) 
                     if ( _nextComputedLock.tryAcquire() ) 
                     {
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 132 );)
                        //o << "insert from invSet: { ";
                        for ( _wd_set_t::const_iterator it = _invSet.begin(); it != _invSet.end(); it++ ) {
                           WD *this_wd = *it;
                           //o << this_wd << "," << this_wd->getId() << " ";
                           this->_prepareToPush( thread, this_wd );
            //*myThread->_file << myThread->getId() << " remove (from process _invSet) wd " << this_wd->getId() << std::endl;
                           //this->_remove(this_wd);
                           WDDeque *queue = _selectLocalQueue( this_wd );
                           if ( queue != NULL ) {
                              queue->push_back( this_wd );
                           } else {
                              _topWDsQueue.push_back( this_wd );
                           }
                        }
                        //o << "}" << std::endl;
                        _invSet.clear();
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
                        //*myThread->_file<<"done process inv set" << std::endl;
                        _nextComputedLock.release();
                     }
                  }

                  if ( _topWDsQueue.empty() && sys.getSeparateMemory(id).getCache().getCurrentAllocations() == 0 ) {
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 146 );)
               _scan(thread);
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
                  }
                  top_wd = _topWDsQueue.pop_front( thread );
                  _lock.release();
               } else {
                  myThread->processTransfers();
               }
               if ( top_wd == NULL ) {
                  if ( _lock.tryAcquire() ) {
                     _tryComputeNextSet( thread );
                     _lock.release();
                  }
               }
            } else {
               if ( _lock.tryAcquire() ) {
                  _tryComputeNextSet( thread );
                  _lock.release();
               }
               top_wd = _topWDsQueue.pop_front( thread );
               //*myThread->_file << thread->getId() << " fetch WD no lock, got wd " << (top_wd != NULL ? top_wd->getId() : -1 ) << std::endl;
            }

            if ( top_wd == NULL ) {
               top_wd = _noCopiesWDs.pop_front( thread );
            }

            return top_wd;
         }

         void _scan( BaseThread *thread ) {
            NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
            memory_space_id_t id = thread->runningOn()->getMemorySpaceId();
            std::map<GlobalRegionDictionary *, std::set<reg_t> > const &map = sys.getSeparateMemory(id).getCache().getAllocatedRegionMap();
            int inserted_wds = 0;
            //std::set< WD * > top_wds;
            std::map< WD *, unsigned int > wd_count;
            //std::ostream &o = *myThread->_file;
            //_topWDs.clear();

            //o << "allocd' objects " << map.size() << std::endl;
            for( std::map<GlobalRegionDictionary *, std::set<reg_t> >::const_iterator it = map.begin();
                  it != map.end(); it++ ) {
               _reg_map_t &reg_map = _wdMap[ it->first ];
               //it->first->lockContainer();
               NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 133 );)
               //o << "\tobject " << it->first << " has regs: " <<  it->second.size() << std::endl;
               for ( std::set<reg_t>::iterator rit = it->second.begin(); rit != it->second.end(); rit++ ) {
                  //o << "\t\t"; it->first->printRegion(o, *rit ); o << std::endl;
                  for ( _reg_map_t::iterator srit = reg_map.begin(); srit != reg_map.end(); ) 
                  {

                     //o << "\t"; it->first->printRegion(o, srit->first ); o << std::endl;
                     if ( *rit == 1 || *rit == srit->first
#if 0
                           || ( it->first->checkIntersect( /*allocated reg*/ *rit, /*wd reg */ srit->first )
                              && it->first->computeIntersect( /*allocated reg*/ *rit, /*wd reg */ srit->first ) == srit->first
                              )
#endif
                        ) {
                        _wd_set_t &wd_set = srit->second;
                        srit++;
                        //o << "FUCKING HIT " << wd_set.size() << std::endl;

                        //o << *rit << " & " << srit->first << " wds: " << wd_set.size() << std::endl;
                        for ( _wd_set_t::iterator wit = wd_set.begin(); wit != wd_set.end(); ) {
                           WD *wd = *wit;
                           //_wd_set_t::iterator this_it = wit;
                           wit++;
                           if ( wd->getNumCopies() == 1 ) {
                              this->_prepareToPush(thread, wd);
            //*myThread->_file << myThread->getId() << " remove (from scan) wd " << wd->getId() << std::endl;
                              this->_remove(wd);
                              //_wdCount -= 1;
                              //wd_set.erase(this_it);
            //NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 133 );)
                              WDDeque *queue = _selectLocalQueue( wd );
                              if ( queue != NULL ) {
                                 queue->push_back( wd );
                              } else {
                                 _topWDsQueue.push_back( wd );
                              }
            //NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
                              inserted_wds += 1;
                           } else {
                              unsigned int current_count = wd_count[ wd ] + 1;
                              if ( current_count == wd->getNumCopies() ) {
                                 this->_prepareToPush(thread, wd);
            //*myThread->_file << myThread->getId() << " remove (from scan(2)) wd " << wd->getId() << std::endl;
                                 this->_remove(wd);
                                 wd_count.erase( wd );
            //NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 133 );)
                                 WDDeque *queue = _selectLocalQueue( wd );
                                 if ( queue != NULL ) {
                                    queue->push_back( wd );
                                 } else {
                                    _topWDsQueue.push_back( wd );
                                 }
            //NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
                                 inserted_wds += 1;
                                 //_topWDs.erase( wd );
                              } else {
                                 wd_count[ wd ] = current_count;
                                 //if ( wd->getNumCopies() > 1 && current_count == wd->getNumCopies() - 1 ) {
                                 //   _topWDs.insert( wd );
                                 //}
                              }
                           }
                        }
                     } else {
                        srit++;
                     }
                  }
               }
               //it->first->releaseContainer();
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
            }
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 1000000 + inserted_wds );)
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
            if ( selected_wd == NULL ) {
               NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 140 );)
               // if ( _topWDs.size() > 0 ) {
               //    selected_wd = *_topWDs.begin();
               // } else
               if ( wd_count.size() > 0 ) {
                  unsigned int min_diff = UINT_MAX;
                  for (std::map<WD*, unsigned int>::const_iterator wdit = wd_count.begin(); wdit != wd_count.end(); wdit++) {
                     unsigned int this_diff = ( wdit->first->getNumCopies() - wdit->second );
                     if ( this_diff < min_diff ) {
                        selected_wd = wdit->first;
                  //o << "computed selected_wd w/count, wd "<< (selected_wd!=NULL ? selected_wd->getId() : 0) << std::endl;
                        min_diff = this_diff;
                     }
                     //o << "[wd: " << wdit->first->getId() << "," << wdit->first->getNumCopies()<<","<< wdit->second <<"]";
                  } //o << std::endl << " computed selected_wd to " << selected_wd << std::endl;
                  //print();
               } else {
                  selected_wd = this->_getRandomWD();
                  //o << "computed selected_wd w/random, wd "<< (selected_wd!=NULL ? selected_wd->getId() : 0) << std::endl;
               }
               NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
            }
         }

         void _remove( WD *wd ) {
            if ( wd->getNumCopies() > 0 ) {
               _wdCount -= 1;
               memory_space_id_t mem_id = myThread->runningOn()->getMemorySpaceId();
               for ( unsigned int idx = 0; idx < wd->getNumCopies(); idx += 1 ) {
                  global_reg_t alloc_reg;
                  sys.getSeparateMemory(mem_id).getCache().getAllocatableRegion(wd->_mcontrol._memCacheCopies[idx]._reg, alloc_reg);
                  GlobalRegionDictionary *key = alloc_reg.key;
                  reg_t id = alloc_reg.id;
                  _reg_map_t &reg_map = _wdMap[key];
                  _reg_map_t::iterator it = reg_map.find(id);
                  if ( it == reg_map.end() ) {
                     *myThread->_file << "Warning, attempt to erase a WD without reg map. WD( " << wd << " ) id: "<< wd->getId() << std::endl;
                  }
                  _wd_set_t &wd_set = it->second;
                  wd_set.erase(wd);
                  if ( wd_set.empty() ) {
                     reg_map.erase(it);
                  }
               }
               //*myThread->_file << "Removed wd " << wd->getId() << std::endl;
            }
         }

         void prefetch( BaseThread *thread, WD &wd ) {
            NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
            if ( &wd == last_selected_wd ) {
               //*myThread->_file << "could prefetch!!" << std::endl;
               _lock.acquire();
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 144 );)
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
               last_selected_wd = NULL;
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( key, 145 );)
               _scan(thread);
            NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 );)
               _lock.release();
            }
         }

         void print() const {
            std::ostream &o = *myThread->_file;
            for( _wd_map_t::const_iterator it = _wdMap.begin(); it != _wdMap.end(); it++ ) {
               _reg_map_t const &reg_map = it->second;
               o << "Object " << it->first << std::endl;
               for ( _reg_map_t::const_iterator rit = reg_map.begin(); rit != reg_map.end(); rit++ ) {
                  o << "\treg " << rit->first << ": ";
                  _wd_set_t const &wd_set = rit->second;
                  for ( _wd_set_t::const_iterator wit = wd_set.begin(); wit != wd_set.end(); wit++ ) {
                     o << "[" << (*wit)->getId() << "," << (*wit)->getNumCopies() << "]";
                  }
                  o << std::endl;
               }
            }
         }
      };

            struct TeamData : public ScheduleTeamData
            {
               WDDeque            _globalReadyQueue;
               WDDeque            _unrankedQueue;
               WDDeque*           _readyQueues;
               unsigned int       _numQueues;
               unsigned int      *_load;
               unsigned int       _lastNodeScheduled;
               WDMap _wdMap;
 
               TeamData ( unsigned int size ) : ScheduleTeamData(), _globalReadyQueue(), _unrankedQueue()
               {
                  unsigned int count = 0;
                  for ( PEList::const_iterator it = sys.getPEs().begin(); it != sys.getPEs().end(); it++ ) {
                     if ( it->second->getClusterNode() == 0 ) {
                        SMPProcessor *cpu = (SMPProcessor *) it->second;
                        if ( cpu->isActive() ) {
                           std::cerr << "This is pe " << it->first << " w/id " << cpu->getBindingId() << " active? " << cpu->isActive() << std::endl;
                           count += 1;
                        }
                     }
                  }

                  _numQueues = count;

                  _readyQueues = NEW WDDeque[_numQueues];
                  _load = NEW unsigned int[_numQueues];
                  for (unsigned int i = 0; i < _numQueues; i += 1) {
                     _load[ i ] = 0;
                  }
                  _lastNodeScheduled = 1;
               }

               ~TeamData ()
               {
                  if (_numQueues >= 1 ) {
                     delete[] _readyQueues;
                  }
                  /* TODO add delete for new members */
                  delete[] _load;
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

            /* disable copy and assigment */
            explicit CacheSchedPolicy ( const CacheSchedPolicy & );
            const CacheSchedPolicy & operator= ( const CacheSchedPolicy & );


            enum DecisionType { /* 0 */ NOOP,
                                /* 1 */ NOCONSTRAINT,
                                /* 2 */ NOCOPY,
                                /* 3 */ SICOPYSIMASTER,
                                /* 4 */ SICOPYNOMASTER,
                                /* 5 */ SICOPY,
                                /* 6 */ SICOPYSIMASTERINIT,
                                /* 7 */ SICOPYNOMASTERINIT,
                                /* 8 */ SICOPYNOMASTERINIT_SELF,
                                /* 9 */ ALREADYINIT };

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
            static bool _verbose;
            static int _currentThd;
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
               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               data.initialize( thread );
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
              
               tdata._wdMap.preQueue( &wd );

                  return;

//                if ( wd.getDepth() > 1 ) {
//             //message("enqueue because of depth > 1 at queue 0 (this node) " << sys.getNetwork()->getNodeNum()<< " wd os " << wd.getId());
//                   //tdata._readyQueues[0].push_back( &wd );
//                   PUSH_BACK_TO_READY_QUEUE( 0, &wd );
//                   return;
//                }
            //message("in queue node " << sys.getNetwork()->getNodeNum()<< " wd os " << wd.getId());
//               if ( wd.isTied() ) {
//                  unsigned int index = wd.isTiedTo()->runningOn()->getClusterNode();
//                  //tdata._readyQueues[index].push_back ( &wd );
//                  PUSH_BACK_TO_READY_QUEUE( index, &wd );
//                  return;
//               }
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

            virtual WD *atIdle ( BaseThread *thread, int numSteal );
            virtual WD *atBlock ( BaseThread *thread, WD *current );

            virtual WD *atAfterExit ( BaseThread *thread, WD *current )
            {
               return atBlock(thread, current );
            }

            WD * atPrefetch ( BaseThread *thread, WD &current )
            {
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               WD * found = NULL;
               if ( _immediateSuccessor ) {
                  found = current.getImmediateSuccessor(*thread);
                  if ( found ) {
                     found->_mcontrol.preInit();
                     if ( current._mcontrol.containsAllCopies( found->_mcontrol ) ) {
                        //*myThread->_file << "wd " << found->getId() << "( " << (found->getDescription() != NULL ? found->getDescription() : "n/a") << ") is a good immediate successor! "<< std::endl;
                     } else {
                        tdata._wdMap.insert(found);
                        found = NULL;// atIdle(thread);
                        //*myThread->_file << "wd " << found->getId() << "( " << (found->getDescription() != NULL ? found->getDescription() : "n/a") << ") is NOT a good immediate successor! "<< std::endl;
                     }
                  }
                  //if ( found ) {
                  //   (*myThread->_file) << " atPrefetch (getImmediateSuccessor) returns wd " << found->getId() << std::endl;
                  //}
               }
               return found != NULL ? found : atIdle(thread,false);
            }
         
            WD * atBeforeExit ( BaseThread *thread, WD &current, bool schedule )
            {
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               WD * found = NULL;
               tdata._wdMap.prefetch( thread, current );
               if ( _immediateSuccessor ) {
                  found = current.getImmediateSuccessor(*thread);
                  if ( found ) {
                     found->_mcontrol.preInit();
                     if ( current._mcontrol.containsAllCopies( found->_mcontrol ) ) {
                        //*myThread->_file << "wd " << found->getId() << "( " << (found->getDescription() != NULL ? found->getDescription() : "n/a") << ") is a good immediate successor! "<< std::endl;
                     } else {
                        tdata._wdMap.insert(found);
                        found = NULL;
                        //*myThread->_file << "wd " << found->getId() << "( " << (found->getDescription() != NULL ? found->getDescription() : "n/a") << ") is NOT a good immediate successor! "<< std::endl;
                     }
                  }
                  //if ( found ) {
                  //   (*myThread->_file) << " atBeforeExit (getImmediateSuccessor) returns wd " << found->getId() << std::endl;
                  //}
               }
               return found;
            }

            WD *fetchWD ( BaseThread *thread, WD *current );  
            virtual void atSupport ( BaseThread *thread );

      };

      inline WD *CacheSchedPolicy::fetchWD( BaseThread *thread, WD *current )
      {
         WorkDescriptor * wd = NULL;
         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         data.initialize( thread );
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         wd = data._locaQueue.pop_front( thread );
         if ( wd == NULL ) {
            wd = tdata._wdMap.fetch( thread );
            //std::cerr << "got it from wdmap" << std::endl;
         } /* else {
            std::cerr << "got it from local queue" << std::endl;
         } */
         return wd;
      }

      WD *CacheSchedPolicy::atBlock ( BaseThread *thread, WD *current )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         data.initialize( thread );
         //TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

//         if ( thread->getId() == 0 ) {
//            while ( !tdata._unrankedQueue.empty() ) {
//               tryGetLocationData( thread );
//            }
//         }

         wd = fetchWD( thread, current ) ;

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
      WD * CacheSchedPolicy::atIdle ( BaseThread *thread, int numSteal )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         data.initialize( thread );
         //TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         //if ( tdata._numQueues == 1 ) {
         //   wd = tdata._globalReadyQueue.pop_front( thread );
         //   return wd;
         //}

//         if ( thread->getId() == 0 ) {
//            while ( !tdata._unrankedQueue.empty() ) {
//               tryGetLocationData( thread );
//            }
//         }
         wd = fetchWD( thread, NULL );

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
               //   tdata._unrankedQueue.push_back( wd );
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
               // *myThread->_file << "Affinity score for region "; wd._mcontrol._memCacheCopies[ i ]._reg.key->printRegion( *myThread->_file, wd._mcontrol._memCacheCopies[ i ]._reg.id );
               // {
               //    NewNewDirectoryEntryData *entry = NewNewRegionDirectory::getDirectoryEntry( *wd._mcontrol._memCacheCopies[ i ]._reg.key, wd._mcontrol._memCacheCopies[ i ]._reg.id );
               //    *myThread->_file << " " << *entry << std::endl;
               // }
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
                     if ( score_idx == 0 && !wd.canRunIn( *getSMPDevice() ) ) {
                        scores[ 0 ] = 0;
                     }

                  }
               } else {
                  for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                     global_reg_t data_source_reg( it->second, wd._mcontrol._memCacheCopies[ i ]._reg.key );
                     global_reg_t region_shape( it->first, wd._mcontrol._memCacheCopies[ i ]._reg.key );

                     for ( std::set< memory_space_id_t >::const_iterator locIt = data_source_reg.getLocations().begin();
                           locIt != data_source_reg.getLocations().end(); locIt++ ) {
                        memory_space_id_t loc = *locIt;

                        unsigned int score_idx = ( loc != 0 ? sys.getSeparateMemory( loc ).getNodeNumber() : 0 );
                        if (data_source_reg.isRooted()) {
                           scores[ score_idx ] = (std::size_t) -1;
                        } else if ( scores[ score_idx ] != (std::size_t) -1 ) {
                           scores[ score_idx ] += region_shape.getDataSize();
                        }
                        if ( loc != 0 && sys.getSeparateMemory( loc ).isAccelerator()
                              && wd.canRunIn( sys.getSeparateMemory( loc ).getDevice() ) ) {
                           int accelerator_id = sys.getSeparateMemory( loc ).getAcceleratorNumber();
                           scores[ numNodes + accelerator_id ] += region_shape.getDataSize();
                           if ( score_idx == 0 && !wd.canRunIn( *getSMPDevice() ) ) {
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
         //int winner = computeAffinityScore( wd, tdata._numQueues, tdata._numNodes, scores, max_possible_score );
         int winner = 1;
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
         //   (*myThread->_file) << "WD: " << wd.getId() << " Locality results: [ ";
         //   for (unsigned int i = 0; i < tdata._numNodes ; i += 1) (*myThread->_file) << i << ": " << scores[i] << " "; 
         //   (*myThread->_file) <<"] ties " << ties << " winner " << winner << std::endl;
         //}
         //if (winner == -1) winner = start;
         //message("queued wd " << wd.getId() << " to queue " << winner << " scores " << scores[0] << "," << scores[1] << "," << scores[2] << "," << scores[3] );
         //fprintf(stderr, "queued wd %d to queue %d scores %x %x %x %x \n", wd.getId(), winner, scores[0], scores[1], scores[2], scores[3] );
         //(*myThread->_file) << "the winner is " << winner << std::endl;
         wd._mcontrol.setAffinityScore( scores[ winner ] );
         wd._mcontrol.setMaxAffinityScore( max_possible_score );

         /* end of rank by cluster node */

         // (*myThread->_file) << "Winner is " << winner << std::endl;
         // printBt(*myThread->_file);
         if ( winner != 0 ) {
//          *thread->_file << "Ranked wd (id=" << wd.getId() << ", desc=" <<
//             (wd.getDescription() != NULL ? wd.getDescription() : "n/a") <<
//             ") to queue " << winner << std::endl;

            //tdata._readyQueues[winner].push_back( &wd );
            PUSH_BACK_TO_READY_QUEUE( winner, &wd );
         } else {
            //tdata._readyQueues[winner].push_back( &wd );
            PUSH_BACK_TO_READY_QUEUE( winner, &wd );
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
      bool CacheSchedPolicy::_verbose = false;

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

               cfg.registerConfigOption ( "affinity-verbose", NEW Config::FlagOption( CacheSchedPolicy::_verbose ), "Print verbose messages.");
               cfg.registerArgOption( "affinity-verbose", "affinity-verbose" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW CacheSchedPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-affinity",nanos::ext::CacheSchedPlugin);
