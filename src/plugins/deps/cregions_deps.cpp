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

#include "basedependenciesdomain.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"
#include "depsregion.hpp"
#include "compatibility.hpp"
#include <vector>
#include <map>

namespace nanos {
   namespace ext {
       
       
      class CRegionsPair {
          private:
          
            std::vector< TrackableObject* > * _trackableObjs;
            unsigned int _itNum;
          
          public:

            CRegionsPair(std::vector< TrackableObject* > * trackableObjs, unsigned int itNum):
             _trackableObjs(trackableObjs) , _itNum(itNum){}
              
            unsigned int getItNum() const {
                return _itNum;
            }

            void setItNum(unsigned int itNum) {
                this->_itNum = itNum;
            }

            std::vector<TrackableObject*>* getTrackableObjs() const {
                return _trackableObjs;
            }

            void setTrackableObjs(std::vector<TrackableObject*>* trackableObjs) {
                this->_trackableObjs = trackableObjs;
            } 

          
      };

      class CRegionsDependenciesDomain : public BaseDependenciesDomain
      {
         private:
            typedef std::vector< std::pair < DepsRegion, TrackableObject*> > DepsVector; /**< Maps addresses to Trackable objects */
            typedef std::map< DepsRegion, CRegionsPair* > DepsCacheMap; /**< Maps addresses to Trackable objects */
            typedef std::map< DepsRegion, TrackableObject* > TrackablesMap; /**< Maps addresses to Trackable objects */
            typedef std::map< DepsRegion, TrackableObject* > SizesMap; /**< Maps addresses to Trackable objects */

         private:
            DepsVector _addressDependencyVector; /**< Used to track dependencies between DependableObject */
            DepsCacheMap _addressDependencyCache; /**< Used to track dependencies between DependableObject */
            SizesMap _addressDependencySmall;
            SizesMap _addressDependencyMid;
            const size_t _sizeThresholdSmall;
            const size_t _sizeThresholdMid;
         private:
            /*! \brief Looks for the dependency's address in the domain and returns the trackableObject associated.
             *  \param dep Dependency to be checked.
             *  \sa Dependency TrackableObject
             */
            void lookupDependency ( const DepsRegion& target, std::vector<TrackableObject* > * result  )
            {
               
//               DepsMap::iterator it = _addressDependencyMap.find( target ); 
//               if ( it == _addressDependencyMap.end() ) {
//               } else {
//                  status = it->second;
//               }               unsigned int vectorIdx = 0;
               unsigned int vectorIdx = 0;
               
               DepsCacheMap::iterator itCache = _addressDependencyCache.find( target ); 
               std::vector<TrackableObject* > * objs;
               unsigned int vectorEnd=_addressDependencyVector.size();
               //If object not in cache, its new, initialize empty cache
               //And address Maps
               TrackableObject* currStatus = NULL;
               bool hasToInsert=itCache == _addressDependencyCache.end();
               if ( hasToInsert ) { 
                   std::vector<TrackableObject*>* new_objs=NEW std::vector<TrackableObject*>();
                   currStatus = NEW TrackableObject();
                   new_objs->push_back(currStatus);
                   if (target.getSize() > _sizeThresholdMid ) {
                      _addressDependencyVector.push_back( std::make_pair( target, currStatus ) );
                   }
                   CRegionsPair* pair=NEW CRegionsPair(new_objs, 0);
                   itCache=_addressDependencyCache.insert( std::make_pair( target, pair ) ).first;
               }
               CRegionsPair* cacheItem= itCache->second;
               objs=cacheItem->getTrackableObjs();
               vectorIdx=cacheItem->getItNum();
               //Update cache position marker to the end of the vector
               cacheItem->setItNum(_addressDependencyVector.size());
               
               //Search in dependency vector
               for ( ; vectorIdx < vectorEnd ; ++vectorIdx ) {
                   std::pair < DepsRegion, TrackableObject*> item=_addressDependencyVector.at(vectorIdx);
                   if ( item.first.overlap(target) ) {
                        objs->push_back(item.second); 
                   } 
               }
               (*result)=*objs;
               //std::cout << _addressDependencyVector.size() << "," << target.getSize() << "\n";
               
               //return *objs;
                if (target.getSize() <= _sizeThresholdSmall ) {
                    if ( hasToInsert ){
                       _addressDependencySmall.insert( std::make_pair( target, currStatus ) );
                    }
                }

                if (target.getSize() > _sizeThresholdSmall && target.getSize() <= _sizeThresholdMid ) {
                    if ( hasToInsert ){
                       _addressDependencyMid.insert( std::make_pair( target, currStatus ) );
                    }
                }
                
                {
                    
                    //result contains the matches with the "big "arrays"
                    //Now deal with the small ones
                    //All the addresses starting lower than startAddr-threshold do not collide
                    void* startAddress=(void*)(((uint64_t)target.getAddress())-(_sizeThresholdSmall+1));
                    DepsRegion startAddr(startAddress,startAddress);
                    //All the addresses starting higher than endAddr do not collide
                    DepsRegion finalAddr(target.getEndAddress(),target.getEndAddress());
                    SizesMap::iterator startingIter=_addressDependencySmall.lower_bound(startAddr);
                    SizesMap::iterator endIter=_addressDependencySmall.upper_bound(finalAddr);
                    for ( ; startingIter!=_addressDependencySmall.end() && startingIter != endIter; ++startingIter ) {
                        if ( startingIter->first.overlap(target) ) {
                             result->push_back(startingIter->second); 
                        } 
                    }
                }

                {
                    //result contains the matches with the "small and big "arrays"
                    //Now deal with the mid ones
                    //All the addresses starting lower than startAddr-threshold do not collide
                    void* startAddress=(void*)(((uint64_t)target.getAddress())-(_sizeThresholdMid+1));
                    DepsRegion startAddr(startAddress,startAddress);
                    //All the addresses starting higher than endAddr do not collide
                    DepsRegion finalAddr(target.getEndAddress(),target.getEndAddress());
                    SizesMap::iterator startingIter=_addressDependencyMid.lower_bound(startAddr);
                    SizesMap::iterator endIter=_addressDependencyMid.upper_bound(finalAddr);
                    for ( ; startingIter!=_addressDependencyMid.end() && startingIter != endIter; ++startingIter ) {
                        if ( startingIter->first.overlap(target) ) {
                             result->push_back(startingIter->second); 
                        } 
                    }
                }
            }
            /**
             * Iterate over addresses... this method is rarely used
             * @param target
             * @return 
             */
            DepsVector::iterator findAddrInAddressDependencyMap( const DepsRegion& target ) {   
               for ( DepsVector::iterator it = _addressDependencyVector.begin(); it != _addressDependencyVector.end(); ++it ) {
                   if (it->first==target) return it;
               }
               return _addressDependencyVector.end();
            }
            
         protected:
            /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
             *  \param depObj DependableObject to be added to the domain.
             *  \param begin Iterator to the start of the list of dependencies to be associated to the Dependable Object.
             *  \param end Iterator to the end of the mentioned list.
             *  \param callback A function to call when a WD has a successor [Optional].
             *  \sa Dependency DependableObject TrackableObject
             */
            template<typename iterator>
            void submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end, SchedulePolicySuccessorFunctor* callback )
            {
               depObj.setId ( _lastDepObjId++ );
               depObj.init();
               depObj.setDependenciesDomain( this );
            
               // Object is not ready to get its dependencies satisfied
               // so we increase the number of predecessors to permit other dependableObjects to free some of
               // its dependencies without triggering the "dependenciesSatisfied" method
               depObj.increasePredecessors();
            
               std::list<DataAccess *> filteredDeps;
               for ( iterator it = begin; it != end; it++ ) {
                  DataAccess& newDep = (*it);

                  // if address == NULL, just ignore it
                  if ( newDep.getDepAddress() == NULL ) continue;
                  
                  bool found = false;
                  // For every dependency processed earlier
                  for ( std::list<DataAccess *>::iterator current = filteredDeps.begin(); current != filteredDeps.end(); current++ ) {
                     DataAccess* currentDep = *current;                     
                     if ( newDep.getDepAddress()  == currentDep->getDepAddress() && newDep.getSize()  == currentDep->getSize() ) {
                        // Both dependencies use the same address, put them in common
                        currentDep->setInput( newDep.isInput() || currentDep->isInput() );
                        currentDep->setOutput( newDep.isOutput() || currentDep->isOutput() );
                        found = true;
                        break;
                     }
                  }

                  if ( !found ) filteredDeps.push_back(&newDep);
               }
               
               // This list is needed for waiting
               std::list<uint64_t> flushDeps;
               
               TR1::unordered_map<TrackableObject*, bool> statusMap; /**< Tracks dependencies so we 
                                                                                  * do not submit dependencies with our same task */
               
               for ( std::list<DataAccess *>::iterator it = filteredDeps.begin(); it != filteredDeps.end(); it++ ) {
                  DataAccess &dep = *(*it);
                  
                  DepsRegion target( dep.getDepAddress(), (void*)((uint64_t)dep.getDepAddress()+dep.getSize()-1));
                  AccessType const &accessType = dep.flags;
                  
                  submitDependableObjectDataAccess( depObj, target, accessType, callback, statusMap );
                  flushDeps.push_back( (uint64_t) target() );
               }
               sys.getDefaultSchedulePolicy()->atCreate( depObj );              
 
               // To keep the count consistent we have to increase the number of tasks in the graph before releasing the fake dependency
               increaseTasksInGraph();
            
               depObj.submitted();
            
               // now everything is ready
               depObj.decreasePredecessors( &flushDeps, NULL, false, true );
            }
            /*! \brief Adds a region access of a DependableObject to the domains dependency system.
             *  \param depObj target DependableObject
             *  \param target accessed memory address
             *  \param accessType kind of region access
             *  \param callback Function to call if an immediate predecessor is found.
             */
            void submitDependableObjectDataAccess( DependableObject &depObj, DepsRegion &target, AccessType const &accessType, SchedulePolicySuccessorFunctor* callback, TR1::unordered_map<TrackableObject*, bool>& statusMap )
            {
               SyncRecursiveLockBlock lock1( getInstanceLock() );
               
               if ( accessType.concurrent || accessType.commutative ) {
                  if ( !( accessType.input && accessType.output ) || depObj.waits() ) {
                     fatal( "Commutation/concurrent task must be inout" );
                  }
               }
               
               if ( accessType.concurrent && accessType.commutative ) {
                  fatal( "Task cannot be concurrent AND commutative" ); 
               }
                
               std::vector<TrackableObject*> objs;
               lookupDependency( target, &objs );
               std::vector<TrackableObject*>::iterator it= objs.begin();
               TrackableObject &status = *(*it);
               target.setTrackable(&status);
               
               //Adding as reader/writer in my "own" status (pos 0)
               if ( accessType.concurrent || accessType.commutative ) {
                  submitDependableObjectCommutativeDataAccess( depObj, target, accessType, status, callback );
               } else if ( accessType.input && accessType.output ) {
                  submitDependableObjectInoutDataAccess( depObj, target, accessType, status, callback );
                  statusMap.insert( std::make_pair( &status, true ) );
               } else if ( accessType.input ) {
                  submitDependableObjectInputDataAccess( depObj, target, accessType, status, callback );
                  statusMap.insert( std::make_pair( &status, false ) );
               } else if ( accessType.output ) {
                  submitDependableObjectOutputDataAccess( depObj, target, accessType, status, callback );
                  statusMap.insert( std::make_pair( &status, true ) );
               } else {
                  fatal( "Invalid data access" );
               }
               
               
               ++it;
               //Now add every other access as "input"
               for ( ; it != objs.end(); ++it) {
                    TrackableObject &stat = *(*it);
                    TR1::unordered_map<TrackableObject*, bool>::iterator iterStat=statusMap.find (&stat);
                    if ( iterStat == statusMap.end() ){  
                        if ( accessType.output && !accessType.concurrent && !accessType.commutative ) {
                           submitDependableObjectOutputNoWriteDataAccess( depObj, target, accessType, stat, callback );    
                        } 
                        if ( accessType.input && !accessType.concurrent && !accessType.commutative ) {
                           submitDependableObjectInputNoReadDataAccess( depObj, target, accessType, stat, callback );  
                        }
                    } else {
                        bool isWriter=iterStat->second;
                        //This region was previously marked as "input" in this task, but our current writer 
                        //has to wait for it until all readers finish, reorder dependencies so we do not depend on ourselves
                        if (!isWriter && accessType.output && !accessType.concurrent && !accessType.commutative ) {
                            stat.getReadersLock().acquire();
                            stat.deleteReader(depObj);
                            stat.getReadersLock().release();
                            //depObj.decreasePredecessors(NULL, true);
                            submitDependableObjectOutputNoWriteDataAccess( depObj, target, accessType, stat, callback ); 
                            submitDependableObjectInputDataAccess( depObj, target, accessType, stat, callback );  
                            //Now we wait for all the possible restrictions on this Trackable, set it as true
                            iterStat->second=true;
                        }
                    } 
               }

                if ( !depObj.waits() && !accessType.concurrent && !accessType.commutative ) {
                   if ( accessType.output ) {
                      depObj.addWriteTarget( target );
                   } else if (accessType.input ) {
                      depObj.addReadTarget( target ); 
                   }
                }
            }
            
            inline void deleteLastWriter ( DependableObject &depObj, BaseDependency const &target )
            {
               const DepsRegion& address( static_cast<const DepsRegion&>( target ) );    
               
               TrackableObject &status = *address.getTrackable();
               status.deleteLastWriter(depObj);
            }
            
            
            inline void deleteReader ( DependableObject &depObj, BaseDependency const &target )
            {
               const DepsRegion& address( static_cast<const DepsRegion&>( target ) );
               TrackableObject &status = *address.getTrackable();
               {
                  SyncLockBlock lock2( status.getReadersLock() );
                  status.deleteReader(depObj);
               }
            }
            
            inline void removeCommDO ( CommutationDO *commDO, BaseDependency const &target )
            {
               const DepsRegion& address( static_cast<const DepsRegion&>( target ) );
               TrackableObject &status = *address.getTrackable();
                  
               if ( status.getCommDO ( ) == commDO ) {
                  status.setCommDO ( 0 );
               }
            }

         public: 
            CRegionsDependenciesDomain() : BaseDependenciesDomain(), _addressDependencyVector(), _addressDependencyCache(),
                    _addressDependencySmall(), _addressDependencyMid() ,_sizeThresholdSmall( 500 ),
            _sizeThresholdMid( 200000 ){
                _addressDependencyVector.reserve(5000);
            }
            CRegionsDependenciesDomain ( const CRegionsDependenciesDomain &depDomain )
               : BaseDependenciesDomain( depDomain ),
               _addressDependencyVector ( depDomain._addressDependencyVector ), _addressDependencyCache( depDomain._addressDependencyCache ),
               _addressDependencySmall( depDomain._addressDependencySmall ), _addressDependencyMid( depDomain._addressDependencyMid ),
               _sizeThresholdSmall ( depDomain._sizeThresholdSmall ),_sizeThresholdMid ( depDomain._sizeThresholdMid ) {}
            
            ~CRegionsDependenciesDomain()
            {
               for ( DepsCacheMap::iterator it = _addressDependencyCache.begin(); it != _addressDependencyCache.end(); it++ ) {
                  delete it->second->getTrackableObjs();
                  delete it->second;
               }
               for ( DepsVector::iterator it = _addressDependencyVector.begin(); it != _addressDependencyVector.end(); it++ ) {
                  delete it->second;
               }
            }
            
            /*!
             *  \note This function cannot be implemented in
             *  BaseDependenciesDomain since it calls a template function,
             *  and they cannot be virtual.
             */
            inline void submitDependableObject ( DependableObject &depObj, std::vector<DataAccess> &deps, SchedulePolicySuccessorFunctor* callback )
            {
               submitDependableObjectInternal ( depObj, deps.begin(), deps.end(), callback );
            }
            
            /*!
             *  \note This function cannot be implemented in
             *  BaseDependenciesDomain since it calls a template function,
             *  and they cannot be virtual.
             */
            inline void submitDependableObject ( DependableObject &depObj, size_t numDeps, DataAccess* deps, SchedulePolicySuccessorFunctor* callback )
            {
               submitDependableObjectInternal ( depObj, deps, deps+numDeps, callback );
            }

            bool haveDependencePendantWrites ( void *addr )
            {
               SyncRecursiveLockBlock lock1( getInstanceLock() );                
               DepsRegion address( addr, addr );
               DepsVector::iterator it = findAddrInAddressDependencyMap( address );
               if ( it == _addressDependencyVector.end() ) {
                  return false;
               } else {
                  TrackableObject* status = it->second;
                  DependableObject *lastWriter = status->getLastWriter();
                  return (lastWriter != NULL);
               }
            }
            
         
      };
      
      template void CRegionsDependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, DataAccess* begin, DataAccess* end, SchedulePolicySuccessorFunctor* callback );
      template void CRegionsDependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, std::vector<DataAccess>::iterator begin, std::vector<DataAccess>::iterator end, SchedulePolicySuccessorFunctor* callback );
      
      /*! \brief Default plugin implementation.
       */
      class CRegionsDependenciesManager : public DependenciesManager
      {
         public:
            CRegionsDependenciesManager() : DependenciesManager("Nanos plain dependencies domain") {}
            virtual ~CRegionsDependenciesManager () {}
            
            /*! \brief Creates a default dependencies domain.
             */
            DependenciesDomain* createDependenciesDomain () const
            {
               return NEW CRegionsDependenciesDomain();
            }
      };
  
      class NanosDepsPlugin : public Plugin
      {
            
         public:
            NanosDepsPlugin() : Plugin( "Nanos++ plain dependencies management plugin",1 )
            {
            }

            virtual void config ( Config &cfg )
            {
            }

            virtual void init()
            {
               sys.setDependenciesManager(NEW CRegionsDependenciesManager());
            }
      };

   }
}

DECLARE_PLUGIN("deps-cregions",nanos::ext::NanosDepsPlugin);
