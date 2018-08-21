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

#ifndef REGIONDICT_H
#define REGIONDICT_H

#include "regiondict_decl.hpp"
#include "atomic.hpp"
#include "lock.hpp"
#include "memorymap.hpp"
#include "system_decl.hpp"
#include "os.hpp"

namespace nanos {


inline RegionVectorEntry::RegionVectorEntry() : _leaf( NULL ), _data( NULL ) {
}

inline RegionVectorEntry::RegionVectorEntry( RegionVectorEntry const &rve ) : _leaf( rve._leaf ), _data( rve._data ) {
}

inline RegionVectorEntry &RegionVectorEntry::operator=( RegionVectorEntry const &rve ) {
   _leaf = rve._leaf;
   _data = rve._data;
   return *this;
}

inline RegionVectorEntry::~RegionVectorEntry() {
}

inline void RegionVectorEntry::setLeaf( RegionNode *rn ) {
   _leaf = rn;
}

inline RegionNode *RegionVectorEntry::getLeaf() const {
   return _leaf;
}

inline void RegionVectorEntry::setData( Version *d ) {
   _data = d;
}

inline Version *RegionVectorEntry::getData() const {
   return _data;
}

inline RegionNode::RegionNode( RegionNode *parent, std::size_t value, reg_t id ) : _parent( parent ), _value( value ), _id( id ), _sons( NULL ) {
   if ( id > 1 ) {
      _memoIntersectInfo = NEW reg_t[ id - 1 ];
      ::memset(_memoIntersectInfo, 0, sizeof( reg_t ) * (id - 1) );
   } else {
      _memoIntersectInfo = NULL;
   }
   //std::cerr << "created node with value " << value << " and id " << id << std::endl;
}

inline RegionNode::RegionNode( RegionNode const& rn ) : _parent( rn._parent ),
   _value( rn._value ), _id( rn._id ), _sons( rn._sons ),
   _memoIntersectInfo( rn._memoIntersectInfo ) {
}

inline RegionNode &RegionNode::operator=( RegionNode const& rn ) {
   _parent = rn._parent;
   _value = rn._value;
   _id = rn._id;
   _sons = rn._sons;
   _memoIntersectInfo = rn._memoIntersectInfo;
   return *this;
}

inline RegionNode::~RegionNode() {
   if ( _sons != NULL ) {
      for (std::map<std::size_t, RegionNode *>::const_iterator it = _sons->begin();
            it != _sons->end(); it++ ) {
         delete it->second;
      }
      delete _sons;
   }
   delete[] _memoIntersectInfo;
   _memoIntersectInfo = NULL;
}

inline reg_t RegionNode::getId() const {
   return _id;
}

inline reg_t RegionNode::getMemoIntersect( reg_t target ) const {
   return _memoIntersectInfo[ target-1 ];
}

inline void RegionNode::setMemoIntersect( reg_t target, reg_t value ) const {
   _memoIntersectInfo[ target-1 ] = value;
}

inline std::size_t RegionNode::getValue() const {
   return _value;
}

inline RegionNode *RegionNode::getParent() const {
   return _parent;
}


template <class T>
ContainerDense< T >::ContainerDense( CopyData const &cd ) : _container(64, T())
	, _leafCount( 0 )
	, _idSeed( 1 )
	, _dimensionSizes( cd.getNumDimensions(), 0 )
	, _root( NULL, 0, 0 )
	, _invalidationsLock()
	, _masterIdToLocalId()
	, _containerMi2LiLock()
	, _keepAtOrigin( false )
	, _registeredObject( NULL )
	, sparse( false ) {
   //_container.reserve( 64 );
   for ( unsigned int idx = 0; idx < cd.getNumDimensions(); idx += 1 ) {
      _dimensionSizes[ idx ] = cd.getDimensions()[ idx ].size;
   }
   if ( pthread_rwlock_init( &_containerLock, NULL ) ) {
      message0("error initializing containerlock ");
      fatal("can not continue")
   }
}

template <class T>
ContainerDense< T >::~ContainerDense() {
}

template <class T>
RegionNode * ContainerDense< T >::getRegionNode( reg_t id ) {
   RegionNode *n = NULL;
   if ( pthread_rwlock_rdlock(&_containerLock) ) {
      message0("lock error " );
      fatal("can not continue");
   }
   n = _container[ id ].getLeaf();
   if ( pthread_rwlock_unlock(&_containerLock) ) {
      message0("lock error " );
      fatal("can not continue");
   }
   return n;
}

template <class T>
void ContainerDense< T >::addRegionNode( RegionNode *leaf ) {
   // no locking needed, only called from addRegion -> _root.addNode() -> addRegionNode
   _container[ leaf->getId() ].setLeaf( leaf );
   _container[ leaf->getId() ].setData( NULL );
   _leafCount++;
}

template <class T>
Version *ContainerDense< T >::getRegionData( reg_t id ) {
   Version *v = NULL;
   //_containerLock.acquire();
   //while ( !_containerLock.tryAcquire() ) {
   //   myThread->idle();
   //}
   //std::cerr << "acquired @ " << __func__ << std::endl;
   if ( pthread_rwlock_rdlock(&_containerLock) ) {
      message0("lock error ");
      fatal("can not continue");
   }
   v = _container[ id ].getData();
   //std::cerr << "released @ " << __func__ << std::endl;
   //_containerLock.release();
   if ( pthread_rwlock_unlock(&_containerLock) ) {
      message0("lock error ");
      fatal("can not continue");
   }
   return v;
}

template <class T>
void ContainerDense< T >::setRegionData( reg_t id, Version *data ) {
   //_containerLock.acquire();
   //while ( !_containerLock.tryAcquire() ) {
   //   myThread->idle();
   //}
   //std::cerr << "acquired @ " << __func__ << std::endl;
   if ( pthread_rwlock_rdlock(&_containerLock) ) {
      message0("lock error ");
      fatal("can not continue");
   }
   _container[ id ].setData( data );
   //std::cerr << "released @ " << __func__ << std::endl;
   if ( pthread_rwlock_unlock(&_containerLock) ) {
      message0("lock error ");
      fatal("can not continue");
   }
   //_containerLock.release();
}

template <class T>
unsigned int ContainerDense< T >::getRegionNodeCount() const {
   return _leafCount.value();
}

template <class T>
unsigned int ContainerDense< T >::getNumDimensions() const {
   return _dimensionSizes.size();
}

template <class T>
reg_t ContainerDense< T >::addRegion( nanos_region_dimension_internal_t const region[] ) {
   if ( pthread_rwlock_wrlock(&_containerLock) ) {
      message0("lock error " );
      fatal("can not continue");
   }

   reg_t id = _root.addNode( region, _dimensionSizes.size(), 0, *this );

   if ( pthread_rwlock_unlock(&_containerLock) ) {
      message0("lock error " );
      fatal("can not continue");
   }

   return id;
}

template <class T>
reg_t ContainerDense< T >::getNewRegionId() {
   reg_t id = _idSeed++;
   if ( id % 64 == 0 ) {
      _container.resize( id + 64 );
   }
   if (id >= MAX_REG_ID) { std::cerr <<"Max regions reached."<<std::endl;}
   return id;
}

template <class T>
reg_t ContainerDense< T >::checkIfRegionExists( nanos_region_dimension_internal_t const region[] ) {
   bool result;
   if ( pthread_rwlock_rdlock(&_containerLock) ) {
      message0("lock error ");
      fatal("can not continue");
   }
   result = _root.checkNode( region, _dimensionSizes.size(), 0 );
   if ( pthread_rwlock_unlock(&_containerLock) ) {
      message0("lock error " );
      fatal("can not continue");
   }
   return result;
}

template <class T>
std::vector< std::size_t > const &ContainerDense< T >::getDimensionSizes() const {
   return _dimensionSizes;
}

template <class T>
reg_t ContainerDense< T >::getMaxRegionId() const {
   return _idSeed.value();
}

template <class T>
void ContainerDense< T >::invalLock() {
   while ( !_invalidationsLock.tryAcquire() ) {
      myThread->processTransfers();
   }
}

template <class T>
void ContainerDense< T >::invalUnlock() {
   return _invalidationsLock.release();
}

template <class T>
void ContainerDense< T >::addMasterRegionId( reg_t masterId, reg_t localId ) {
   _containerMi2LiLock.acquire();
   _masterIdToLocalId[ masterId ] = localId;
   _containerMi2LiLock.release();
}

template <class T>
reg_t ContainerDense< T >::getLocalRegionIdFromMasterRegionId( reg_t masterId ) {
   reg_t result = 0;
   _containerMi2LiLock.acquire();
   std::map< reg_t, reg_t >::const_iterator it = _masterIdToLocalId.find( masterId );
   if ( it != _masterIdToLocalId.end() ) {
      result = it->second;
   }
   _containerMi2LiLock.release();
   return result;
}

template <class T>
void ContainerDense< T >::setKeepAtOrigin( bool value ) {
   _keepAtOrigin = value;
}

template <class T>
bool ContainerDense< T >::getKeepAtOrigin() const {
   return _keepAtOrigin;
}

template <class T>
void ContainerDense< T >::setRegisteredObject( CopyData *cd ) {
   _registeredObject = cd;
}

template <class T>
CopyData *ContainerDense< T >::getRegisteredObject() const {
   return _registeredObject;
}

template <class T>
ContainerSparse< T >::ContainerSparse( RegionDictionary< ContainerDense > &orig ) : _container(), _containerLock(), _orig( orig ), sparse( true ) {
}


template <class T>
ContainerSparse< T >::~ContainerSparse() {
}

template <class T>
RegionNode * ContainerSparse< T >::getRegionNode( reg_t id ) const {
   std::map< reg_t, RegionVectorEntry >::const_iterator it = _container.lower_bound( id );
   if ( it == _container.end() || _container.key_comp()(id, it->first) ) {
     RegionNode *leaf = _orig.getRegionNode( id );
   //if ( leaf == NULL ) { *(myThread->_file) << "NULL LEAF CHECK by orig: " << std::endl; printBt( *(myThread->_file) ); }
      return leaf;
   }
   return it->second.getLeaf();
}

template <class T>
Version *ContainerSparse< T >::getRegionData( reg_t id ) {
   //_containerLock.acquire();
   while ( !_containerLock.tryAcquire() ) {
      myThread->processTransfers();
   }
   std::map< reg_t, RegionVectorEntry >::iterator it = _container.lower_bound( id );
   if ( it == _container.end() || _container.key_comp()(id, it->first) ) {
      it = _container.insert( it, std::map< reg_t, RegionVectorEntry >::value_type( id, RegionVectorEntry() ) );
      it->second.setLeaf( _orig.getRegionNode( id ) );
      _containerLock.release();
      return NULL;
   }
   _containerLock.release();
   return it->second.getData();
}

template <class T>
void ContainerSparse< T >::setRegionData( reg_t id, Version *data ) {
   if ( _container[ id ].getLeaf() == NULL ) {
   std::cerr << "WARNING: null leaf, region "<< id << " region node is " << (void*) _orig.getRegionNode( id ) << " addr orig " << (void*)&_orig <<std::endl;
      _container[ id ].setLeaf( _orig.getRegionNode( id ) );
   }
   _container[ id ].setData( data );
}

template <class T>
unsigned int ContainerSparse< T >::getRegionNodeCount() const {
   return _container.size();
}

template <class T>
reg_t ContainerSparse< T >::addRegion( nanos_region_dimension_internal_t const region[] ) {
   reg_t id = _orig.addRegion( region );
   if ( sys.getNetwork()->getNodeNum() > 0) { std::cerr << " ADDED REG " << id << std::endl; }
   return id;
}

template <class T>
unsigned int ContainerSparse< T >::getNumDimensions() const {
   return _orig.getNumDimensions();
}

template <class T>
reg_t ContainerSparse< T >::checkIfRegionExists( nanos_region_dimension_internal_t const region[] ) {
   return _orig.checkIfRegionExists( region );
}

template <class T>
typename std::map< reg_t, T >::const_iterator ContainerSparse< T >::begin() {
   return _container.begin();
}

template <class T>
typename std::map< reg_t, T >::const_iterator ContainerSparse< T >::end() {
   return _container.end();
}

template <class T>
ContainerDense< T > &ContainerSparse< T >::getOrigContainer() {
   return _orig;
}

template <class T>
RegionNode *ContainerSparse< T >::getGlobalRegionNode( reg_t id ) const {
   return _orig.getRegionNode( id );
}

template <class T>
Version *ContainerSparse< T >::getGlobalRegionData( reg_t id ) {
   return _orig.getRegionData( id );
}

template <class T>
RegionDictionary< ContainerDense > *ContainerSparse< T >::getGlobalDirectoryKey() {
   return &_orig;
}

template <class T>
std::vector< std::size_t > const &ContainerSparse< T >::getDimensionSizes() const {
   return _orig.getDimensionSizes();
}

template <class T>
reg_t ContainerSparse< T >::getMaxRegionId() const {
   return _orig.getMaxRegionId();
}

template < template <class> class Sparsity>
RegionDictionary< Sparsity >::RegionDictionary( CopyData const &cd ) : 
      Sparsity< RegionVectorEntry >( cd ), Version( 1 ),
      _intersects( cd.getNumDimensions(), MemoryMap< std::set< reg_t > >() ),
      _keyBaseAddress( cd.getHostBaseAddress() == 0 ? ( (uint64_t) cd.getBaseAddress() ) : cd.getHostBaseAddress() ),
      _realBaseAddress( (uint64_t) cd.getBaseAddress() ), _lock() {
   //std::cerr << "CREATING MASTER DICT: tree: " << (void *) &_tree << std::endl;
   nanos_region_dimension_internal_t dims[ cd.getNumDimensions() ];
   for ( unsigned int idx = 0; idx < cd.getNumDimensions(); idx++ ) {
      dims[ idx ].size  = cd.getDimensions()[ idx ].size;
      dims[ idx ].accessed_length = cd.getDimensions()[ idx ].size;
      dims[ idx ].lower_bound = 0;
   }
   reg_t id = this->addRegion( dims );
   //std::list< std::pair< reg_t, reg_t > > missingParts;
   //unsigned int version;
   ensure( id == 1, "Whole region did not get id 1");
   (void) id;
}

template < template <class> class Sparsity>
RegionDictionary< Sparsity >::RegionDictionary( GlobalRegionDictionary &dict ) : Sparsity< RegionVectorEntry >( dict ), _intersects( dict.getNumDimensions(), MemoryMap< std::set< reg_t > >() ),
      _keyBaseAddress( dict.getKeyBaseAddress() ), _realBaseAddress( dict.getRealBaseAddress() ), _lock(), _fixedRegions() {
   //std::cerr << "CREATING CACHE DICT: tree: " << (void *) &_tree << " orig tree: " << (void *) &dict._tree << std::endl;
         //std::cerr << "Created dir " << (void *) this << " w/dims " << dict.getNumDimensions() << std::endl;
}

template < template <class> class Sparsity>
RegionDictionary< Sparsity >::~RegionDictionary() {
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::lockObject() {
   //_lock.acquire();
   while ( !_lock.tryAcquire() ) {
      myThread->processTransfers();
   }
}

template < template <class> class Sparsity>
bool RegionDictionary< Sparsity >::tryLockObject() {
   return _lock.tryAcquire();
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::unlockObject() {
   _lock.release();
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::_computeIntersect( reg_t regionIdA, reg_t regionIdB, nanos_region_dimension_internal_t *outReg ) {
   RegionNode const *regA = this->getRegionNode( regionIdA );
   RegionNode const *regB = this->getRegionNode( regionIdB );

   if ( regionIdA == regionIdB ) {
      *(myThread->_file) << __FUNCTION__ << " Dummy check! regA == regB ( " << regionIdA << " )" << std::endl;
      printBt( *(myThread->_file) );
      for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {
         outReg[ dimensionCount ].accessed_length = 0;
         outReg[ dimensionCount ].lower_bound = 0;
      }
      return;
   }

   //reg_t maxReg = std::max( regionIdA, regionIdB );
   //if ( regionIdA > regionIdB ) {
   //} else {
   //}

   //std::cerr << "Computing intersect between reg " << regionIdA << " and "<< regionIdB << "... "<<std::endl;
   //printRegion(regionIdA ); std::cerr << std::endl;
   //printRegion(regionIdB ); std::cerr << std::endl;

   for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {
      std::size_t accessedLengthA = regA->getValue();
      std::size_t accessedLengthB = regB->getValue();
      regA = regA->getParent();
      regB = regB->getParent();
      std::size_t lowerBoundA = regA->getValue();
      std::size_t lowerBoundB = regB->getValue();

      std::size_t upperBoundA = lowerBoundA + accessedLengthA;
      std::size_t upperBoundB = lowerBoundB + accessedLengthB;

      std::size_t lowerBoundC = 0;
      std::size_t accessedLengthC = 0;

      if ( lowerBoundA > lowerBoundB ) {
         lowerBoundC = lowerBoundA;
         if ( upperBoundA > upperBoundB ) {
            accessedLengthC = upperBoundB - lowerBoundA;
         } else if ( upperBoundA <= upperBoundB ) {
            accessedLengthC = accessedLengthA;
         } 
      } else if ( lowerBoundA < lowerBoundB ) {
         lowerBoundC = lowerBoundB;
         if ( upperBoundA >= upperBoundB ) {
            accessedLengthC = accessedLengthB;
         } else if ( upperBoundA < upperBoundB ) {
            accessedLengthC = upperBoundA - lowerBoundB;
         }
      } else {
         lowerBoundC = lowerBoundA;
         if ( upperBoundA > upperBoundB ) {
            accessedLengthC = accessedLengthB;
         } else if ( upperBoundA <= upperBoundB ) {
            accessedLengthC = accessedLengthA;
         } 
      }

      outReg[ dimensionCount ].accessed_length = accessedLengthC;
      outReg[ dimensionCount ].lower_bound = lowerBoundC;

      regA = regA->getParent();
      regB = regB->getParent();
   }
}

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::computeTestIntersect( reg_t regionIdA, reg_t regionIdB ) {
   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode const *maxReg = this->getRegionNode( maxRegionId );
      reg_t data = maxReg->getMemoIntersect( minRegionId );
      if ( data == (unsigned int)-1 ) {
         //std::cerr << "hit compute!"<<std::endl;
         return 0;
      } else if ( data != (unsigned int)-2 && data != 0 ) {
         return data;
      }
   }

   if ( !this->checkIntersect( regionIdA, regionIdB ) ) {
      std::cerr << " Regions do not intersect: " << regionIdA << ", " << regionIdB << std::endl;
   }
   ensure( this->checkIntersect( regionIdA, regionIdB ) ," Regions do not intersect." );

   nanos_region_dimension_internal_t resultingRegion[ this->getNumDimensions() ];
   _computeIntersect( regionIdA, regionIdB, resultingRegion );
   reg_t regId = this->checkIfRegionExists( resultingRegion );

   return regId;
}

template < template <class> class Sparsity>
reg_t RegionDictionary<Sparsity>::computeIntersect( reg_t regionIdA, reg_t regionIdB ) {
   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode const *maxReg = this->getRegionNode( maxRegionId );
      reg_t data = maxReg->getMemoIntersect( minRegionId );
      if ( data == (unsigned int)-1 ) {
         //std::cerr << "hit compute!"<<std::endl;
         return 0;
      } else if ( data != (unsigned int)-2 && data != 0 ) {
         return data;
      }
   }
   if ( !this->checkIntersect( regionIdA, regionIdB ) ) {
      std::cerr << " Regions do not intersect: " << regionIdA << ", " << regionIdB << std::endl;
   }
   ensure( this->checkIntersect( regionIdA, regionIdB ) ," Regions do not intersect." );
   nanos_region_dimension_internal_t resultingRegion[ this->getNumDimensions() ];
   _computeIntersect( regionIdA, regionIdB, resultingRegion );
   reg_t regId = this->addRegion( resultingRegion );

   //std::cerr << (void *) this << " Computed intersect bewteen " << regionIdA << " and " << regionIdB << " resulting region is "<< regId << std::endl;

   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode *maxReg = this->getRegionNode( maxRegionId );
      maxReg->setMemoIntersect( minRegionId, regId );
   }

   return regId;
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::addRegionAndComputeIntersects( reg_t id, std::list< std::pair< reg_t, reg_t > > &finalParts, unsigned int &version ) {
	class LocalFunction {
		RegionDictionary &_currentDict;
		public:
		LocalFunction( RegionDictionary &dict ) : _currentDict( dict ) { } 

		void _recursive ( reg_t this_id, unsigned int total_dims, unsigned int dim, MemoryMap< std::set< reg_t > >::MemChunkList results[], nanos_region_dimension_internal_t current_regions[], std::set<reg_t> *current_sets[], std::map<reg_t, std::set< reg_t > > &resulting_regions, bool compute_region ) {
			if ( dim == 0 ) {
				for ( MemoryMap< std::set< reg_t > >::MemChunkList::const_iterator it = results[ dim ].begin(); it != results[ dim ].end(); it++ ) {
					if ( *(it->second) == NULL ) {
						*(it->second) = NEW std::set< reg_t >();
					}
					current_regions[dim].lower_bound = it->first->getAddress();
					current_regions[dim].accessed_length = it->first->getLength();
					current_sets[dim] = *it->second;
					bool this_compute_region = ( compute_region && current_sets[dim] != NULL && !current_sets[dim]->empty() );
					reg_t parent_reg = this_id;
					if ( this_compute_region ) {
						std::set<reg_t> metadata_regs;
						for ( std::set<reg_t>::const_iterator sit = current_sets[0]->begin(); sit != current_sets[0]->end(); sit++ ) {
							unsigned int count = 1;
							for ( unsigned int dim_idx = 1; dim_idx < total_dims; dim_idx += 1 ) {
								count += current_sets[dim_idx]->count(*sit);
							}
							if ( count == total_dims ) {
								metadata_regs.insert(*sit);
							}
						}
                  if ( this_id == 1 ) {
                     metadata_regs.insert( 1 );
                  }
						if ( metadata_regs.size() >= 1 ) {
							//  if ( metadata_regs.size() > 1 ) {
							//  	*myThread->_file << "Multiple regions can be the parent region: ";
							//  	for ( std::set<reg_t>::const_iterator sit = metadata_regs.begin(); sit != metadata_regs.end(); sit++ ) {
							//  		*myThread->_file << *sit << " ";
							//  	}
							//  	*myThread->_file << std::endl;
							//  }
							reg_t max_version_reg = 0;
							unsigned int current_version = 0;
							//bool has_own_region = false;
							unsigned int own_version = 0;
							for ( std::set<reg_t>::const_iterator sit = metadata_regs.begin(); sit != metadata_regs.end(); sit++ ) {
								Version *entry = _currentDict.getRegionData( *sit );
								unsigned int this_version = entry != NULL ? entry->getVersion() : 0;
								if (*sit == this_id) {
									//has_own_region = true;
									own_version = this_version;
								}
								if ( this_version > current_version ) {
									current_version = this_version;
									max_version_reg = *sit;
								}
							}
							// if ( metadata_regs.size() > 1 && max_version_reg != 0 ) {
							// 	*myThread->_file << "[w/id " << this_id << "] Selected region (by version: " << current_version <<") : " << max_version_reg << " own_version: " << own_version << std::endl;
							// 	for ( std::set<reg_t>::const_iterator sit = metadata_regs.begin(); sit != metadata_regs.end(); sit++ ) {
							// 		*myThread->_file << *sit << " ";
							//    }
							// 	*myThread->_file << std::endl;
							// }
							if ( own_version == current_version && max_version_reg != 0 ) {
								max_version_reg = this_id;
							}
							parent_reg = ( max_version_reg != 0 ) ? max_version_reg : parent_reg;
						}
					}
					reg_t part = _currentDict.addRegion( current_regions );
					resulting_regions[parent_reg].insert( part );
				}
			} else {
				for ( MemoryMap< std::set< reg_t > >::MemChunkList::const_iterator it = results[ dim ].begin(); it != results[ dim ].end(); it++ ) {
					if ( *(it->second) == NULL ) {
						*(it->second) = NEW std::set< reg_t >();
					}
					current_regions[dim].lower_bound = it->first->getAddress();
					current_regions[dim].accessed_length = it->first->getLength();
					current_sets[dim] = *it->second;
					bool this_compute_region = ( compute_region && current_sets[dim] != NULL && !current_sets[dim]->empty() );
					_recursive( this_id, total_dims, dim - 1, results, current_regions, current_sets, resulting_regions, this_compute_region );
				}
			}
		}

		void computeIntersections( reg_t this_id, std::map< reg_t, std::set< reg_t > > &resulting_regions ) {
			RegionNode const *regNode = _currentDict.getRegionNode( this_id );
			MemoryMap< std::set< reg_t > >::MemChunkList results[ _currentDict.getNumDimensions() ];

			for ( int idx = _currentDict.getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
				std::size_t accessedLength = regNode->getValue();
				regNode = regNode->getParent();
				std::size_t lowerBound = regNode->getValue();
				regNode = regNode->getParent();

				_currentDict._intersects[ idx ].getOrAddChunk( lowerBound, accessedLength, results[ idx ] );
			}

			std::set<reg_t> *current_sets[ _currentDict.getNumDimensions() ];
			nanos_region_dimension_internal_t current_regions[ _currentDict.getNumDimensions() ];

         if ( _currentDict.getNumDimensions() == 0 ) {
            std::cerr << "Invalid DICT: " << &_currentDict << std::endl;
         }
         ensure(_currentDict.getNumDimensions() > 0, "Invalid, object, 0 dimensions");
			_recursive( this_id, _currentDict.getNumDimensions(), _currentDict.getNumDimensions() - 1, results, current_regions, current_sets, resulting_regions, true );

			if ( this_id != 1 ) {
				for ( int idx = _currentDict.getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
					for ( MemoryMap< std::set< reg_t > >::MemChunkList::const_iterator it = results[ idx ].begin(); it != results[ idx ].end(); it++ ) {
						(*it->second)->insert( this_id );
					}
				}
			}
		}

	};
	LocalFunction local( *this );
	//double tiniCI2 = OS::getMonotonicTime();
	std::map< reg_t, std::set< reg_t > > resulting_regions;
	local.computeIntersections( id, resulting_regions );
	//double tfiniCI2 = OS::getMonotonicTime();
	//*myThread->_file << "CI2 time: " << (tfiniCI2 - tiniCI2) << std::endl;

	for ( std::map< reg_t, std::set< reg_t > >::const_iterator mit = resulting_regions.begin(); mit != resulting_regions.end(); mit++ ) {
		Version *entry = this->getRegionData( mit->first );
		unsigned int this_version = entry != NULL ? entry->getVersion() : ( this->sparse ? 0 : 1 );
		version = this_version > version ? this_version : version;
		for ( std::set< reg_t >::const_iterator sit = mit->second.begin(); sit != mit->second.end(); sit++ ) {
			finalParts.push_back( std::make_pair( *sit, mit->first ) );
		}
	}

}

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::obtainRegionId( CopyData &cd, WD const &wd, unsigned int idx ) {
   reg_t id = 0;
   CopyData *deductedCd = NULL;
   if ( this->getRegisteredObject() != NULL && !this->getRegisteredObject()->equalGeometry( cd ) ) {
      CopyData *tmp = NEW CopyData( *this->getRegisteredObject() );
      nanos_region_dimension_internal_t *dims = NEW nanos_region_dimension_internal_t[tmp->getNumDimensions()];
      cd.deductCd( *this->getRegisteredObject(), dims );
      tmp->setDimensions( dims );
      deductedCd = tmp;
      cd.setDeductedCD( tmp );
   }
   CopyData const &realCd = deductedCd != NULL ? *deductedCd : cd;
   if ( realCd.getNumDimensions() != this->getNumDimensions() ) {
     fatal("Error: cd.getNumDimensions() returns " << realCd.getNumDimensions()
         << " but I already have the object registered with " << this->getNumDimensions()
         << " dimensions. WD is : "
         << ( wd.getDescription() != NULL ? wd.getDescription() : "n/a" )
         << " copy index: " << idx << " got reg object? " << this->getRegisteredObject() );
   }
   ensure( realCd.getNumDimensions() == this->getNumDimensions(), "ERROR" );
   ensure( this->getNumDimensions() > 0, "ERROR" );
   if ( realCd.getNumDimensions() != this->getNumDimensions() ) {
      std::cerr << "Error, invalid numDimensions" << std::endl;
   } else {
      for ( unsigned int cidx = 0; cidx < realCd.getNumDimensions(); cidx += 1 ) {
         if ( this->getDimensionSizes()[ cidx ] != realCd.getDimensions()[ cidx ].size ) {
            printBt(*myThread->_file);
            fatal("Object with base address " << (void *)realCd.getBaseAddress() <<
                  " was previously registered with a different size in dimension " <<
                  std::dec << cidx << " (previously was " <<
                  std::dec << this->getDimensionSizes()[ cidx ] <<
                  " now received size " << std::dec <<
                  realCd.getDimensions()[ cidx ].size << "). WD: " <<
                  ( wd.getDescription() != NULL ? wd.getDescription() : "n/a") );
         }
      }
      id = this->addRegion( realCd.getDimensions() );
   }
   //this->printRegion(std::cerr, id); std::cerr << std::endl;
   return id;
}

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::obtainRegionId( nanos_region_dimension_internal_t const region[] ) {
   return this->addRegion( region );
}

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::registerRegion( reg_t id, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version ) {
   this->addRegionAndComputeIntersects( id, missingParts, version );
   return id;
}

template < template <class> class Sparsity>
bool RegionDictionary< Sparsity >::checkIntersect( reg_t regionIdA, reg_t regionIdB ) {
   if ( regionIdA == regionIdB ) {
      *(myThread->_file) << __FUNCTION__ << " Dummy check! regA == regB ( " << regionIdA << " )" << std::endl;
      printBt( *(myThread->_file) );
   }

   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode const *maxReg = this->getRegionNode( maxRegionId );
      reg_t data = maxReg->getMemoIntersect( minRegionId );
      if ( data != 0 ) {
         //std::cerr << "hit!"<<std::endl;
         return ( data != (unsigned int) -1 );
      }
   }

   RegionNode const *regA = this->getRegionNode( regionIdA );
   RegionNode const *regB = this->getRegionNode( regionIdB );
   bool result = true;

   //std::cerr << "Computing intersect between reg " << regionIdA << " and "<< regionIdB << std::endl;

   for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0 && result; dimensionCount -= 1 ) {
      std::size_t accessedLengthA = regA->getValue();
      std::size_t accessedLengthB = regB->getValue();
      regA = regA->getParent();
      regB = regB->getParent();
      std::size_t lowerBoundA = regA->getValue();
      std::size_t lowerBoundB = regB->getValue();

      std::size_t upperBoundA = lowerBoundA + accessedLengthA;
      std::size_t upperBoundB = lowerBoundB + accessedLengthB;

      if ( lowerBoundA > lowerBoundB ) {
          result = ( upperBoundB > lowerBoundA );
      } else if ( lowerBoundA < lowerBoundB ) {
          result = ( lowerBoundB < upperBoundA );
      } else {
          result = true;
      }

      regA = regA->getParent();
      regB = regB->getParent();
   }

   //std::cerr << "Computing intersect between reg " << regionIdA << " and "<< regionIdB << " " << result<< std::endl;

   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode *maxReg = this->getRegionNode( maxRegionId );
      maxReg->setMemoIntersect( minRegionId, result ? -2 : -1  );
   }
   

   return result;
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::substract( reg_t base, reg_t regionToSubstract, std::list< reg_t > &resultingPieces ) {

   //std::cerr << __FUNCTION__ << ": base("<< base << ") "; this->printRegion(base); std::cerr<< " regToSubs(" << regionToSubstract<< ") "; this->printRegion(regionToSubstract); std::cerr << std::endl;
   //std::cerr << __FUNCTION__ << ": base("<< base << ") regToSubs(" << regionToSubstract<< ") " << std::endl;
   if ( !checkIntersect( base, regionToSubstract ) ) {
      return;
   }

   nanos_region_dimension_internal_t fragments[ this->getNumDimensions() ][ 3 ];
   RegionNode const *regBase   = this->getRegionNode( base );
   RegionNode const *regToSubs = this->getRegionNode( regionToSubstract );

   for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {
      std::size_t accessedLengthBase = regBase->getValue();
      std::size_t accessedLengthToSubs = regToSubs->getValue();
      regBase = regBase->getParent();
      regToSubs = regToSubs->getParent();
      std::size_t lowerBoundBase = regBase->getValue();
      std::size_t lowerBoundToSubs = regToSubs->getValue();

      std::size_t upperBoundBase = lowerBoundBase + accessedLengthBase;
      std::size_t upperBoundToSubs = lowerBoundToSubs + accessedLengthToSubs;

      std::size_t lowerBoundLTIntersect = 0;
      std::size_t accessedLengthLTIntersect = 0;
      std::size_t lowerBoundIntersect = 0;
      std::size_t accessedLengthIntersect = 0;
      std::size_t lowerBoundGTIntersect = 0;
      std::size_t accessedLengthGTIntersect = 0;

      if ( lowerBoundBase >= lowerBoundToSubs ) {
         lowerBoundLTIntersect = 0;
         accessedLengthLTIntersect = 0;

         lowerBoundIntersect = lowerBoundBase;

         if ( upperBoundBase <= upperBoundToSubs ) {
            accessedLengthIntersect = upperBoundBase - lowerBoundBase;

            lowerBoundGTIntersect = 0;
            accessedLengthGTIntersect = 0;
         } else {
            accessedLengthIntersect = upperBoundToSubs - lowerBoundBase;

            lowerBoundGTIntersect = upperBoundToSubs;
            accessedLengthGTIntersect = upperBoundBase - upperBoundToSubs;
         }
      } else {
         lowerBoundLTIntersect = lowerBoundBase;
         accessedLengthLTIntersect = lowerBoundToSubs - lowerBoundBase;
         lowerBoundIntersect = lowerBoundToSubs;

         if ( upperBoundBase <= upperBoundToSubs ) {
            accessedLengthIntersect = upperBoundBase - lowerBoundToSubs;
            lowerBoundGTIntersect = 0;
            accessedLengthGTIntersect = 0;
         } else {
            accessedLengthIntersect = upperBoundToSubs - lowerBoundToSubs;
            lowerBoundGTIntersect = upperBoundToSubs;
            accessedLengthGTIntersect = upperBoundBase - upperBoundToSubs;
         }
      }

      //std::cerr << dimensionCount << " A: lb=" << lowerBoundBase << " al=" << accessedLengthBase << std::endl;
      //std::cerr << dimensionCount << " B: lb=" << lowerBoundToSubs << " al=" << accessedLengthToSubs << std::endl;
      //std::cerr << dimensionCount << " LT: lb=" << lowerBoundLTIntersect << " al=" << accessedLengthLTIntersect << std::endl;
      //std::cerr << dimensionCount << " EQ: lb=" << lowerBoundIntersect << " al=" << accessedLengthIntersect << std::endl;
      //std::cerr << dimensionCount << " GT: lb=" << lowerBoundGTIntersect << " al=" << accessedLengthGTIntersect << std::endl;

      fragments[ dimensionCount ][ 0 ].accessed_length = accessedLengthLTIntersect;
      fragments[ dimensionCount ][ 0 ].lower_bound     = lowerBoundLTIntersect;
      fragments[ dimensionCount ][ 1 ].accessed_length = accessedLengthIntersect;
      fragments[ dimensionCount ][ 1 ].lower_bound     = lowerBoundIntersect;
      fragments[ dimensionCount ][ 2 ].accessed_length = accessedLengthGTIntersect;
      fragments[ dimensionCount ][ 2 ].lower_bound     = lowerBoundGTIntersect;

      regBase = regBase->getParent();
      regToSubs = regToSubs->getParent();
   }

   nanos_region_dimension_internal_t tmpFragment[ this->getNumDimensions() ];

   _combine( tmpFragment, this->getNumDimensions()-1, 0, fragments, false, resultingPieces );
   _combine( tmpFragment, this->getNumDimensions()-1, 1, fragments, true, resultingPieces );
   _combine( tmpFragment, this->getNumDimensions()-1, 2, fragments, false, resultingPieces );


}
template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::_combine ( nanos_region_dimension_internal_t tmpFragment[], int dim, int currentPerm, nanos_region_dimension_internal_t fragments[ ][3], bool allFragmentsIntersect, std::list< reg_t > &resultingPieces  ) {
   //std::cerr << "dim "<<dim << " currentPerm " << currentPerm<<std::endl;;
   if ( fragments[ dim ][ currentPerm ].accessed_length > 0 ) { 
      tmpFragment[ dim ].accessed_length = fragments[ dim ][ currentPerm ].accessed_length;
      tmpFragment[ dim ].lower_bound     = fragments[ dim ][ currentPerm ].lower_bound;
      //for( unsigned int _i = 0; _i < ( this->getDimensionSizes().size() - (1+dim) ); _i++ ) std::cerr << ">";
      //std::cerr <<" [" << fragments[ dim ][ currentPerm ].lower_bound << ":" << fragments[ dim ][ currentPerm ].accessed_length << "] " << std::endl;

      if ( dim > 0 )  {
         //std::cerr << __FUNCTION__ << " dim: " << dim << " currentPerm: " << currentPerm << std::endl;
         _combine( tmpFragment, dim-1, 0, fragments, ( allFragmentsIntersect && false ), resultingPieces ); 
         //if ( currentPerm != 1 ) {
         //tmpFragment[ dim ].accessed_length = fragments[ dim ][ currentPerm ].accessed_length;
         //tmpFragment[ dim ].lower_bound     = fragments[ dim ][ currentPerm ].lower_bound;
         _combine( tmpFragment, dim-1, 1, fragments, ( allFragmentsIntersect && true ), resultingPieces ); 
         //}
         //tmpFragment[ dim ].accessed_length = fragments[ dim ][ currentPerm ].accessed_length;
         //tmpFragment[ dim ].lower_bound     = fragments[ dim ][ currentPerm ].lower_bound;
         _combine( tmpFragment, dim-1, 2, fragments, ( allFragmentsIntersect && false ), resultingPieces ); 
      } else {
         if ( !allFragmentsIntersect ) {
            //std::cerr << "not all intersect " << std::endl;
            //for ( unsigned int i = 0; i < this->getDimensionSizes().size(); i++ ) {
            //   std::cerr << "["<<tmpFragment[this->getDimensionSizes().size()-(i+1)].lower_bound<< ":" << tmpFragment[this->getDimensionSizes().size()-(i+1)].accessed_length << "]";
            //}
            //std::cerr << std::endl;
            reg_t id = this->addRegion( tmpFragment );
            //std::cerr << (void *) this <<" computed a subchunk " << id ;this->printRegion( id ); std::cerr << std::endl;
            resultingPieces.push_back( id );
         } else {
            //std::cerr << "ignored intersect region ";
            //for ( unsigned int i = 0; i < getDimensionSizes().size(); i++ ) {
            //   std::cerr << "["<<tmpFragment[getDimensionSizes().size()-(i+1)].lower_bound<< ":" << tmpFragment[getDimensionSizes().size()-(i+1)].accessed_length << "]";
            //}
            //std::cerr << std::endl;
         }
      }
   }
}  

template < template <class> class Sparsity>
uint64_t RegionDictionary< Sparsity >::getKeyBaseAddress() const {
   return _keyBaseAddress;
}

template < template <class> class Sparsity>
uint64_t RegionDictionary< Sparsity >::getRealBaseAddress() const {
   return _realBaseAddress;
}

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::isThisPartOf( reg_t target, std::map< reg_t, unsigned int >::const_iterator begin, std::map< reg_t, unsigned int >::const_iterator end, unsigned int &version ) {
   reg_t result = 0;
   while ( begin != end && result == 0 ) {
      if ( begin->first != target ) {
         if ( checkIntersect( target, begin->first ) ) {
            reg_t intersect = computeIntersect( target, begin->first );
            //std::cerr << __FUNCTION__<<" "<< (void*)this <<" This " << begin->first; printRegion( begin->first ); std::cerr << " vs target " << target; printRegion( target ); std::cerr << " intersect result " << intersect << std::endl;
            if ( target == intersect ) {
               result = begin->first;
               version = begin->second;
            }
         }
      } 
      begin++;
   }
   return result;
}

template < template <class> class Sparsity>
bool RegionDictionary< Sparsity >::doTheseRegionsForm( reg_t target, std::map< reg_t, unsigned int >::const_iterator ibegin, std::map< reg_t, unsigned int >::const_iterator iend, unsigned int &version ) {
   std::size_t totalSize = 0;
   global_reg_t gtarget( target, this );
   unsigned int maxVersion = 0;
   while ( ibegin != iend ) {
      if ( ibegin->first != target ) {
         if ( checkIntersect( target, ibegin->first ) ) {
            reg_t intersect = computeIntersect( target, ibegin->first );
            //std::cerr << __FUNCTION__ << " " << (void*)this <<" This " << ibegin->first; printRegion( ibegin->first ); std::cerr << " vs target " << target; printRegion( target ); std::cerr << " intersect result " << intersect << std::endl;
            if ( ibegin->first == intersect ) {
               global_reg_t greg( ibegin->first, this );
               totalSize += greg.getDataSize();
               maxVersion = std::max( maxVersion, ibegin->second );
            }      
         }
      }
      ibegin++;
   }
   version = ( totalSize == gtarget.getDataSize() ) ? maxVersion : 0;
   return ( totalSize == gtarget.getDataSize() );
}

template < template <class> class Sparsity>
bool RegionDictionary< Sparsity >::doTheseRegionsForm( reg_t target, std::list< std::pair< reg_t, reg_t > >::const_iterator ibegin, std::list< std::pair< reg_t, reg_t > >::const_iterator iend, bool checkVersion ) {
   std::size_t totalSize = 0;
   global_reg_t gtarget( target, this->getGlobalDirectoryKey() );
   Version *target_entry = this->getRegionData( target );
   unsigned int targetVersion = target_entry->getVersion();
   while ( ibegin != iend ) {
      if ( ibegin->first != target ) {
         if ( checkIntersect( target, ibegin->first ) ) {
            reg_t intersect = computeIntersect( target, ibegin->first );
            //std::cerr << __FUNCTION__ << " " << (void*)this <<" This " << ibegin->first << " "; printRegion( std::cerr, ibegin->first ); std::cerr << " vs target " << target; printRegion( std::cerr, target ); std::cerr << " intersect result " << intersect << std::endl;
            if ( ibegin->first == intersect ) {
               global_reg_t greg( ibegin->first, this->getGlobalDirectoryKey() );
               Version *it_entry = this->getRegionData( ibegin->second );
               totalSize += ( it_entry->getVersion() == targetVersion || !checkVersion ) ? greg.getDataSize() : 0;
            }
         }
      }
      ibegin++;
   }
   return ( totalSize == gtarget.getDataSize() );
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::printRegionGeom( std::ostream &o, reg_t region ) {
   RegionNode const *regNode = this->getRegionNode( region );
   //fprintf(stderr, "%p:%d", this, region);
   o << (void *) this << ":" << std::dec << region;
   if ( regNode == NULL ) {
      //fprintf(stderr, "NULL LEAF !");
      o << "NULL LEAF !";
      return;
   }
   for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {  
      std::size_t accessedLength = regNode->getValue();
      regNode = regNode->getParent();
      std::size_t lowerBound = regNode->getValue();
      //fprintf(stderr, "[%zu;%zu]", lowerBound, accessedLength);
      o << "[" << std::dec << lowerBound << ";" << std::dec << accessedLength << "]";
      regNode = regNode->getParent();
   }
}

template < template <class> class Sparsity>
std::set< reg_t > const &RegionDictionary< Sparsity >::getFixedRegions() const {
   return _fixedRegions;
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::addFixedRegion( reg_t id ) {
   _fixedRegions.insert( id );
}

} // namespace nanos

#endif /* REGIONDICT_H */
