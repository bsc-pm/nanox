#ifndef REGIONDICT_H
#define REGIONDICT_H

#include "regiondict_decl.hpp"
#include "atomic.hpp"
#include "memorymap.hpp"
#include "system_decl.hpp"
#include "printbt_decl.hpp"

namespace nanos {

template <class T>
ContainerDense< T >::ContainerDense( CopyData const &cd ) : _container(), _leafCount( 0 ), _idSeed( 1 ), _dimensionSizes( cd.getNumDimensions(), 0 ), _root( NULL, 0, 0 ), _rogueLock(), _lock(), _keepAtOrigin( false ), _registeredObject( NULL ), sparse( false ) {
   //_container.reserve( MAX_REG_ID );
   for ( unsigned int idx = 0; idx < cd.getNumDimensions(); idx += 1 ) {
      _dimensionSizes[ idx ] = cd.getDimensions()[ idx ].size;
   }
}

template <class T>
RegionNode * ContainerDense< T >::getRegionNode( reg_t id ) const {
   return _container[ id ].getLeaf();
}

template <class T>
void ContainerDense< T >::addRegionNode( RegionNode *leaf, bool rogue ) {
   _container[ leaf->getId() ].setLeaf( leaf );
   _container[ leaf->getId() ].setData( NULL );
   if (!rogue) _leafCount++;
}

template <class T>
Version *ContainerDense< T >::getRegionData( reg_t id ) {
   return _container[ id ].getData();
}

template <class T>
void ContainerDense< T >::setRegionData( reg_t id, Version *data ) {
   _container[ id ].setData( data );
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
reg_t ContainerDense< T >::addRegion( nanos_region_dimension_internal_t const region[], bool rogue ) {
   if ( rogue ) _rogueLock.acquire();
   _lock.acquire();
   reg_t id = _root.addNode( region, _dimensionSizes.size(), 0, *this, rogue );
   _lock.release();
   if ( rogue ) _rogueLock.release();
   return id;
}

template <class T>
reg_t ContainerDense< T >::getNewRegionId() {
   reg_t id = _idSeed++;
   _container.resize( id + 1 );
   if (id >= MAX_REG_ID) { std::cerr <<"Max regions reached."<<std::endl;}
   return id;
}

template <class T>
reg_t ContainerDense< T >::checkIfRegionExists( nanos_region_dimension_internal_t const region[] ) {
   return _root.checkNode( region, _dimensionSizes.size(), 0 );
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
   return _invalidationsLock.acquire();
}

template <class T>
void ContainerDense< T >::invalUnlock() {
   return _invalidationsLock.release();
}

template <class T>
void ContainerDense< T >::addMasterRegionId( reg_t masterId, reg_t localId ) {
   _lock.acquire();
   _masterIdToLocalId[ masterId ] = localId;
   _lock.release();
}

template <class T>
reg_t ContainerDense< T >::getLocalRegionIdFromMasterRegionId( reg_t masterId ) const {
   reg_t result = 0;
   std::map< reg_t, reg_t >::const_iterator it = _masterIdToLocalId.find( masterId );
   if ( it != _masterIdToLocalId.end() ) {
      result = it->second;
   }
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
ContainerSparse< T >::ContainerSparse( RegionDictionary< ContainerDense > &orig ) : _container(), _orig( orig ), sparse( true ) {
}

template <class T>
RegionNode * ContainerSparse< T >::getRegionNode( reg_t id ) const {
   std::map< reg_t, RegionVectorEntry >::const_iterator it = _container.lower_bound( id );
   if ( it == _container.end() || _container.key_comp()(id, it->first) ) {
      //fatal0( "Error, RegionMap::getLeaf does not contain region" );
     RegionNode *leaf = _orig.getRegionNode( id );
   if ( leaf == NULL ) { *(myThread->_file) << "NULL LEAF CHECK by orig: " << std::endl; printBt( *(myThread->_file) ); }
      return leaf;
   }
   return it->second.getLeaf();
}

template <class T>
void ContainerSparse< T >::addRegionNode( RegionNode *leaf, bool rogue ) {
   if ( leaf == NULL ) { *(myThread->_file) << "NULL LEAF INSERT: " << std::endl; printBt( *(myThread->_file) ); }
   _container[ leaf->getId() ].setLeaf( leaf );
}

template <class T>
Version *ContainerSparse< T >::getRegionData( reg_t id ) {
   std::map< reg_t, RegionVectorEntry >::iterator it = _container.lower_bound( id );
   if ( it == _container.end() || _container.key_comp()(id, it->first) ) {
      //fatal0(  "Error, RegionMap::getRegionData does not contain region " );
      it = _container.insert( it, std::map< reg_t, RegionVectorEntry >::value_type( id, RegionVectorEntry() ) );
      it->second.setLeaf( _orig.getRegionNode( id ) );
      return NULL;
   }
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
   reg_t id = _orig.addRegion( region, true );
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
   std::list< std::pair< reg_t, reg_t > > missingParts;
   unsigned int version;
   ensure( id == 1, "Whole region did not get id 1");
   this->addRegionAndComputeIntersects( id, missingParts, version, false );
}

template < template <class> class Sparsity>
RegionDictionary< Sparsity >::RegionDictionary( GlobalRegionDictionary &dict ) : Sparsity< RegionVectorEntry >( dict ), _intersects( dict.getNumDimensions(), MemoryMap< std::set< reg_t > >() ),
      _keyBaseAddress( dict.getKeyBaseAddress() ), _realBaseAddress( dict.getRealBaseAddress() ), _lock() {
   //std::cerr << "CREATING CACHE DICT: tree: " << (void *) &_tree << " orig tree: " << (void *) &dict._tree << std::endl;
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::lock() {
   _lock.acquire();
}

template < template <class> class Sparsity>
bool RegionDictionary< Sparsity >::tryLock() {
   return _lock.tryAcquire();
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::unlock() {
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
void RegionDictionary< Sparsity >::getRegionIntersects( reg_t id, unsigned int version, std::list< reg_t > &superParts, std::list< reg_t > &subParts ) {
   RegionNode const *regNode = this->getRegionNode( id );
   MemoryMap< std::set< reg_t > >::MemChunkList results[ this->getNumDimensions() ];
   unsigned int skipDimensions = 0;
   std::map< reg_t, unsigned int > interacts;
   for ( int idx = this->getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
      std::size_t accessedLength = regNode->getValue();
      regNode = regNode->getParent();
      std::size_t lowerBound = regNode->getValue();
      regNode = regNode->getParent();

      _intersects[ idx ].getOrAddChunk( lowerBound, accessedLength, results[ idx ] );
      std::set< reg_t > thisDimInteracts;
      if ( results[idx].size() == 1 ) {
         bool justCreatedRegion = false;
         if ( *(results[idx].begin()->second) == NULL ) {
            *(results[idx].begin()->second) = NEW std::set< reg_t >();
            justCreatedRegion = true;
         } else if ( (*(results[idx].begin()->second))->count( id ) == 0 ) {
            justCreatedRegion = true;
         } 
         //if(sys.getNetwork()->getNodeNum() == 0) {
         //   std::cerr <<"reg "<< id <<"("<< thisRegionVersion <<") dimIdx "<< idx <<": [" << lowerBound << ":" << accessedLength <<"] set size " << (*(results[idx].begin()->second))->size() << " maxregId " << getRegionCount() << " leafCount "<< getRegionNodeCount()  << " { ";
         //   for ( std::set< reg_t >::iterator sit = (*(results[idx].begin()->second))->begin(); sit != (*(results[idx].begin()->second))->end(); sit++ ) {
         //       Version *itEntry = getRegionData( *sit );
         //       unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : 1 );
         //      std::cerr << *sit << "("<< itVersion <<") ";
         //   }
         //   std::cerr <<"}" << std::endl;
         //}
         if ( ( (*(results[idx].begin()->second))->size() == this->getRegionNodeCount() || ( justCreatedRegion && ((*(results[idx].begin()->second))->size() + 1 ) == this->getRegionNodeCount() ) ) &&
               !( (idx + 1) == (int) this->getNumDimensions() ) ) {
            skipDimensions += 1;
         } else {
            for ( std::set< reg_t >::iterator sit = (*(results[idx].begin()->second))->begin(); sit != (*(results[idx].begin()->second))->end(); sit++ ) {
               if ( *sit == id ) continue;
               Version *itEntry = this->getRegionData( *sit );
               unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : ( this->sparse ? 0 : 1 ) );
               if ( itVersion >= version ) {
                  thisDimInteracts.insert( *sit );
               }
            }
         }
      } else {
         for ( MemoryMap< std::set< reg_t > >::MemChunkList::iterator it = results[ idx ].begin(); it != results[ idx ].end(); it++ ) {
            if ( *(it->second) == NULL ) {
               *(it->second) = NEW std::set< reg_t >();
            } else {
               //thisDimInteracts.insert( (*(it->second))->begin(), (*(it->second))->end());
               //std::cerr <<"Region " << id << " "; printRegion( id ); std::cerr << " dim: " << idx << " LB: " << it->first->getAddress() << " AL: "<< it->first->getLength()  << " interacts with: { ";
               //unsigned int maxVersion = 0;
               //compute max version of registered interactions
               for ( std::set< reg_t >::iterator sit = (*(it->second))->begin(); sit != (*(it->second))->end(); sit++ ) {
                  if ( *sit == id ) continue;
                  Version *itEntry = this->getRegionData( *sit );
                  unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : ( this->sparse ? 0 : 1 ) );
                  if ( itVersion >= version ) {
                     thisDimInteracts.insert( *sit );
                  }
               }
            } 
         }
         //std::cerr <<"Region " << id << " "; printRegion( id ); std::cerr << " dim: " << idx << " interacts with: { ";
      }

      for ( std::set< reg_t >::iterator sit = thisDimInteracts.begin(); sit != thisDimInteracts.end(); sit++ ) {
         interacts[ *sit ]++;
      }
      //std::cerr << "}" << std::endl;;
   }
   //double tfiniINTERS = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " total intersect time " << (tfiniINTERS-tiniINTERS) << std::endl;

   //std::cerr << __FUNCTION__ << " node "<< sys.getNetwork()->getNodeNum() << " reg "<< id << " numdims " <<getNumDimensions() << " skipdims " << skipDimensions << " rogue? " << rogue << " leafs "<< getRegionNodeCount() ; printRegion(id); std::cerr << " results sizes ";
   //for ( int idx = getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
   //   std::cerr << "[ " << idx << "=" << results[idx].size() << ", " << (*(results[idx].begin()->second))->size()<<" ]";
   //}
   //std::cerr << std::endl;
   //if ( skipDimensions == 0 && !rogue ) sys.printBt();
   for ( std::map< reg_t, unsigned int >::iterator mip = interacts.begin(); mip != interacts.end(); mip++ ) {
      //std::cerr <<"numdims " <<getNumDimensions() << " skipdims " << skipDimensions << " count " <<mip->second <<std::endl;
      if ( mip->second == ( this->getNumDimensions() - skipDimensions ) ) {
         reg_t intersectRegId = this->computeIntersect( id, mip->first );
         if ( intersectRegId != id ) {
            //std::cerr << "Looks like a subPart " << intersectRegId << std::endl;
            subParts.push_back( intersectRegId );
         } else {
            //std::cerr << "Looks like a superPart " << mip->first << std::endl;
            superParts.push_back( mip->first );
         }
      } else {
            //std::cerr << "<skip> interact count for reg " << mip->first << " -> " << mip->second << std::endl;
      }
   }
}

template < template <class> class Sparsity>
void RegionDictionary< Sparsity >::addRegionAndComputeIntersects( reg_t id, std::list< std::pair< reg_t, reg_t > > &finalParts, unsigned int &version, bool superPrecise, bool giveSubFragmentsWithSameVersion ) {
   class LocalFunction {
      RegionDictionary &_currentDict;
      public:
      LocalFunction( RegionDictionary &dict ) : _currentDict( dict ) { } 

      reg_t addIntersect( reg_t regionIdA, reg_t regionIdB ) {
         nanos_region_dimension_internal_t resultingRegion[ _currentDict.getNumDimensions() ];
         _currentDict._computeIntersect( regionIdA, regionIdB, resultingRegion );
         reg_t regId = _currentDict.addRegion( resultingRegion );
         return regId;
      }


      void addSubRegion( std::list< std::pair< reg_t, reg_t > > &partsList, std::pair< reg_t, reg_t > const &regionPairToInsert, std::list< std::pair< reg_t, reg_t > > &partsToInsert ) {
         //ensure( !partsList.empty(), "Empty parts list!" );
         //if ( partsList.empty() ) {
         //   std::cerr << "FAIL " << __FUNCTION__ << std::endl; 
         //}

         
         std::list< std::pair< reg_t, reg_t > > intersectList;
         std::list< std::pair< reg_t, reg_t > >::iterator it = partsList.begin();

         //std::cerr << "BEGIN addSubRegion, insert reg: " << regionPairToInsert.first << " content of partsList: ";
         //for( std::list< std::pair< reg_t, reg_t > >::iterator pit = partsList.begin(); pit != partsList.end(); pit++ ) {
         //   std::cerr << "[" << pit->first << "," << pit->second << "] ";
         //}
         //std::cerr << std::endl;

         while ( it != partsList.end() ) {
            if ( it->first == regionPairToInsert.first ) {
               //std::cerr << __FUNCTION__ << ": skip self intersect: " << it->first << std::endl;
               partsToInsert.push_back( regionPairToInsert );
               it = partsList.erase( it );
            } else if ( _currentDict.checkIntersect( it->first, regionPairToInsert.first ) ) {
               intersectList.push_back( *it );
               it = partsList.erase( it );
            } else {
               it++;
            }
         }
      
         for ( it = intersectList.begin(); it != intersectList.end(); it++ ) {
            reg_t intersection = _currentDict.computeIntersect( it->first, regionPairToInsert.first );
            //at this point intersection is either it->first or a subpart of regionPairToInsert.first
            if ( intersection == it->first ) {
               //std::cerr << "region " << regionPairToInsert.first << " totally overlaps " << it->first << std::endl;
               partsToInsert.push_back( std::make_pair( it->first, regionPairToInsert.second ));
            } else {
               partsToInsert.push_back( std::make_pair( intersection, regionPairToInsert.second ));
               std::list<reg_t> pieces;
               _currentDict.substract( it->first, regionPairToInsert.first, pieces );
               for ( std::list< reg_t >::iterator piecesIt = pieces.begin(); piecesIt != pieces.end(); piecesIt++ ) {
                  //std::cerr << "Add part ( " << *piecesIt << ", " << it->second << " )" << std::endl;
                  partsList.push_back( std::make_pair( *piecesIt, it->second ) );
               }
            }
         }
         //std::cerr << "END addSubRegion, content of partsList: ";
         //for( std::list< std::pair< reg_t, reg_t > >::iterator pit = partsList.begin(); pit != partsList.end(); pit++ ) {
         //   std::cerr << "[" << pit->first << "," << pit->second << "] ";
         //}
         //std::cerr << std::endl;
      }
   };

   LocalFunction local( *this );
   //typename RegionDictionary::RegionList subParts;
   std::list< std::pair< reg_t, reg_t > > subParts;
   typename RegionDictionary::RegionList superParts;
   //typename RegionDictionary::RegionList missingParts;
   std::list< std::pair< reg_t, reg_t > > missingParts;
   std::map< unsigned int, std::list< std::pair< reg_t, reg_t > > > missingPartsOrderedByVersion;
   reg_t backgroundRegion = 0;

   Version *thisEntry = this->getRegionData( id );
   unsigned int thisRegionVersion = thisEntry != NULL ? thisEntry->getVersion() : ( this->sparse ? 0 : 1 );

   RegionNode const *regNode = this->getRegionNode( id );
   if ( regNode == NULL ) { *(myThread->_file) << "NULL RegNode, this must come from a rogue insert from a cache. Id " << id << " sparse? "<< (this->sparse ? 1 : 0)  <<std::endl; printBt( *(myThread->_file) );}
   MemoryMap< std::set< reg_t > >::MemChunkList results[ this->getNumDimensions() ];
   std::map< reg_t, unsigned int > interacts;
   unsigned int skipDimensions = 0;


   //double tiniINTERS = OS::getMonotonicTime();

   for ( int idx = this->getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
      std::size_t accessedLength = regNode->getValue();
      regNode = regNode->getParent();
      std::size_t lowerBound = regNode->getValue();
      regNode = regNode->getParent();

      _intersects[ idx ].getOrAddChunk( lowerBound, accessedLength, results[ idx ] );
      std::set< reg_t > thisDimInteracts;
      if ( results[idx].size() == 1 ) {
         bool justCreatedRegion = false;
         if ( *(results[idx].begin()->second) == NULL ) {
            *(results[idx].begin()->second) = NEW std::set< reg_t >();
             justCreatedRegion = true;
         } else if ( (*(results[idx].begin()->second))->count( id ) == 0 ) {
             justCreatedRegion = true;
         } 
         //if(sys.getNetwork()->getNodeNum() == 0) {
            //*(myThread->_file) <<"reg "<< id <<"("<< thisRegionVersion <<") dimIdx "<< idx <<": [" << lowerBound << ":" << accessedLength <<"] set size " << (*(results[idx].begin()->second))->size() << " maxregId " << this->getRegionNodeCount() << " leafCount "<< this->getRegionNodeCount()  << " { ";
            //for ( std::set< reg_t >::iterator sit = (*(results[idx].begin()->second))->begin(); sit != (*(results[idx].begin()->second))->end(); sit++ ) {
            //    Version *itEntry = this->getRegionData( *sit );
            //    unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : 1 );
            //   *(myThread->_file) << *sit << "("<< itVersion <<") ";
            //}
            //*(myThread->_file) <<"}" << std::endl;
         //}
         if ( ( (*(results[idx].begin()->second))->size() == this->getRegionNodeCount() || ( justCreatedRegion && ((*(results[idx].begin()->second))->size() + 1 ) == this->getRegionNodeCount() ) ) && !( (idx + 1) == (int) this->getNumDimensions() ) ) {
            skipDimensions += 1;
         } else {
            for ( std::set< reg_t >::iterator sit = (*(results[idx].begin()->second))->begin(); sit != (*(results[idx].begin()->second))->end(); sit++ ) {
                if ( *sit == id ) continue;
                if ( superPrecise ) {
                   std::set< reg_t >::iterator sit2 = sit;
                   Version *itEntry = this->getRegionData( *sit );
                   unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : ( this->sparse ? 0 : 1 ) );
                   bool insert = true;
                   if ( (*(results[idx].begin()->second))->size() > 1 ) {
                      for ( sit2++; insert && sit2 != (*(results[idx].begin()->second))->end(); sit2++ ) {
                         //std::cerr << "\tintersect test "<< *sit << " vs " << *sit2 << std::endl;
                         if ( *sit2 != id && checkIntersect( *sit, *sit2 ) && checkIntersect( *sit2, id ) ) {
                            Version *entry2 = this->getRegionData( *sit2 );
                            unsigned int thisVersion2 = ( entry2 != NULL ? entry2->getVersion() : ( this->sparse ? 0 : 1 ) );
                            if ( thisVersion2 > itVersion ) {
                               insert = false;
                            }
                         }
                      }
                   }
                   if ( insert ) {
                     // std::cerr << "insert reg " << *sit << std::endl;
                      thisDimInteracts.insert( *sit );
                   }
                } else {
                   thisDimInteracts.insert( *sit );
                }
            }
         }
         (*(results[idx].begin()->second))->insert( id );
      } else {
        //std::cerr << "case with > 1 results"<< std::endl;
        //if(sys.getNetwork()->getNodeNum() == 0)  std::cerr << idx  <<": [" << lowerBound << ":" << accessedLength <<"] results size size " << results[idx].size() << " maxregId " << getMaxRegionId() << std::endl;
      //std::cerr << "intersect map query, dim " << idx << " got entries: " <<  results[ idx ].size() << std::endl;
      for ( MemoryMap< std::set< reg_t > >::MemChunkList::iterator it = results[ idx ].begin(); it != results[ idx ].end(); it++ ) {
         if ( *(it->second) == NULL ) {
            *(it->second) = NEW std::set< reg_t >();
         } else {
            //thisDimInteracts.insert( (*(it->second))->begin(), (*(it->second))->end());
            //std::cerr <<"Region " << id << " "; printRegion( id ); std::cerr << " dim: " << idx << " LB: " << it->first->getAddress() << " AL: "<< it->first->getLength()  << " interacts with: { ";
            //unsigned int maxVersion = 0;
            //compute max version of registered interactions
            for ( std::set< reg_t >::iterator sit = (*(it->second))->begin(); sit != (*(it->second))->end(); sit++ ) {
                if ( *sit == id ) continue;
                if ( superPrecise ) {
                   std::set< reg_t >::iterator sit2 = sit;
                   Version *itEntry = this->getRegionData( *sit );
                   unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : ( this->sparse ? 0 : 1 ) );
                   bool insert = true;
                   if ( (*(it->second))->size() > 1 ) {
                      for ( sit2++; sit2 != (*(it->second))->end(); sit2++ ) {
                         //std::cerr << "\tintersect test "<< *sit << " vs " << *sit2 << std::endl;
                         if ( *sit2 != id && checkIntersect( *sit, *sit2 ) && checkIntersect( *sit2, id ) ) {
                            Version *entry2 = this->getRegionData( *sit2 );
                            unsigned int thisVersion2 = ( entry2 != NULL ? entry2->getVersion() : ( this->sparse ? 0 : 1 ) );
                            if ( thisVersion2 > itVersion ) {
                               insert = false;
                            }
                         }
                      }
                   }
                   if ( insert ) {
                      thisDimInteracts.insert( *sit );
                   }
                } else {
                   thisDimInteracts.insert( *sit );
                }
            }
         } 
         (*(it->second))->insert( id );
      }
      }

      //*(myThread->_file) <<"Region " << id << " "; printRegion( *myThread->_file, id ); *(myThread->_file) << " dim: " << idx << " interacts with: { ";
      for ( std::set< reg_t >::iterator sit = thisDimInteracts.begin(); sit != thisDimInteracts.end(); sit++ ) {
         //*(myThread->_file) << *sit << " ";
         interacts[ *sit ]++;
      }
      //*(myThread->_file) << "}" << std::endl;;
   }
   //double tfiniINTERS = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " total intersect time " << (tfiniINTERS-tiniINTERS) << std::endl;

   //std::cerr << __FUNCTION__ << " node "<< sys.getNetwork()->getNodeNum() << " reg "<< id << " numdims " <<this->getNumDimensions() << " skipdims " << skipDimensions << " leafs "<< this->getRegionNodeCount()<< " results sizes ";
   //for ( int idx = this->getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
   //   std::cerr << "[ " << idx << "=" << results[idx].size() << ", " << (*(results[idx].begin()->second))->size()<<" ]";
   //}
   //std::cerr << std::endl;
   //if ( skipDimensions == 0 && !rogue ) sys.printBt();
      //std::cerr <<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" <<std::endl;
   for ( std::map< reg_t, unsigned int >::iterator mip = interacts.begin(); mip != interacts.end(); mip++ ) {
      //std::cerr <<"numdims " <<this->getNumDimensions() << " skipdims " << skipDimensions << " count " <<mip->second <<std::endl;
      if ( mip->second == ( this->getNumDimensions() - skipDimensions ) ) {
         reg_t intersectRegId = local.addIntersect( id, mip->first );
         if ( intersectRegId != id ) {
            //*(myThread->_file) << "Looks like a subPart " << intersectRegId << ", results from intersect with " << mip->first <<std::endl;
            //subParts.push_back( intersectRegId );
            subParts.push_back( std::make_pair( intersectRegId, mip->first ) );
         } else {
            //*(myThread->_file) << "Looks like a superPart " << mip->first << std::endl;
            superParts.push_back( mip->first );
         }
      } else {
            //*(myThread->_file) << "<skip> interact count for reg " << mip->first << " -> " << mip->second << std::endl;
      }
   }

   version = ( this->sparse ? 0 : 1 );
   unsigned int bgVersion;
   reg_t highestVersionSuperRegion = 0;

   for ( typename RegionDictionary::RegionList::iterator it = superParts.begin(); it != superParts.end(); it++ ) {
      //if (this->sparse)std::cerr << "super region of reg " << id<< ": " << *it<<std::endl;
      unsigned int itVersion = ( this->getRegionData( *it ) == NULL ? ( this->sparse ? 0 : 1 ) : this->getRegionData( *it )->getVersion() );
      if ( itVersion > version ) {
         highestVersionSuperRegion = *it;
         version = itVersion;
      } 
   }

      //if (this->sparse)std::cerr << "this region version " << thisRegionVersion << " super version " << version <<std::endl;
   if ( version > thisRegionVersion ) {
      backgroundRegion = highestVersionSuperRegion;
   } else {
      version = thisRegionVersion;
      backgroundRegion = id;
   }

   // selected sub regions are those with version > than bgVersion
   // version will be the maximum computed version (needs to be returned)
   bgVersion = version;

   //for ( typename RegionDictionary<Sparsity>::RegionList::iterator it = subParts.begin(); it != subParts.end(); it++ ) {
   for ( std::list< std::pair< reg_t, reg_t > >::iterator it = subParts.begin(); it != subParts.end(); it++ ) {
      unsigned int itVersion = ( this->getRegionData( it->second ) == NULL ? ( this->sparse ? 0 : 1 ) : this->getRegionData( it->second )->getVersion() );
      //*(myThread->_file) << (void *)this <<" processing part "<< it->first << " with metadata reg " << it->second << " bgVersion " << bgVersion << " this version " << itVersion << " thisEnrty "<< this->getRegionData( it->second ) << std::endl;
      if ( ( itVersion > bgVersion && !giveSubFragmentsWithSameVersion ) || ( itVersion >= bgVersion && giveSubFragmentsWithSameVersion ) ) {
         //bool intersect = false;
         bool subpart = false;
         //for ( typename RegionDictionary::RegionList::const_iterator cit = missingParts.begin();
         for ( std::list< std::pair< reg_t, reg_t > >::const_iterator cit = missingParts.begin();
               cit != missingParts.end(); cit++ ) {
            if ( it->first != cit->first && this->checkIntersect( it->first, cit->first ) ) {
               reg_t intersectReg = computeTestIntersect( it->first, cit->first );
               if ( intersectReg == it->first )
               {
                  unsigned int citVersion = ( this->getRegionData( cit->second ) == NULL ? ( this->sparse ? 0 : 1 ) : this->getRegionData( cit->second )->getVersion() );
                  //std::cerr << " Intersect between " << *it << ", " << itVersion << " and " << *cit << ", " << citVersion << std::endl;
                  subpart = ( citVersion >= itVersion );
               }
            }
            //if ( intersect ) {
            //   std::cerr << "*** WARNING: subregions intersect in "<< __FUNCTION__ << " regions: " << *it << " and " << *cit << std::endl;
            //}
         }

         if ( !subpart ) {
            missingParts.push_back( *it );
            missingPartsOrderedByVersion[ itVersion ].push_back( *it );
            version = itVersion > version ? itVersion : version;
         }
      }
   }

   finalParts.push_back( std::make_pair( id, backgroundRegion ) );
   std::list< std::pair< reg_t, reg_t > > effectiveParts;

   //for ( typename RegionDictionary::RegionList::const_iterator cit = missingParts.begin();
   //      cit != missingParts.end(); cit++ ) {
   //   unsigned int citVersion = ( this->getRegionData( *cit ) == NULL ? ( this->sparse ? 0 : 1 ) : this->getRegionData( *cit )->getVersion() );
   //   *(myThread->_file) << "region " << *cit << " has version " << citVersion << std::endl;
   //}
   
   //for ( std::map< unsigned int, typename RegionDictionary::RegionList >::const_reverse_iterator lit = missingPartsOrderedByVersion.rbegin();
   for ( std::map< unsigned int, std::list< std::pair< reg_t, reg_t > > >::const_reverse_iterator lit = missingPartsOrderedByVersion.rbegin();
         lit != missingPartsOrderedByVersion.rend(); lit++ ) {
     // std::cerr << "process regions with version " << lit->first << " elems: " << lit->second.size() <<std::endl;
      //for ( typename RegionDictionary::RegionList::const_iterator cit = lit->second.begin();
      for ( std::list< std::pair< reg_t, reg_t > >::const_iterator cit = lit->second.begin();
            cit != lit->second.end(); cit++ ) {
      //   unsigned int citVersion = ( this->getRegionData( *cit ) == NULL ? ( this->sparse ? 0 : 1 ) : this->getRegionData( *cit )->getVersion() );
     //    std::cerr << "region " << *cit << " has version " << citVersion << std::endl;

      local.addSubRegion( finalParts, *cit, effectiveParts ); //FIXME: handle case when "finalParts" is empty
      }
   }

   // /*if (this->sparse)*/ std::cerr << "starting with " << id << "," << backgroundRegion << std::endl;
   //for ( typename RegionDictionary::RegionList::const_iterator cit = missingParts.begin();
   //      cit != missingParts.end(); cit++ ) {
   //   // /*if (this->sparse) */ std::cerr << /*myThread->getId() <<*/ "BG rgion is " << backgroundRegion<< ", Add sub region " << *cit << " resulting set: { ";
   //   local.addSubRegion( finalParts, *cit, effectiveParts ); //FIXME: handle case when "finalParts" is empty
   //   //for (std::list< std::pair< reg_t, reg_t > >::const_iterator cpit = finalParts.begin(); cpit != finalParts.end(); cpit++ ) {
   //   //   /*if (this->sparse)*/ std::cerr << "(" << cpit->first << "," << cpit->second << ")";
   //   //}
   //   ///*if (this->sparse)*/ std::cerr <<" }"<< std::endl;
   //}

   for ( std::list< std::pair< reg_t, reg_t > >::const_iterator cit = effectiveParts.begin();
         cit != effectiveParts.end(); cit++ ) {
      finalParts.push_back( *cit );
   }
   //for ( typename RegionDictionary::RegionList::const_iterator cit = missingParts.begin();
   //      cit != missingParts.end(); cit++ ) {
   //   //if (this->sparse)std::cerr << /*myThread->getId() <<*/ " final parts region is " << *cit << std::endl;
   //   finalParts.push_back( std::make_pair( *cit, *cit ) );
   //}
   //double tfiniTOTAL = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " rest of time " << (tfiniTOTAL-tfiniINTERS) << std::endl;
}

#if 0
template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::tryObtainRegionId( CopyData const &cd ) {
   reg_t id = 0;
   //std::cerr << "cd numRegs: " << cd.getNumDimensions() << cd << " this: " << this->getNumDimensions() << std::endl;
   ensure( cd.getNumDimensions() == this->getNumDimensions(), "ERROR" );
   if ( cd.getNumDimensions() != this->getNumDimensions() ) {
      std::cerr << "Error, invalid numDimensions" << std::endl;
   } else {
      id = this->checkIfRegionExists( cd.getDimensions() );
   }
   return id;
}
#endif

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::obtainRegionId( CopyData const &cd, WD const &wd, unsigned int idx ) {
   reg_t id = 0;
   CopyData *deductedCd = NULL;
   if ( this->getRegisteredObject() != NULL && !this->getRegisteredObject()->equalGeometry( cd ) ) {
      CopyData *tmp = NEW CopyData( *this->getRegisteredObject() );
      nanos_region_dimension_internal_t *dims = NEW nanos_region_dimension_internal_t[tmp->getNumDimensions()];
      tmp->setDimensions( dims );
      cd.deductCd( *this->getRegisteredObject(), tmp );
      deductedCd = tmp;
   }
   CopyData const &realCd = deductedCd != NULL ? *deductedCd : cd;
   if ( realCd.getNumDimensions() != this->getNumDimensions() ) {
      std::cerr << "Error: cd.getNumDimensions() returns " << realCd.getNumDimensions()
         << " but I already have the object registered with " << this->getNumDimensions()
         << " dimensions. WD is : "
         << ( wd.getDescription() != NULL ? wd.getDescription() : "n/a" )
         << " copy index: " << idx << " got reg object? " << this->getRegisteredObject()
         << std::endl;
   }
   ensure( realCd.getNumDimensions() == this->getNumDimensions(), "ERROR" );
   if ( realCd.getNumDimensions() != this->getNumDimensions() ) {
      std::cerr << "Error, invalid numDimensions" << std::endl;
   } else {
      for ( unsigned int cidx = 0; cidx < realCd.getNumDimensions(); cidx += 1 ) {
         if ( this->getDimensionSizes()[ cidx ] != realCd.getDimensions()[ cidx ].size ) {
            fatal("Object with base address " << (void *)realCd.getBaseAddress() <<
                  " was previously registered with a different size in dimension " <<
                  std::dec << cidx << " (previously was " <<
                  std::dec << this->getDimensionSizes()[ cidx ] <<
                  " now received size " << std::dec <<
                  realCd.getDimensions()[ cidx ].size << ")." );
         }
      }
      id = this->addRegion( realCd.getDimensions() );
   }
   //this->printRegion(std::cerr, id); std::cerr << std::endl;
   return id;
}

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::obtainRegionId( nanos_region_dimension_internal_t region[] ) {
   return this->addRegion( region );
}

// template < template <class> class Sparsity>
// reg_t RegionDictionary< Sparsity >::registerRegion( CopyData const &cd, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version, WD const &wd, unsigned int idx ) {
//    reg_t id = 0;
//    //unsigned int currentLeafCount = 0;
//    //bool newlyCreatedRegion = false;
//    //std::cerr << "=== RegionDictionary::addRegion ====================================================" << std::endl;
//    //std::cerr << cd ;
//    //{
//    //double tini = OS::getMonotonicTime();
// 
//    //currentLeafCount = this->getRegionNodeCount();
//    id = obtainRegionId( cd, wd, idx );
//    //newlyCreatedRegion = ( this->getRegionNodeCount() > currentLeafCount );
// 
//    //double tfini = OS::getMonotonicTime();
//    //std::cerr << __FUNCTION__ << " Insert region into node time " << (tfini-tini) << std::endl;
//    //}
//    //std::cerr << cd << std::endl;
//    //std::cerr << "got id "<< id << std::endl;
//    //if ( newlyCreatedRegion ) { std::cerr << __FUNCTION__ << ": just created region " << id << std::endl; }
// 
//    //{
//    //double tini = OS::getMonotonicTime();
//    this->addRegionAndComputeIntersects( id, missingParts, version );
//    //double tfini = OS::getMonotonicTime();
//    //std::cerr << __FUNCTION__ << " add and compute intersects time " << (tfini-tini) << std::endl;
//    //}
// 
//    //std::cerr << "===== reg " << id << " ====================================================" << std::endl;
//    return id;
// }

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::registerRegion( reg_t id, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version, bool superPrecise ) {
   this->addRegionAndComputeIntersects( id, missingParts, version, superPrecise, false );
   return id;
}

template < template <class> class Sparsity>
reg_t RegionDictionary< Sparsity >::registerRegionReturnSameVersionSubparts( reg_t id, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version, bool superPrecise ) {
   this->addRegionAndComputeIntersects( id, missingParts, version, superPrecise, true );
   return id;
}

template < template <class> class Sparsity>
bool RegionDictionary< Sparsity >::checkIntersect( reg_t regionIdA, reg_t regionIdB ) const {
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

}
#endif /* REGIONDICT_H */
