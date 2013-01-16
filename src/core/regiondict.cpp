
#include "regiondict_decl.hpp"
#include "memorymap.hpp"
#include "atomic.hpp"
#include "version.hpp"
#include "system_decl.hpp"
#include "basethread.hpp"
#include "os.hpp"

using namespace nanos;
#define MAX_REG_ID 16384
RegionNode::RegionNode( RegionNode *parent, std::size_t value, reg_t id ) : _parent( parent ), _value( value ), _id( id ), _sons( NULL ) {
   if ( id > 1 ) {
      _memoIntersectInfo = NEW reg_t[ id - 1 ];
      ::memset(_memoIntersectInfo, 0, sizeof( reg_t ) * (id - 1) );
   } else {
      _memoIntersectInfo = NULL;
   }
   //std::cerr << "created node with value " << value << " and id " << id << std::endl;
}

RegionNode::~RegionNode() {
   delete _sons;
}

reg_t RegionNode::getId() const {
   return _id;
}

reg_t RegionNode::getMemoIntersect( reg_t target ) const {
   return _memoIntersectInfo[ target-1 ];
}

void RegionNode::setMemoIntersect( reg_t target, reg_t value ) const {
   _memoIntersectInfo[ target-1 ] = value;
}

reg_t RegionNode::addNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep, RegionDictionary & dict ) {
   bool lastNode = ( deep == ( 2 * numDimensions - 1 ) );
   std::size_t value = ( ( deep & 1 ) == 0 ) ? dimensions[ (deep >> 1) ].lower_bound : dimensions[ (deep >> 1) ].accessed_length;
   //std::cerr << "this node value is "<< _value << " gonna add value " << value << " this deep " << deep<< std::endl;
   if ( !_sons ) {
      _sons = new std::map<std::size_t, RegionNode>();
   }

   std::map<std::size_t, RegionNode>::iterator it = _sons->lower_bound( value );
   bool haveToInsert = ( it == _sons->end() || _sons->key_comp()(value, it->first) );
   reg_t newId = ( lastNode && haveToInsert ) ? dict.getNewRegionId() : 0;
   reg_t retId = 0;

   if ( haveToInsert ) {
      it = _sons->insert( it, std::map<std::size_t, RegionNode>::value_type( value, RegionNode( this, value, newId ) ) );
      if ( lastNode ) dict.addLeaf( &(it->second) );
   }

   if ( lastNode ) {
      retId = it->second.getId();
   } else {
      retId = it->second.addNode( dimensions, numDimensions, deep + 1, dict );
   }
   return retId;
}

reg_t RegionNode::checkNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep ) {
   bool lastNode = ( deep == ( 2 * numDimensions - 1 ) );
   std::size_t value = ( ( deep & 1 ) == 0 ) ? dimensions[ (deep >> 1) ].lower_bound : dimensions[ (deep >> 1) ].accessed_length;
   //std::cerr << "this node value is "<< _value << " gonna add value " << value << " this deep " << deep<< std::endl;
   if ( !_sons ) {
      return 0;
   }

   std::map<std::size_t, RegionNode>::iterator it = _sons->lower_bound( value );
   bool haveToInsert = ( it == _sons->end() || _sons->key_comp()(value, it->first) );
   reg_t retId = 0;

   if ( haveToInsert ) {
      return 0;
   }

   if ( lastNode ) {
      retId = it->second.getId();
   } else {
      retId = it->second.checkNode( dimensions, numDimensions, deep + 1 );
   }
   return retId;
}

std::size_t RegionNode::getValue() const {
   return _value;
}

RegionNode *RegionNode::getParent() const {
   return _parent;
}

RegionAssociativeContainer::RegionVectorEntry::RegionVectorEntry() : _leaf( NULL ), _data( NULL ) {
}

RegionAssociativeContainer::RegionVectorEntry::RegionVectorEntry( RegionVectorEntry const &rve ) : _leaf( rve._leaf ), _data( rve._data ) {
}

RegionAssociativeContainer::RegionVectorEntry &RegionAssociativeContainer::RegionVectorEntry::operator=( RegionVectorEntry const &rve ) {
   _leaf = rve._leaf;
   _data = rve._data;
   return *this;
}

RegionAssociativeContainer::RegionVectorEntry::~RegionVectorEntry() {
}

void RegionAssociativeContainer::RegionVectorEntry::setLeaf( RegionNode *rn ) {
   _leaf = rn;
}
RegionNode *RegionAssociativeContainer::RegionVectorEntry::getLeaf() const {
   return _leaf;
}

void RegionAssociativeContainer::RegionVectorEntry::setData( Version *d ) {
   _data = d;
}

Version *RegionAssociativeContainer::RegionVectorEntry::getData() const {
   return _data;
}

RegionIntersectionDictionary::RegionIntersectionDictionary( RegionDictionary &d ) : _dict( d ), _intersects( d.getNumDimensions(), MemoryMap< std::set< reg_t > >() ) {
}

RegionTreeRoot::RegionTreeRoot( CopyData const &cd ) : _idSeed( 1 ), _dimensionSizes( cd.getNumDimensions(), 0 ), _root( NULL, 0, 0 ) {
   for ( unsigned int idx = 0; idx < cd.getNumDimensions(); idx += 1 ) {
      _dimensionSizes[ idx ] = cd.getDimensions()[ idx ].size;
   }
}

unsigned int RegionTreeRoot::getNumDimensions() const {
   return _dimensionSizes.size();
}

reg_t RegionTreeRoot::getNewRegionId() {
   reg_t id = _idSeed++;
   if (id >= MAX_REG_ID) { std::cerr <<"Max regions reached."<<std::endl;}
   //std::cerr << "Created id "<< id << std::endl;
   return id;
}

reg_t RegionTreeRoot::addRegion( nanos_region_dimension_internal_t const region[], RegionDictionary &dict ) {
   return _root.addNode( region, getNumDimensions(), 0, dict );
}

reg_t RegionTreeRoot::checkIfRegionExists( nanos_region_dimension_internal_t const region[] ) {
   return _root.checkNode( region, getNumDimensions(), 0 );
}

reg_t RegionTreeRoot::getMaxRegionId() const {
   return _idSeed.value();
}

RegionDictionary::RegionDictionary( CopyData const &cd, RegionAssociativeContainer &container, bool rogue ) : _tree( *( NEW RegionTreeRoot( cd ) ) ), _intersects( *this ), _regionContainer( container ), _baseAddress( (uint64_t) cd.getBaseAddress() ), _rogue( rogue ), _lock(), _rogueLock( NULL ) {
   //std::cerr << "CREATING MASTER DICT: tree: " << (void *) &_tree << std::endl;
   nanos_region_dimension_internal_t dims[ cd.getNumDimensions() ];
   for ( unsigned int idx = 0; idx < cd.getNumDimensions(); idx++ ) {
      dims[ idx ].size  = cd.getDimensions()[ idx ].size;
      dims[ idx ].accessed_length = cd.getDimensions()[ idx ].size;
      dims[ idx ].lower_bound = 0;
   }
   reg_t id = _tree.addRegion( dims, *this );
   std::list< std::pair< reg_t, reg_t > > missingParts;
   unsigned int version;
   ensure( id == 1, "Whole region did not get id 1");
   _intersects.addRegionAndComputeIntersects( id, missingParts, version, _rogue, true );
}

RegionDictionary::RegionDictionary( RegionDictionary &dict, RegionAssociativeContainer &container, bool rogue ) : _tree( dict._tree ), _intersects( *this ), _regionContainer( container ), _baseAddress( dict._baseAddress ), _rogue( rogue ), _lock(), _rogueLock( &dict._lock ) {
   //std::cerr << "CREATING CACHE DICT: tree: " << (void *) &_tree << " orig tree: " << (void *) &dict._tree << std::endl;
}

void RegionDictionary::lock() {
   _lock.acquire();
}

bool RegionDictionary::tryLock() {
   return _lock.tryAcquire();
}

void RegionDictionary::unlock() {
   _lock.release();
}

unsigned int RegionDictionary::getNumDimensions() const {
   return _tree.getNumDimensions();
}

void RegionIntersectionDictionary::addRegionAndComputeIntersects( reg_t id, std::list< std::pair< reg_t, reg_t > > &finalParts, unsigned int &version, bool rogue, bool justCreatedRegion, bool superPrecise ) {
   class LocalFunction {
      RegionDictionary &_currentDict;
      public:
      LocalFunction( RegionDictionary &dict ) : _currentDict( dict ) { } 

      void _computeIntersect( reg_t regionIdA, reg_t regionIdB, nanos_region_dimension_internal_t *outReg ) {
         RegionNode const *regA = _currentDict.getLeafRegionNode( regionIdA );
         RegionNode const *regB = _currentDict.getLeafRegionNode( regionIdB );

         if ( regionIdA == regionIdB ) {
            std::cerr << __FUNCTION__ << " Dummy check! regA == regB" << std::endl;
            for ( int dimensionCount = _currentDict.getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {
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
         //_dict.printRegion(regionIdA ); std::cerr << std::endl;
         //_dict.printRegion(regionIdB ); std::cerr << std::endl;
      
         for ( int dimensionCount = _currentDict.getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {
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

      reg_t addIntersect( reg_t regionIdA, reg_t regionIdB ) {
         nanos_region_dimension_internal_t resultingRegion[ _currentDict.getNumDimensions() ];
         _computeIntersect( regionIdA, regionIdB, resultingRegion );
         reg_t regId = _currentDict.addRegionByComponents( resultingRegion );
         return regId;
      }

      reg_t computeIntersect( reg_t regionIdA, reg_t regionIdB ) {
   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode const *maxReg = _currentDict.getLeafRegionNode( maxRegionId );
      reg_t data = maxReg->getMemoIntersect( minRegionId );
      if ( data != (unsigned int)-2 ) {
         //std::cerr << "hit compute!"<<std::endl;
         return data;
      }
   }
         nanos_region_dimension_internal_t resultingRegion[ _currentDict.getNumDimensions() ];
         _computeIntersect( regionIdA, regionIdB, resultingRegion );
         reg_t regId = _currentDict.checkIfRegionExistsByComponents( resultingRegion );

   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode *maxReg = _currentDict.getLeafRegionNode( maxRegionId );
      maxReg->setMemoIntersect( minRegionId, regId );
   }

         return regId;
      }

      void addSubRegion( std::list< std::pair< reg_t, reg_t > > &partsList, reg_t regionToInsert ) {
         //ensure( !partsList.empty(), "Empty parts list!" );
         //if ( partsList.empty() ) {
         //   std::cerr << "FAIL " << __FUNCTION__ << std::endl; 
         //}

         
         std::list< std::pair< reg_t, reg_t > > intersectList;
         std::list< std::pair< reg_t, reg_t > >::iterator it = partsList.begin();

         //std::cerr << "BEGIN addSubRegion, insert reg: " << regionToInsert << " content of partsList: ";
         //for( std::list< std::pair< reg_t, reg_t > >::iterator pit = partsList.begin(); pit != partsList.end(); pit++ ) {
         //   std::cerr << "[" << pit->first << "," << pit->second << "] ";
         //}
         //std::cerr << std::endl;

         while ( it != partsList.end() ) {
            if ( it->first == regionToInsert ) {
               //std::cerr << __FUNCTION__ << ": skip self intersect: " << it->first << std::endl;
               it = partsList.erase( it );
            } else if ( _currentDict.checkIntersect( it->first, regionToInsert ) ) {
               intersectList.push_back( *it );
               it = partsList.erase( it );
            } else {
               it++;
            }
         }
      
         for ( it = intersectList.begin(); it != intersectList.end(); it++ ) {
            std::list<reg_t> pieces;
            _currentDict.substract( it->first, regionToInsert, pieces );
            for ( std::list< reg_t >::iterator piecesIt = pieces.begin(); piecesIt != pieces.end(); piecesIt++ ) {
               //std::cerr << "Add part " << *piecesIt << std::endl;
               partsList.push_back( std::make_pair( *piecesIt, it->second ) );
            }
         }
         //std::cerr << "END addSubRegion, content of partsList: ";
         //for( std::list< std::pair< reg_t, reg_t > >::iterator pit = partsList.begin(); pit != partsList.end(); pit++ ) {
         //   std::cerr << "[" << pit->first << "," << pit->second << "] ";
         //}
         //std::cerr << std::endl;
      }
   };

   LocalFunction local( _dict );
   RegionDictionary::RegionList subParts;
   RegionDictionary::RegionList superParts;
   RegionDictionary::RegionList missingParts;
   reg_t backgroundRegion = 0;

   Version *thisEntry = _dict.getRegionData( id );
   unsigned int thisRegionVersion = thisEntry != NULL ? thisEntry->getVersion() : 1;

   RegionNode const *regNode = _dict.getLeafRegionNode( id );
   if ( regNode == NULL ) { std::cerr << "NULL RegNode, this must come from a rogue insert from a cache."<<std::endl;}
   MemoryMap< std::set< reg_t > >::MemChunkList results[ _dict.getNumDimensions() ];
   std::map< reg_t, unsigned int > interacts;
   unsigned int skipDimensions = 0;


   //double tiniINTERS = OS::getMonotonicTime();

   for ( int idx = _dict.getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
      std::size_t accessedLength = regNode->getValue();
      regNode = regNode->getParent();
      std::size_t lowerBound = regNode->getValue();
      regNode = regNode->getParent();

      _intersects[ idx ].getOrAddChunk( lowerBound, accessedLength, results[ idx ] );
      std::set< reg_t > thisDimInteracts;
      if ( results[idx].size() == 1 ) {
         if ( *(results[idx].begin()->second) == NULL ) {
            *(results[idx].begin()->second) = NEW std::set< reg_t >();
         }
         //if(sys.getNetwork()->getNodeNum() == 0) {
         //   std::cerr <<"reg "<< id <<"("<< thisRegionVersion <<") dimIdx "<< idx <<": [" << lowerBound << ":" << accessedLength <<"] set size " << (*(results[idx].begin()->second))->size() << " maxregId " << _dict.getRegionCount() << " leafCount "<< _dict.getLeafCount()  << " { ";
         //   for ( std::set< reg_t >::iterator sit = (*(results[idx].begin()->second))->begin(); sit != (*(results[idx].begin()->second))->end(); sit++ ) {
         //       Version *itEntry = _dict.getRegionData( *sit );
         //       unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : 1 );
         //      std::cerr << *sit << "("<< itVersion <<") ";
         //   }
         //   std::cerr <<"}" << std::endl;
         //}
         if ( (*(results[idx].begin()->second))->size() == _dict.getLeafCount() ||
              ( justCreatedRegion && ((*(results[idx].begin()->second))->size() + 1 ) == _dict.getLeafCount() )) {
            skipDimensions += 1;
         } else {
            for ( std::set< reg_t >::iterator sit = (*(results[idx].begin()->second))->begin(); sit != (*(results[idx].begin()->second))->end(); sit++ ) {
                if ( *sit == id ) continue;
                if ( superPrecise ) {
                   std::set< reg_t >::iterator sit2 = sit;
                   Version *itEntry = _dict.getRegionData( *sit );
                   unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : 1 );
                   bool insert = true;
                   if ( (*(results[idx].begin()->second))->size() > 1 ) {
                      for ( sit2++; insert && sit2 != (*(results[idx].begin()->second))->end(); sit2++ ) {
                         //std::cerr << "\tintersect test "<< *sit << " vs " << *sit2 << std::endl;
                         if ( *sit2 != id && _dict.checkIntersect( *sit, *sit2 ) && _dict.checkIntersect( *sit2, id ) ) {
                            Version *entry2 = _dict.getRegionData( *sit2 );
                            unsigned int thisVersion2 = ( entry2 != NULL ? entry2->getVersion() : 1 );
                            if ( thisVersion2 > itVersion ) {
                               insert = false;
                            }
                         }
                      }
                   }
                   if ( insert ) {
                      //std::cerr << "insert reg " << *sit << std::endl;
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
        //if(sys.getNetwork()->getNodeNum() == 0)  std::cerr << idx  <<": [" << lowerBound << ":" << accessedLength <<"] results size size " << results[idx].size() << " maxregId " << _dict.getMaxRegionId() << std::endl;
      //std::cerr << "intersect map query, dim " << idx << " got entries: " <<  results[ idx ].size() << std::endl;
      for ( MemoryMap< std::set< reg_t > >::MemChunkList::iterator it = results[ idx ].begin(); it != results[ idx ].end(); it++ ) {
         if ( *(it->second) == NULL ) {
            *(it->second) = NEW std::set< reg_t >();
         } else {
            //thisDimInteracts.insert( (*(it->second))->begin(), (*(it->second))->end());
            //std::cerr <<"Region " << id << " "; _dict.printRegion( id ); std::cerr << " dim: " << idx << " LB: " << it->first->getAddress() << " AL: "<< it->first->getLength()  << " interacts with: { ";
            //unsigned int maxVersion = 0;
            //compute max version of registered interactions
            for ( std::set< reg_t >::iterator sit = (*(it->second))->begin(); sit != (*(it->second))->end(); sit++ ) {
                if ( *sit == id ) continue;
                if ( superPrecise ) {
                   std::set< reg_t >::iterator sit2 = sit;
                   Version *itEntry = _dict.getRegionData( *sit );
                   unsigned int itVersion = ( itEntry != NULL ? itEntry->getVersion() : 1 );
                   bool insert = true;
                   if ( (*(it->second))->size() > 1 ) {
                      for ( sit2++; sit2 != (*(it->second))->end(); sit2++ ) {
                         //std::cerr << "\tintersect test "<< *sit << " vs " << *sit2 << std::endl;
                         if ( *sit2 != id && _dict.checkIntersect( *sit, *sit2 ) && _dict.checkIntersect( *sit2, id ) ) {
                            Version *entry2 = _dict.getRegionData( *sit2 );
                            unsigned int thisVersion2 = ( entry2 != NULL ? entry2->getVersion() : 1 );
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
      //std::cerr <<"Region " << id << " "; _dict.printRegion( id ); std::cerr << " dim: " << idx << " interacts with: { ";
      }

      for ( std::set< reg_t >::iterator sit = thisDimInteracts.begin(); sit != thisDimInteracts.end(); sit++ ) {
        // std::cerr << *sit << " ";
         interacts[ *sit ]++;
      }
      //std::cerr << "}" << std::endl;;
   }
   //double tfiniINTERS = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " total intersect time " << (tfiniINTERS-tiniINTERS) << std::endl;

   //std::cerr << __FUNCTION__ << " node "<< sys.getNetwork()->getNodeNum() << " reg "<< id << " numdims " <<_dict.getNumDimensions() << " skipdims " << skipDimensions << " rogue? " << rogue << " leafs "<< _dict.getLeafCount() ; _dict.printRegion(id); std::cerr << " results sizes ";
   //for ( int idx = _dict.getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
   //   std::cerr << "[ " << idx << "=" << results[idx].size() << ", " << (*(results[idx].begin()->second))->size()<<" ]";
   //}
   //std::cerr << std::endl;
   //if ( skipDimensions == 0 && !rogue ) sys.printBt();
   for ( std::map< reg_t, unsigned int >::iterator mip = interacts.begin(); mip != interacts.end(); mip++ ) {
      //std::cerr <<"numdims " <<_dict.getNumDimensions() << " skipdims " << skipDimensions << " count " <<mip->second <<std::endl;
      if ( mip->second == ( _dict.getNumDimensions() - skipDimensions ) ) {
         reg_t intersectRegId = local.addIntersect( id, mip->first );
         if ( intersectRegId != id ) {
            //std::cerr << "Looks like a subPart " << intersectRegId << std::endl;
            subParts.push_back( intersectRegId );
         } else {
            //std::cerr << "Looks like a superPart " << mip->first << std::endl;
            superParts.push_back( mip->first );
         }
      }else {
            //std::cerr << "<skip> interact count for reg " << mip->first << " -> " << mip->second << std::endl;
      }
   }

   version = 1;
   unsigned int bgVersion;
   reg_t highestVersionSuperRegion = 0;

   for ( RegionDictionary::RegionList::iterator it = superParts.begin(); it != superParts.end(); it++ ) {
      //std::cerr << "super region of reg " << id<< ": " << *it<<std::endl;
      unsigned int itVersion = ( _dict.getRegionData( *it ) == NULL ? 1 : _dict.getRegionData( *it )->getVersion() );
      if ( itVersion > version ) {
         highestVersionSuperRegion = *it;
         version = itVersion;
      } 
   }

   if ( version > thisRegionVersion ) {
      backgroundRegion = highestVersionSuperRegion;
   } else {
      version = thisRegionVersion;
      backgroundRegion = id;
   }

   // selected sub regions are those with version > than bgVersion
   // version will be the maximum computed version (needs to be returned)
   bgVersion = version;

   for ( RegionDictionary::RegionList::iterator it = subParts.begin(); it != subParts.end(); it++ ) {
      unsigned int itVersion = ( _dict.getRegionData( *it ) == NULL ? 1 : _dict.getRegionData( *it )->getVersion() );
      if ( itVersion > bgVersion ) {
         //bool intersect = false;
         bool subpart = false;
         for ( RegionDictionary::RegionList::const_iterator cit = missingParts.begin(); cit != missingParts.end(); cit++ ) {
            //intersect = _dict.checkIntersect( *it, *cit );
            reg_t intersectReg = local.computeIntersect( *it, *cit );
            if ( intersectReg == *it )
            {
               unsigned int citVersion = ( _dict.getRegionData( *cit ) == NULL ? 1 : _dict.getRegionData( *cit )->getVersion() );
               subpart = ( citVersion >= itVersion );
            }
            //if ( intersect ) {
            //   std::cerr << "*** WARNING: subregions intersect in "<< __FUNCTION__ << " regions: " << *it << " and " << *cit << std::endl;
            //}
         }

         if ( !subpart ) {
            missingParts.push_back( *it );
            version = itVersion;
         }
      }
   }

   finalParts.push_back( std::make_pair( id, backgroundRegion ) );
   //std::cerr << "starting with " << id << "," << backgroundRegion << std::endl;
   for ( RegionDictionary::RegionList::const_iterator cit = missingParts.begin(); cit != missingParts.end(); cit++ ) {
      //std::cerr << /*myThread->getId() <<*/ "BG rgion is " << backgroundRegion<< ", Add sub region " << *cit << " resulting set: { ";
      local.addSubRegion( finalParts, *cit ); //FIXME: handle case when "finalParts" is empty
      //for (std::list< std::pair< reg_t, reg_t > >::const_iterator cpit = finalParts.begin(); cpit != finalParts.end(); cpit++ ) {
      //   std::cerr << "(" << cpit->first << "," << cpit->second << ")";
      //}
      //std::cerr <<" }"<< std::endl;
   }
   for ( RegionDictionary::RegionList::const_iterator cit = missingParts.begin(); cit != missingParts.end(); cit++ ) {
      //std::cerr << /*myThread->getId() <<*/ " final parts region is " << *cit << std::endl;
      finalParts.push_back( std::make_pair( *cit, *cit ) );
   }
   //double tfiniTOTAL = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " rest of time " << (tfiniTOTAL-tfiniINTERS) << std::endl;
}

reg_t RegionDictionary::addRegion( CopyData const &cd, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version ) {
   reg_t id = 0;
   unsigned int currentLeafCount = 0;
   bool newlyCreatedRegion = false;
   //std::cerr << "=== RegionDictionary::addRegion ====================================================" << std::endl;

   //{
   //double tini = OS::getMonotonicTime();
   ensure( cd.getNumDimensions() == getNumDimensions(), "ERROR" );
   if ( cd.getNumDimensions() != getNumDimensions() ) {
      std::cerr << "Error, invalid numDimensions" << std::endl;
   } else {
      currentLeafCount = getLeafCount();
      id = _tree.addRegion( cd.getDimensions(), *this );
      newlyCreatedRegion = ( getLeafCount() > currentLeafCount );
   }
   //double tfini = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " Insert region into node time " << (tfini-tini) << std::endl;
   //}
   //std::cerr << cd << std::endl;
   //std::cerr << "got id "<< id << std::endl;
   //if ( newlyCreatedRegion ) { std::cerr << __FUNCTION__ << ": just created region " << id << std::endl; }

   //{
   //double tini = OS::getMonotonicTime();
   _intersects.addRegionAndComputeIntersects( id, missingParts, version, _rogue, newlyCreatedRegion );
   //double tfini = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " add and compute intersects time " << (tfini-tini) << std::endl;
   //}

   //std::cerr << "===== reg " << id << " ====================================================" << std::endl;
   return id;
}

reg_t RegionDictionary::addRegion( reg_t id, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version, bool superPrecise ) {
   _intersects.addRegionAndComputeIntersects( id, missingParts, version, _rogue, false, superPrecise );
   return id;
}

reg_t RegionDictionary::addRegionByComponents( nanos_region_dimension_internal_t const region[] ) {
   //return _root.addNode( region, getNumDimensions(), 0, *this );
   reg_t t = _tree.addRegion( region, *this );
   return t;//_tree.addRegion( region, *this );
}

reg_t RegionDictionary::checkIfRegionExistsByComponents( nanos_region_dimension_internal_t const region[] ) {
   //return _root.addNode( region, getNumDimensions(), 0, *this );
   return _tree.checkIfRegionExists( region );
}

reg_t RegionDictionary::getNewRegionId() {
   return _tree.getNewRegionId();
}

RegionNode *RegionDictionary::getLeafRegionNode( reg_t id ) const {
   return _regionContainer.getLeaf( id );
}

Version *RegionDictionary::getRegionData( reg_t id ) const {
   return _regionContainer.getRegionData( id );
}

void RegionDictionary::setRegionData( reg_t id, Version *v ) {
   _regionContainer.setRegionData( id, v );
}

RegionAssociativeContainer const &RegionDictionary::getContainer() const {
   return _regionContainer;
}

/*
reg_t RegionDictionary::addIntersect( reg_t regionIdA, reg_t regionIdB ) {
   nanos_region_dimension_internal_t resultingRegion[ _dimensionSizes.size() ];
   RegionNode const *regA = _regionList[ regionIdA ].getLeaf();
   RegionNode const *regB = _regionList[ regionIdB ].getLeaf();

   //std::cerr << "Computing intersect between reg " << regionIdA << " and "<< regionIdB << "... ";
   //printRegion(regionIdA ); std::cerr << std::endl;
   //printRegion(regionIdB ); std::cerr << std::endl;

   for ( int dimensionCount = _dimensionSizes.size() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {
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

      resultingRegion[ dimensionCount ].accessed_length = accessedLengthC;
      resultingRegion[ dimensionCount ].lower_bound = lowerBoundC;

      regA = regA->getParent();
      regB = regB->getParent();
   }

   reg_t id = _root.addNode( resultingRegion, _dimensionSizes.size(), 0, *this );
   //std::cerr << " result is"<< id << std::endl;
   return id;
}
*/

bool RegionDictionary::checkIntersect( reg_t regionIdA, reg_t regionIdB ) const {
   if ( regionIdA == regionIdB ) {
            std::cerr << __FUNCTION__ << " Dummy check! regA == regB" << std::endl;
   }

   {
      reg_t maxRegionId = std::max( regionIdA, regionIdB );
      reg_t minRegionId = std::min( regionIdA, regionIdB );
      RegionNode const *maxReg = getLeafRegionNode( maxRegionId );
      reg_t data = maxReg->getMemoIntersect( minRegionId );
      if ( data != 0 ) {
         //std::cerr << "hit!"<<std::endl;
         return ( data != (unsigned int) -1 );
      }
   }

   RegionNode const *regA = getLeafRegionNode( regionIdA );
   RegionNode const *regB = getLeafRegionNode( regionIdB );
   bool result = true;

   //std::cerr << "Computing intersect between reg " << regionIdA << " and "<< regionIdB << std::endl;

   for ( int dimensionCount = getNumDimensions() - 1; dimensionCount >= 0 && result; dimensionCount -= 1 ) {
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
      RegionNode *maxReg = getLeafRegionNode( maxRegionId );
      maxReg->setMemoIntersect( minRegionId, result ? -2 : -1  );
   }
   

   return result;
}

RegionVector::RegionVector( ) : _regionList( MAX_REG_ID, RegionVectorEntry() ), _maxId( 0 ), _leafCount( 0 ) {
}

RegionNode *RegionVector::getLeaf( reg_t id ) const {
   return _regionList[ id ].getLeaf();
}

void RegionDictionary::addLeaf( RegionNode *leaf ) {
   _regionContainer.addLeaf( leaf, _rogue );
}

void RegionVector::addLeaf( RegionNode *leaf, bool rogue ) {
   _regionList[ leaf->getId() ].setLeaf( leaf );
   _maxId = std::max( _maxId, leaf->getId() );
   if (!rogue) _leafCount++;
}

Version *RegionVector::getRegionData( reg_t id ) {
   return _regionList[ id ].getData();
}

void RegionVector::setRegionData( reg_t id, Version *data ) {
   _regionList[ id ].setData( data );
}

unsigned int RegionVector::getRegionCount() const {
   return _maxId;
}

unsigned int RegionVector::getLeafCount() const {
   return _leafCount.value();
}

RegionMap::RegionMap( RegionAssociativeContainer const &orig ) : _regionList(), _orig( orig ) {
}

RegionNode *RegionMap::getLeaf( reg_t id ) const {
   std::map< reg_t, RegionVectorEntry >::const_iterator it = _regionList.lower_bound( id );
   if ( it == _regionList.end() || _regionList.key_comp()(id, it->first) ) {
      //fatal0( "Error, RegionMap::getLeaf does not contain region" );
      return _orig.getLeaf( id );
   }
   return it->second.getLeaf();
}

void RegionMap::addLeaf( RegionNode *leaf, bool rogue ) {
   _regionList[ leaf->getId() ].setLeaf( leaf );
}

Version *RegionMap::getRegionData( reg_t id ) {
   std::map< reg_t, RegionVectorEntry >::iterator it = _regionList.lower_bound( id );
   if ( it == _regionList.end() || _regionList.key_comp()(id, it->first) ) {
      //fatal0(  "Error, RegionMap::getRegionData does not contain region " );
      it = _regionList.insert( it, std::map< reg_t, RegionVectorEntry >::value_type( id, RegionVectorEntry() ) );
      it->second.setLeaf( _orig.getLeaf( id ) );
   }
   return it->second.getData();
}

void RegionMap::setRegionData( reg_t id, Version *data ) {
   _regionList[ id ].setData( data );
}

unsigned int RegionMap::getRegionCount() const {
   return _regionList.size();
}

unsigned int RegionMap::getLeafCount() const {
   return _regionList.size();
}
reg_t RegionDictionary::getMaxRegionId() const {
   return _tree.getMaxRegionId();
}

unsigned int RegionDictionary::getRegionCount() const {
   return _regionContainer.getRegionCount();
}
unsigned int RegionDictionary::getLeafCount() const {
   return _regionContainer.getLeafCount();
}

//void RegionDictionary::addSubRegion( RegionDictionary &dict, std::list< std::pair< reg_t, reg_t > > partsList, reg_t regionToInsert ) {
//   ensure( !partsList.empty(), "Empty parts list!" );
//   
//   std::list< std::pair< reg_t, reg_t > > intersectList;
//   std::list< std::pair< reg_t, reg_t > >::iterator it = partsList.begin();
//   while ( it != partsList.end() ) {
//      if ( dict.checkIntersect( it->first, regionToInsert ) ) {
//         intersectList.push_back( *it );
//         it = partsList.erase( it );
//      } else {
//         it++;
//      }
//   }
//
//   for ( it = intersectList.begin(); it != intersectList.end(); it++ ) {
//      //reg_t intersection = dict.addIntersect( it->first, regionToInsert );
//      std::list<reg_t> pieces;
//      dict.substract( it->first, regionToInsert, pieces );
//      //partsList.push_back( std::make_pair( intersection, it->second ) );
//   }
//}

void RegionDictionary::printRegion( reg_t region ) const {
   RegionNode const *regNode = getLeafRegionNode( region );
   if ( regNode == NULL ) {
      std::cerr <<"NULL LEAF !";
      return;
   }
   for ( int dimensionCount = getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {  
      std::size_t accessedLength = regNode->getValue();
      regNode = regNode->getParent();
      std::size_t lowerBound = regNode->getValue();
      std::cerr << "[" << lowerBound <<";" <<accessedLength<< "]";
      regNode = regNode->getParent();
   }
}

void RegionDictionary::substract( reg_t base, reg_t regionToSubstract, std::list< reg_t > &resultingPieces ) {

   //std::cerr << __FUNCTION__ << ": base("<< base << ") "; printRegion(base); std::cerr<< " regToSubs(" << regionToSubstract<< ") "; printRegion(regionToSubstract); std::cerr << std::endl;
   if ( !checkIntersect( base, regionToSubstract ) ) {
      return;
   }

   nanos_region_dimension_internal_t fragments[ getNumDimensions() ][ 3 ];
   RegionNode const *regBase   = getLeafRegionNode( base );
   RegionNode const *regToSubs = getLeafRegionNode( regionToSubstract );

   for ( int dimensionCount = getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {
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
          accessedLengthIntersect = upperBoundToSubs - lowerBoundBase;

          lowerBoundGTIntersect = upperBoundToSubs;
          accessedLengthGTIntersect = upperBoundBase - upperBoundToSubs;
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

   nanos_region_dimension_internal_t tmpFragment[ getNumDimensions() ];

   _combine( tmpFragment, getNumDimensions()-1, 0, fragments, false, resultingPieces );
   _combine( tmpFragment, getNumDimensions()-1, 1, fragments, true, resultingPieces );
   _combine( tmpFragment, getNumDimensions()-1, 2, fragments, false, resultingPieces );


}
void RegionDictionary::_combine ( nanos_region_dimension_internal_t tmpFragment[], int dim, int currentPerm, nanos_region_dimension_internal_t fragments[ ][3], bool allFragmentsIntersect, std::list< reg_t > &resultingPieces  ) {
   //std::cerr << "dim "<<dim << " currentPerm " << currentPerm<<std::endl;;
   if ( fragments[ dim ][ currentPerm ].accessed_length > 0 ) { 
      tmpFragment[ dim ].accessed_length = fragments[ dim ][ currentPerm ].accessed_length;
      tmpFragment[ dim ].lower_bound     = fragments[ dim ][ currentPerm ].lower_bound;
      //for( unsigned int _i = 0; _i < ( getDimensionSizes().size() - (1+dim) ); _i++ ) std::cerr << ">";
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
            //for ( unsigned int i = 0; i < getDimensionSizes().size(); i++ ) {
            //   std::cerr << "["<<tmpFragment[getDimensionSizes().size()-(i+1)].lower_bound<< ":" << tmpFragment[getDimensionSizes().size()-(i+1)].accessed_length << "]";
            //}
            //std::cerr << std::endl;
            if ( _rogue ) _rogueLock->acquire();
            reg_t id = _tree.addRegion( tmpFragment, *this );
            if ( _rogue ) _rogueLock->release();
            //std::cerr << "computed a subchunk " << id ;printRegion( id ); std::cerr << std::endl;
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

uint64_t RegionDictionary::getBaseAddress() const {
   return _baseAddress;
}

std::vector< std::size_t > const &RegionTreeRoot::getDimensionSizes() const {
   return _dimensionSizes;
}

std::vector< std::size_t > const &RegionDictionary::getDimensionSizes() const {
   return _tree.getDimensionSizes();
}

uint64_t global_reg_t::getFirstAddress() const {
   RegionNode *n = key->getLeafRegionNode( id );
   uint64_t baseAddress = key->getBaseAddress();
   uint64_t offset = 0;
   std::vector< std::size_t > const &sizes = key->getDimensionSizes();
   uint64_t acumSizes = 1;

   for ( unsigned int dimIdx = 0; dimIdx < key->getNumDimensions() - 1; dimIdx += 1 ) {
      acumSizes *= sizes[ dimIdx ];
   }
   
   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      //std::size_t accessedLength = n->getValue();
      n = n->getParent();
      std::size_t lowerBound = n->getValue();
      n = n->getParent();
      offset += acumSizes * lowerBound;
      if ( dimIdx >= 1 ) acumSizes = acumSizes / sizes[ dimIdx - 1 ];
   }
   return baseAddress + offset; 
}

std::size_t global_reg_t::getBreadth() const {
   RegionNode *n = key->getLeafRegionNode( id );
   //uint64_t baseAddress = key->getBaseAddress();
   std::size_t offset = 0;
   std::size_t lastOffset = 0;
   std::vector< std::size_t > const &sizes = key->getDimensionSizes();
   uint64_t acumSizes = 1;

   for ( unsigned int dimIdx = 0; dimIdx < key->getNumDimensions() - 1; dimIdx += 1 ) {
      acumSizes *= sizes[ dimIdx ];
   }
   
   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      std::size_t accessedLength = n->getValue();
      n = n->getParent();
      std::size_t lowerBound = n->getValue();
      n = n->getParent();
      offset += acumSizes * lowerBound;
      lastOffset += acumSizes * ( lowerBound + accessedLength - 1 );
      if ( dimIdx >= 1 ) acumSizes = acumSizes / sizes[ dimIdx - 1 ];
   }
   return lastOffset - offset; 
}

std::size_t global_reg_t::getDataSize() const {
   RegionNode *n = key->getLeafRegionNode( id );
   //uint64_t baseAddress = key->getBaseAddress();
   std::size_t dataSize = 1;

   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      std::size_t accessedLength = n->getValue();
      n = n->getParent();
      n = n->getParent();
      dataSize *= accessedLength;
   }
   return dataSize; 
}

unsigned int global_reg_t::getNumDimensions() const {
   return key->getNumDimensions();
}

global_reg_t::global_reg_t( reg_t r, reg_key_t k ) : id( r ), key( k ) {
}

global_reg_t::global_reg_t() : id( 0 ), key( NULL ) {
}

void global_reg_t::fillDimensionData( nanos_region_dimension_internal_t region[]) const {
   RegionNode *n = key->getLeafRegionNode( id );
   std::vector< std::size_t > const &sizes = key->getDimensionSizes();
   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      std::size_t accessedLength = n->getValue();
      n = n->getParent();
      std::size_t lowerBound = n->getValue();
      n = n->getParent();
      region[ dimIdx ].accessed_length = accessedLength;
      region[ dimIdx ].lower_bound = lowerBound;
      region[ dimIdx ].size = sizes[ dimIdx ];
   }
}

