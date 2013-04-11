#ifndef REGIONDICTIONARY_DECL_HPP
#define REGIONDICTIONARY_DECL_HPP

#include <map>
#include <vector>
#include <set>
#include <list>
#include <iostream>

#include "copydata.hpp"
#include "memorymap_decl.hpp"
#include "atomic_decl.hpp"
#include "version_decl.hpp"

#define MAX_REG_ID 16384

namespace nanos {

typedef unsigned int reg_t;

   template < class > class ContainerDense;

   class RegionVectorEntry;

   class RegionNode {
      RegionNode  *_parent;
      std::size_t  _value;
      reg_t _id;
      std::map<std::size_t, RegionNode> *_sons;
      reg_t *_memoIntersectInfo;

      public:
      RegionNode( RegionNode *parent, std::size_t value, reg_t id );
      ~RegionNode();
      reg_t getId() const;
      reg_t addNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep, ContainerDense< RegionVectorEntry > &container, bool rogue);
      reg_t checkNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep );
      std::size_t getValue() const;
      RegionNode *getParent() const;

      reg_t getMemoIntersect( reg_t target ) const;
      void setMemoIntersect( reg_t target, reg_t value ) const;
   };

   class RegionVectorEntry {
      RegionNode *_leaf;
      Version    *_data;

      public:
      RegionVectorEntry();
      RegionVectorEntry( RegionVectorEntry const &rve );
      RegionVectorEntry &operator=( RegionVectorEntry const &rve );
      ~RegionVectorEntry();

      void setLeaf( RegionNode *rm );
      RegionNode *getLeaf() const;
      void setData( Version *d );
      Version *getData() const;
   };

   template < class T >
   class ContainerDense {
      std::vector< T >            _container;
      Atomic<unsigned int>        _leafCount;
      Atomic<reg_t>               _idSeed;
      std::vector< std::size_t >  _dimensionSizes;
      RegionNode                  _root;
      Lock                        _rogueLock;
      Lock                        _lock;
      public:
      bool sparse;
      ContainerDense( CopyData const &cd );
      RegionNode *getRegionNode( reg_t id ) const;
      void addRegionNode( RegionNode *leaf, bool rogue );
      Version *getRegionData( reg_t id );
      void setRegionData( reg_t id, Version * );
      unsigned int getRegionNodeCount() const;

      unsigned int getNumDimensions() const;
      reg_t getNewRegionId();
      reg_t addRegion( nanos_region_dimension_internal_t const region[], bool rogue=false );
      reg_t checkIfRegionExists( nanos_region_dimension_internal_t const region[] );
      reg_t getMaxRegionId() const;
      std::vector< std::size_t > const &getDimensionSizes() const;
   };

   template < template <class> class > class RegionDictionary;
   template < class T >
   class ContainerSparse {
      std::map< reg_t, T > _container;
      //ContainerDense< T > &_orig;
      RegionDictionary< ContainerDense > &_orig;
      public:
      bool sparse;
      ContainerSparse( RegionDictionary< ContainerDense > &orig );
      RegionNode *getRegionNode( reg_t id ) const;
      void addRegionNode( RegionNode *leaf, bool rogue );
      Version *getRegionData( reg_t id );
      void setRegionData( reg_t id, Version * );
      unsigned int getRegionNodeCount() const;

      reg_t addRegion( nanos_region_dimension_internal_t const region[] );
      unsigned int getNumDimensions() const;
      reg_t checkIfRegionExists( nanos_region_dimension_internal_t const region[] );
      ContainerDense< T > &getOrigContainer();

      Version *getGlobalRegionData( reg_t id );
      RegionNode *getGlobalRegionNode( reg_t id ) const;
      RegionDictionary< ContainerDense > *getGlobalDirectoryKey();
      std::vector< std::size_t > const &getDimensionSizes() const;

      typename std::map< reg_t, T >::const_iterator begin();
      typename std::map< reg_t, T >::const_iterator end();

      typedef typename std::map< reg_t, T >::const_iterator citerator;
   };

   typedef RegionDictionary<ContainerDense> GlobalRegionDictionary;
   typedef RegionDictionary<ContainerSparse> CacheRegionDictionary;

   template< template <class> class Sparsity >
   class RegionDictionary : public Sparsity< RegionVectorEntry > {
      std::vector< MemoryMap< std::set< reg_t > > > _intersects;
      uint64_t _baseAddress;
      Lock _lock;

      public:
      void addRegionAndComputeIntersects( reg_t id, std::list< std::pair< reg_t, reg_t > > &finalParts, unsigned int &version, bool superPrecise = false );
      void getRegionIntersects( reg_t id, unsigned int version, std::list< reg_t > &superParts, std::list< reg_t > &subParts );
      void lock();
      bool tryLock();
      void unlock();

      typedef std::list< reg_t > RegionList;

      RegionDictionary( CopyData const &cd );
      RegionDictionary( GlobalRegionDictionary &dict );
      reg_t registerRegion( CopyData const &cd, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version );
      reg_t registerRegion( reg_t, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version, bool superPrecise = false );
      reg_t obtainRegionId( CopyData const &cd );
      reg_t obtainRegionId( nanos_region_dimension_internal_t region[] );
      reg_t tryObtainRegionId( CopyData const &cd );
      void addLeaf( RegionNode *leaf );

      uint64_t getBaseAddress() const;

      void printRegion( reg_t ) const;

      bool checkIntersect( reg_t baseRegionId, reg_t targetRegionId ) const;
      reg_t computeTestIntersect( reg_t regionIdA, reg_t regionIdB ) ;
      reg_t computeIntersect( reg_t regionIdA, reg_t regionIdB ) ;
      void _computeIntersect( reg_t regionIdA, reg_t regionIdB, nanos_region_dimension_internal_t *outReg );

      void substract( reg_t base, reg_t regionToSubstract, std::list< reg_t > &resultingPieces );
      void _combine ( nanos_region_dimension_internal_t tmpFragment[], int dim, int currentPerm, nanos_region_dimension_internal_t fragments[][3], bool allFragmentsIntersect, std::list< reg_t > &resultingPieces );
      reg_t isThisPartOf( reg_t target, std::map< reg_t, unsigned int >::const_iterator begin, std::map< reg_t, unsigned int >::const_iterator end, unsigned int &version );
      bool doTheseRegionsForm( reg_t target, std::map< reg_t, unsigned int >::const_iterator begin, std::map< reg_t, unsigned int >::const_iterator end, unsigned int &version ) ;

   };
   
   typedef RegionDictionary<ContainerSparse>::citerator CacheRegionDictionaryIterator;
   std::ostream& operator<< (std::ostream& o, RegionNode const &rn);
}
#endif /* REGIONDICTIONARY_DECL_HPP */
