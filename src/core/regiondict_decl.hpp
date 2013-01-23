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

namespace nanos {

   class RegionDictionary;

   typedef unsigned int reg_t;
   typedef RegionDictionary *reg_key_t;
   struct global_reg_t {
      reg_t id;
      reg_key_t key;
      global_reg_t();
      global_reg_t( reg_t r , reg_key_t k );
      uint64_t getFirstAddress() const;
      std::size_t getBreadth() const;
      std::size_t getDataSize() const;
      unsigned int getNumDimensions() const;
      void fillDimensionData( nanos_region_dimension_internal_t region[]) const;
      bool operator<( global_reg_t const &reg ) const;
      /*
      struct DimensionData {
         std::size_t lowerBound;
         std::size_t accessedLength;
      }
      class const_iterator {
         RegionNode *_currentNode;

         operator()
        
      }*/
   };

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
      reg_t addNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep, RegionDictionary & dict );
      reg_t checkNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep );
      std::size_t getValue() const;
      RegionNode *getParent() const;

      reg_t getMemoIntersect( reg_t target ) const;
      void setMemoIntersect( reg_t target, reg_t value ) const;
   };

   class RegionIntersectionDictionary {
      RegionDictionary &_dict;
      std::vector< MemoryMap< std::set< reg_t > > > _intersects;

      public:
      RegionIntersectionDictionary( RegionDictionary &d );
      void addRegionAndComputeIntersects( reg_t id, std::list< std::pair< reg_t, reg_t > > &finalParts, unsigned int &version, bool rogue, bool justCreatedRegion, bool superPrecise = false );
   };

   class RegionAssociativeContainer {
      protected:
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
      public:
      virtual RegionNode *getLeaf( reg_t id ) const = 0;
      virtual void addLeaf( RegionNode *leaf, bool rogue ) = 0;
      virtual Version *getRegionData( reg_t id ) = 0;
      virtual void setRegionData( reg_t id, Version * ) = 0;
      virtual unsigned int getRegionCount() const = 0;
      virtual unsigned int getLeafCount() const = 0;
   };
   
   class RegionVector : public RegionAssociativeContainer {
      std::vector< RegionVectorEntry > _regionList;
      unsigned int _maxId;
      Atomic<unsigned int> _leafCount;
      public:
      RegionVector();
      virtual RegionNode *getLeaf( reg_t id ) const;
      virtual void addLeaf( RegionNode *leaf, bool rogue );
      virtual Version *getRegionData( reg_t id );
      virtual void setRegionData( reg_t id, Version * );
      virtual unsigned int getRegionCount() const;
      virtual unsigned int getLeafCount() const;
   };

   class RegionMap : public RegionAssociativeContainer {
      std::map< reg_t, RegionVectorEntry > _regionList;
      RegionAssociativeContainer    const &_orig;
      public:
      RegionMap( RegionAssociativeContainer const &orig );
      virtual RegionNode *getLeaf( reg_t id ) const;
      virtual void addLeaf( RegionNode *leaf, bool rogue );
      virtual Version *getRegionData( reg_t id );
      virtual void setRegionData( reg_t id, Version * );
      virtual unsigned int getRegionCount() const;
      virtual unsigned int getLeafCount() const;
   };

   class RegionTreeRoot {
      Atomic<reg_t>                                 _idSeed;
      std::vector< std::size_t >                    _dimensionSizes;
      RegionNode                                    _root;

      public:
      RegionTreeRoot( CopyData const &cd );
      unsigned int getNumDimensions() const;
      reg_t getNewRegionId();
      reg_t addRegion( nanos_region_dimension_internal_t const region[], RegionDictionary &dict );
      reg_t checkIfRegionExists( nanos_region_dimension_internal_t const region[] );
      reg_t getMaxRegionId() const;
      std::vector< std::size_t > const &getDimensionSizes() const;
   };
 
   class RegionDictionary {
      RegionTreeRoot                               &_tree;
      RegionIntersectionDictionary                  _intersects;
      RegionAssociativeContainer                   &_regionContainer;
      uint64_t _baseAddress;
      bool _rogue;
      Lock _lock;
      Lock *_rogueLock;

      public:
      void lock();
      bool tryLock();
      void unlock();

      typedef std::list< reg_t > RegionList;

      RegionDictionary( CopyData const &cd, RegionAssociativeContainer &container, bool rogue = false );
      RegionDictionary( RegionDictionary &dict, RegionAssociativeContainer &container , bool rogue = false);
      reg_t addRegion( CopyData const &cd, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version );
      reg_t addRegion( reg_t, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version, bool superPrecise = false );
      reg_t obtainRegionId( CopyData const &cd );
      reg_t tryObtainRegionId( CopyData const &cd );
      reg_t getNewRegionId();
      void addLeaf( RegionNode *leaf );

      //void getRegions( reg_t id, std::list<reg_t> &regs );
      
      uint64_t getBaseAddress() const;
      unsigned int getNumDimensions() const;
      std::vector< std::size_t > const &getDimensionSizes() const;
      reg_t addRegionByComponents( nanos_region_dimension_internal_t const region[] );
      reg_t checkIfRegionExistsByComponents( nanos_region_dimension_internal_t const region[] );
      RegionNode *getLeafRegionNode( reg_t id ) const;

      RegionAssociativeContainer const &getContainer() const;

      Version *getRegionData( reg_t id ) const;
      void setRegionData( reg_t id, Version *data );
      reg_t getMaxRegionId() const;
      unsigned int getRegionCount() const;
      unsigned int getLeafCount() const;
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
   
   std::ostream& operator<< (std::ostream& o, RegionNode const &rn);
}
#endif /* REGIONDICTIONARY_DECL_HPP */
