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
      unsigned int getNumDimensions() const;
      void fillDimensionData( nanos_region_dimension_internal_t region[]) const;

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

      public:
      RegionNode( RegionNode *parent, std::size_t value, reg_t id );
      ~RegionNode();
      reg_t getId() const;
      reg_t addNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep, RegionDictionary & dict );
      std::size_t getValue() const;
      RegionNode *getParent() const;
   };

   class RegionIntersectionDictionary {
      RegionDictionary &_dict;
      std::vector< MemoryMap< std::set< reg_t > > > _intersects;

      public:
      RegionIntersectionDictionary( RegionDictionary &d );
      void addRegionAndComputeIntersects( reg_t id, std::list< std::pair< reg_t, reg_t > > &finalParts, unsigned int &version );
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
      virtual void addLeaf( RegionNode *leaf ) = 0;
      virtual Version *getRegionData( reg_t id ) = 0;
      virtual void setRegionData( reg_t id, Version * ) = 0;
   };
   
   class RegionVector : public RegionAssociativeContainer {
      std::vector< RegionVectorEntry > _regionList;
      public:
      RegionVector();
      virtual RegionNode *getLeaf( reg_t id ) const;
      virtual void addLeaf( RegionNode *leaf );
      virtual Version *getRegionData( reg_t id );
      virtual void setRegionData( reg_t id, Version * );
   };

   class RegionMap : public RegionAssociativeContainer {
      std::map< reg_t, RegionVectorEntry > _regionList;
      RegionAssociativeContainer    const &_orig;
      public:
      RegionMap( RegionAssociativeContainer const &orig );
      virtual RegionNode *getLeaf( reg_t id ) const;
      virtual void addLeaf( RegionNode *leaf );
      virtual Version *getRegionData( reg_t id );
      virtual void setRegionData( reg_t id, Version * );
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
      reg_t getMaxRegionId() const;
      std::vector< std::size_t > const &getDimensionSizes() const;
   };
 
   class RegionDictionary {
      RegionTreeRoot                               &_tree;
      RegionIntersectionDictionary                  _intersects;
      RegionAssociativeContainer                   &_regionContainer;
      uint64_t _baseAddress;

      public:

      typedef std::list< reg_t > RegionList;

      RegionDictionary( CopyData const &cd, RegionAssociativeContainer &container );
      RegionDictionary( RegionDictionary const &dict, RegionAssociativeContainer &container );
      reg_t addRegion( CopyData const &cd, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version );
      reg_t addRegion( reg_t, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version );
      reg_t getNewRegionId();
      void addLeaf( RegionNode *leaf );

      //void getRegions( reg_t id, std::list<reg_t> &regs );
      
      uint64_t getBaseAddress() const;
      unsigned int getNumDimensions() const;
      std::vector< std::size_t > const &getDimensionSizes() const;
      reg_t addRegionByComponents( nanos_region_dimension_internal_t const region[] );
      RegionNode *getLeafRegionNode( reg_t id ) const;

      RegionAssociativeContainer const &getContainer() const;

      Version *getRegionData( reg_t id ) const;
      void setRegionData( reg_t id, Version *data );
      reg_t getMaxRegionId() const;
      void printRegion( reg_t ) const;

      bool checkIntersect( reg_t baseRegionId, reg_t targetRegionId ) const;
      void substract( reg_t base, reg_t regionToSubstract, std::list< reg_t > &resultingPieces );
      void _combine ( nanos_region_dimension_internal_t tmpFragment[], int dim, int currentPerm, nanos_region_dimension_internal_t fragments[][3], bool allFragmentsIntersect, std::list< reg_t > &resultingPieces );
   };
   
   std::ostream& operator<< (std::ostream& o, RegionNode const &rn);
}
#endif /* REGIONDICTIONARY_DECL_HPP */
