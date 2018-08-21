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
#include "lock_decl.hpp"
#include "recursivelock_decl.hpp"
#include "version_decl.hpp"
#include "workdescriptor_fwd.hpp"

#define MAX_REG_ID (1024*1024)

namespace nanos {

   template < class > class ContainerDense;

   class RegionVectorEntry;

   class RegionNode {
      RegionNode  *_parent;
      std::size_t  _value;
      reg_t _id;
      std::map<std::size_t, RegionNode *> *_sons;
      reg_t *_memoIntersectInfo;

      public:
      RegionNode( RegionNode *parent, std::size_t value, reg_t id );
      RegionNode( RegionNode const & rn );
      RegionNode &operator=( RegionNode const & rn );
      ~RegionNode();
      reg_t getId() const;
      reg_t addNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep, ContainerDense< RegionVectorEntry > &container );
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
      std::vector< T >           _container;
      Atomic<unsigned int>       _leafCount;
      Atomic<reg_t>              _idSeed;
      std::vector< std::size_t > _dimensionSizes;
      RegionNode                 _root;
      pthread_rwlock_t           _containerLock;


      Lock                       _invalidationsLock;
      std::map< reg_t, reg_t >   _masterIdToLocalId;
      Lock                       _containerMi2LiLock;
      bool                       _keepAtOrigin;
      CopyData                  *_registeredObject;
      public:
      bool sparse;
      ContainerDense( CopyData const &cd );
      ~ContainerDense();
      RegionNode *getRegionNode( reg_t id );
      void addRegionNode( RegionNode *leaf );
      Version *getRegionData( reg_t id );
      void setRegionData( reg_t id, Version * );
      unsigned int getRegionNodeCount() const;
      void lockContainer();
      void releaseContainer();

      unsigned int getNumDimensions() const;
      reg_t getNewRegionId();
      reg_t addRegion( nanos_region_dimension_internal_t const region[] );
      reg_t checkIfRegionExists( nanos_region_dimension_internal_t const region[] );
      reg_t getMaxRegionId() const;
      std::vector< std::size_t > const &getDimensionSizes() const;
      void invalLock();
      void invalUnlock();
      void addMasterRegionId( reg_t masterId, reg_t localId );
      reg_t getLocalRegionIdFromMasterRegionId( reg_t masterId );
      void setKeepAtOrigin( bool value );
      bool getKeepAtOrigin() const;
      void setRegisteredObject( CopyData *cd );
      CopyData *getRegisteredObject() const;
   };

   template < template <class> class > class RegionDictionary;
   template < class T >
   class ContainerSparse {
      std::map< reg_t, T > _container;
      Lock                       _containerLock;
      //ContainerDense< T > &_orig;
      protected:
      RegionDictionary< ContainerDense > &_orig;
      public:
      bool sparse;
      ContainerSparse( RegionDictionary< ContainerDense > &orig );
      ~ContainerSparse();
      RegionNode *getRegionNode( reg_t id ) const;
      Version *getRegionData( reg_t id );
      void setRegionData( reg_t id, Version * );
      unsigned int getRegionNodeCount() const;

      reg_t addRegion( nanos_region_dimension_internal_t const region[] );
      unsigned int getNumDimensions() const;
      reg_t checkIfRegionExists( nanos_region_dimension_internal_t const region[] );
      ContainerDense< T > &getOrigContainer();
      reg_t getMaxRegionId() const;

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
   class RegionDictionary : public Sparsity< RegionVectorEntry >, public Version {
      std::vector< MemoryMap< std::set< reg_t > > > _intersects;
      uint64_t _keyBaseAddress;
      uint64_t _realBaseAddress;
      RecursiveLock _lock;

      /* this should be on a different class, for global objects */
      std::set< reg_t > _fixedRegions;

      public:
      void addRegionAndComputeIntersects( reg_t id, std::list< std::pair< reg_t, reg_t > > &finalParts, unsigned int &version );
      void lockObject();
      bool tryLockObject();
      void unlockObject();

      typedef std::list< reg_t > RegionList;

      RegionDictionary( CopyData const &cd );
      RegionDictionary( GlobalRegionDictionary &dict );
      ~RegionDictionary();
      reg_t registerRegion( reg_t, std::list< std::pair< reg_t, reg_t > > &missingParts, unsigned int &version );
      reg_t obtainRegionId( CopyData &cd, WorkDescriptor const &wd, unsigned int idx );
      reg_t obtainRegionId( nanos_region_dimension_internal_t const region[] );
      //reg_t tryObtainRegionId( CopyData const &cd );
      void addLeaf( RegionNode *leaf );

      uint64_t getKeyBaseAddress() const;
      uint64_t getRealBaseAddress() const;

      void printRegion( std::ostream &o, reg_t );
      void printRegionGeom( std::ostream &o, reg_t );

      bool checkIntersect( reg_t baseRegionId, reg_t targetRegionId );
      reg_t computeTestIntersect( reg_t regionIdA, reg_t regionIdB ) ;
      reg_t computeIntersect( reg_t regionIdA, reg_t regionIdB ) ;
      void _computeIntersect( reg_t regionIdA, reg_t regionIdB, nanos_region_dimension_internal_t *outReg );

      void substract( reg_t base, reg_t regionToSubstract, std::list< reg_t > &resultingPieces );
      void _combine ( nanos_region_dimension_internal_t tmpFragment[], int dim, int currentPerm, nanos_region_dimension_internal_t fragments[][3], bool allFragmentsIntersect, std::list< reg_t > &resultingPieces );
      reg_t isThisPartOf( reg_t target, std::map< reg_t, unsigned int >::const_iterator begin, std::map< reg_t, unsigned int >::const_iterator end, unsigned int &version );
      bool doTheseRegionsForm( reg_t target, std::map< reg_t, unsigned int >::const_iterator begin, std::map< reg_t, unsigned int >::const_iterator end, unsigned int &version ) ;
      bool doTheseRegionsForm( reg_t target, std::list< std::pair< reg_t, reg_t > >::const_iterator ibegin, std::list< std::pair< reg_t, reg_t > >::const_iterator iend, bool checkVersion ) ;

      std::set< reg_t > const &getFixedRegions() const;
      void addFixedRegion( reg_t id );

   };
   
   typedef RegionDictionary<ContainerSparse>::citerator CacheRegionDictionaryIterator;
   std::ostream& operator<< (std::ostream& o, RegionNode const &rn);

} // namespace nanos

#endif /* REGIONDICTIONARY_DECL_HPP */
