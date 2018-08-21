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

#ifndef MEMCONTROLLER_DECL
#define MEMCONTROLLER_DECL
#include <map>
#include "workdescriptor_fwd.hpp"
#include "atomic_decl.hpp"
#include "lock_decl.hpp"
#include "regiondirectory_decl.hpp"
#include "addressspace_decl.hpp"
#include "memoryops_decl.hpp"
#include "memcachecopy_decl.hpp"
#include "regionset_decl.hpp"

namespace nanos {

class MemController {
   bool                        _initialized;
   bool                        _preinitialized;
   bool                        _inputDataReady;
   bool                        _outputDataReady;
   bool                        _memoryAllocated;
   bool                        _invalidating;
   bool                        _mainWd;
   WD                         *_wd;
   ProcessingElement          *_pe;
   Lock                        _provideLock;
   //std::map< RegionDirectory::RegionDirectoryKey, std::map< reg_t, unsigned int > > _providedRegions;
   RegionSet _providedRegions;
   BaseAddressSpaceInOps      *_inOps;
   SeparateAddressSpaceOutOps *_outOps;
   std::size_t                 _affinityScore;
   std::size_t                 _maxAffinityScore;
   RegionSet _ownedRegions;
   RegionSet _parentRegions;

public:
   enum MemControllerPolicy {
      WRITE_BACK,
      WRITE_THROUGH,
      NO_CACHE
   };
   MemCacheCopy *_memCacheCopies;
   MemController( WD *wd, unsigned int numCopies );
   ~MemController();
   bool hasVersionInfoForRegion( global_reg_t reg, unsigned int &version, NewLocationInfoList &locations );
   void getInfoFromPredecessor( MemController const &predecessorController );
   void preInit();
   void initialize( ProcessingElement &pe );
   bool allocateTaskMemory();
   void copyDataIn();
   void copyDataOut( MemControllerPolicy policy );
   bool isDataReady( WD const &wd );
   bool isOutputDataReady( WD const &wd );
   uint64_t getAddress( unsigned int index ) const;
   bool canAllocateMemory( memory_space_id_t memId, bool considerInvalidations ) const;
   void setAffinityScore( std::size_t score );
   std::size_t getAffinityScore() const;
   void setMaxAffinityScore( std::size_t score );
   std::size_t getMaxAffinityScore() const;
   std::size_t getAmountOfTransferredData() const;
   std::size_t getTotalAmountOfData() const;
   bool isRooted( memory_space_id_t &loc ) const ;
   bool isMultipleRooted( std::list<memory_space_id_t> &locs ) const ;
   void setMainWD();
   void synchronize();
   void synchronize( std::size_t numDataAccesses, DataAccess *data);
   bool isMemoryAllocated() const;
   void setCacheMetaData();
   bool ownsRegion( global_reg_t const &reg );
   bool hasObjectOfRegion( global_reg_t const &reg );
   bool containsAllCopies( MemController const &target ) const;
};

} // namespace nanos
#endif /* MEMCONTROLLER_DECL */
