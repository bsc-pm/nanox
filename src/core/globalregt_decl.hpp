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

#ifndef GLOBALREGT_DECL
#define GLOBALREGT_DECL
#include "nanos-int.h"
#include "regiondict_decl.hpp"

#include "addressspace_fwd.hpp"
#include "deviceops_decl.hpp"
#include "processingelement_fwd.hpp"

namespace nanos {


typedef GlobalRegionDictionary *reg_key_t;
typedef GlobalRegionDictionary const * const_reg_key_t;

struct global_reg_t {
   reg_t id;
   union {
      reg_key_t key;
      const_reg_key_t ckey;
   };
   global_reg_t();
   global_reg_t( reg_t r, reg_key_t k );
   global_reg_t( reg_t r, const_reg_key_t k );
   uint64_t getKeyFirstAddress() const;
   uint64_t getRealFirstAddress() const;
   std::size_t getBreadth() const;
   std::size_t getDataSize() const;
   unsigned int getNumDimensions() const;
   void fillDimensionData( nanos_region_dimension_internal_t *region) const;
   bool operator<( global_reg_t const &reg ) const;
   bool operator!=( global_reg_t const &reg ) const;
   memory_space_id_t getFirstLocation() const;
   memory_space_id_t getPreferedSourceLocation( memory_space_id_t dest ) const;
   unsigned int getVersion() const;
   unsigned int getHostVersion( bool increaseVersion ) const;
   reg_t getFitRegionId() const;
   reg_t getSlabRegionId( std::size_t slabSize ) const;
   uint64_t getRealBaseAddress() const;
   DeviceOps *getDeviceOps() const;
   DeviceOps *getHomeDeviceOps( WD const &wd, unsigned int copyIdx );
   void setLocationAndVersion( ProcessingElement *pe, memory_space_id_t loc, unsigned int version ) const;
   bool contains( global_reg_t const &reg ) const;
   bool isLocatedIn( memory_space_id_t loc ) const;
   void fillCopyData( CopyData &cd, uint64_t baseAddress ) const;
   bool isRegistered() const;
   std::set< memory_space_id_t > const &getLocations() const;
   //void setRooted() const;
   bool isRooted() const;
   memory_space_id_t getRootedLocation() const;
   void setOwnedMemory( memory_space_id_t loc ) const;
   unsigned int getNumLocations() const;
   ProcessingElement *getFirstWriterPE() const;
   uint64_t getFirstAddress(uint64_t baseAddress) const;
   bool isLocatedInSeparateMemorySpaces() const;
};

} // namespace nanos

#endif /* GLOBALREGT_DECL */
