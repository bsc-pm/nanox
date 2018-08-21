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

#ifndef REGIONSET_DECL_HPP
#define REGIONSET_DECL_HPP

#include <map>
#include "atomic_decl.hpp"
#include "lock_decl.hpp"
#include "globalregt_decl.hpp"
#include "regiondirectory_decl.hpp"

namespace nanos {

class RegionSet {
   typedef std::map< reg_t, unsigned int > reg_set_t;
   typedef std::map< RegionDirectory::RegionDirectoryKey, reg_set_t > object_set_t;
   Lock         _lock;
   object_set_t _set;
   public:
   RegionSet();
   void addRegion( global_reg_t const &reg, unsigned int version );
   unsigned int hasRegion( global_reg_t const &reg );
   bool hasObjectOfRegion( global_reg_t const &reg );

   bool hasVersionInfoForRegion( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations );
};

} // namespace nanos

#endif /* REGIONSET_DECL_HPP */
