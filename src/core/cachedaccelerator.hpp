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

#ifndef _NANOS_CACHED_ACCELERATOR
#define _NANOS_CACHED_ACCELERATOR

#include "cachedaccelerator_decl.hpp"
#include "accelerator_decl.hpp"
#include "regioncache.hpp"
#include "system.hpp"

namespace nanos {

//inline CachedAccelerator::CachedAccelerator( const Device *arch,
//   const Device *subArch, memory_space_id_t addressSpace ) :
//   Accelerator( arch, subArch ), _addressSpaceId( addressSpace ) {
//}
//
//inline CachedAccelerator::~CachedAccelerator() {
//}
//
//inline void CachedAccelerator::waitInputsDependent( WorkDescriptor &wd )
//{
//   while ( !wd._mcontrol.isDataReady( wd ) ) { myThread->idle(); } 
//}

} // namespace nanos

#endif
