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

#ifndef _NANOS_DEPENDENCIES_DOMAIN
#define _NANOS_DEPENDENCIES_DOMAIN
#include <stdlib.h>
#include <map>
#include <list>
#include <vector>
#include "dependenciesdomain_decl.hpp"
#include "atomic.hpp"
#include "recursivelock_decl.hpp"
#include "lock.hpp"
#include "dependableobject.hpp"
#include "trackableobject.hpp"
#include "dataaccess_decl.hpp"
#include "commutationdepobj.hpp"


namespace nanos {

inline DependenciesDomain::~DependenciesDomain ( )
{
}

inline RecursiveLock& DependenciesDomain::getInstanceLock()
{
   return _instanceLock;
}

inline Lock& DependenciesDomain::getLock()
{
   return _lock;
}

inline void DependenciesDomain::lock ( )
{
   _lock.acquire();
   memoryFence();
}

inline void DependenciesDomain::unlock ( )
{
   memoryFence();
   _lock.release();
}

inline const std::string & DependenciesManager::getName () const
{
   return _name;
}

inline bool DependenciesDomain::haveDependencePendantWrites ( void *addr )
{
   fatal("haveDependencePendantWrites service has not been implemented in that dependence plugin!");
   return true;
}
inline void DependenciesDomain::finalizeAllReductions ( void ) 
{
}

inline void DependenciesDomain::clearDependenciesDomain ( void ) { }

} // namespace nanos

#endif
