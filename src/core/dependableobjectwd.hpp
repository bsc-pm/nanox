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

#ifndef _NANOS_DEPENDABLE_OBJECT_WD
#define _NANOS_DEPENDABLE_OBJECT_WD

#include "dependableobjectwd_decl.hpp"
#include "dependableobject.hpp"

namespace nanos {

inline const DOSubmit & DOSubmit::operator= ( const DOSubmit &dos )
{
   if ( this == &dos ) return *this; 
   DependableObject::operator= (dos);
   return *this;
}

inline void * DOSubmit::getRelatedObject ( )
{
   return (void *) getWD();
}

inline const void * DOSubmit::getRelatedObject ( ) const
{
   return (void *) getWD();
}

inline const DOWait & DOWait::operator= ( const DOWait &dow )
{
   if ( this == &dow ) return *this; 
   DependableObject::operator= ( dow );
   _depsSatisfied = dow._depsSatisfied;
   return *this;
}

inline void * DOWait::getRelatedObject ( )
{
   return (void *) getWD();
}

inline const void * DOWait::getRelatedObject ( ) const
{
   return (void *) getWD();
}

} // namespace nanos

#endif

