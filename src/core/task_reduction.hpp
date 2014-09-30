
/*************************************************************************************/
/*      Copyright 2012 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef _NANOS_TASK_REDUCTION_HPP
#define _NANOS_TASK_REDUCTION_HPP

#include "task_reduction_decl.hpp"

inline void * TaskReduction::have_dependence( const void *ptr, size_t id )
{
   bool inside =  ( ( ptr == _dependence ) || ( (ptr >= _min) && (ptr <= _max) ) );

   if ( inside ) return & _storage[_size*id];
   else return NULL;
}

inline void * TaskReduction::have ( const void *ptr, size_t id )
{
   bool inside =  ( ( ptr == _original ) || ( (ptr >= _min) && (ptr <= _max) ) );

   if ( inside ) return & _storage[_size*id];
   else return NULL;
}

inline void * TaskReduction::finalize ( void )
{
   void * result = _original;
   for ( size_t i=1; i< _threads; i++) _reducer( &_storage[0] ,&_storage[i*_size] );
   _reducer_orig_var( _original, &_storage[0] );
   return result;
}

inline unsigned TaskReduction::getDepth(void) const 
{
   return _depth;
}

#endif
