/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

inline void * TaskReduction::have( const void *ptr, size_t id )
{
   bool inside =  ( ( ptr == _dependence ) || ( (ptr >= _min) && (ptr <= _max) ) );

   if ( inside ) return & _storage[_size_target*id];
   else return NULL;
}

inline void * TaskReduction::finalize( void )
{
   void * result = _original;

   //For each thread
   for ( size_t i=1; i<_num_threads; i++)
	   //Reduce all elements
	   for(size_t j=0; j<_num_elements; j++ )
		 {
			   _reducer( &_storage[j*_size_element] ,&_storage[i*_size_target + j*_size_element] );
		 }

   //reduce all to original
   for(size_t j=0; j<_num_elements; j++ )
	  _reducer_orig_var( &((char*)_original)[j * _size_element], &_storage[j * _size_element] );
   return result;
}

inline unsigned TaskReduction::getDepth( void ) const
{
   return _depth;
}

#endif
