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

inline bool TaskReduction::has( const void *ptr)
{
   bool inside =  ( ( ptr == _dependence ) /*|| ( (ptr >= _min) && (ptr <= _max) )*/ );
   if ( inside ) return true;//_storage[id];
   return false;
}

inline void * TaskReduction::get( size_t id )
{
   return _storage[id];
}

inline void * TaskReduction::finalize( void )
{
	NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 2); )
	void * result = _original;
	//For each thread
        for ( size_t i=0; i<_num_threads; i++) {
           if ( _storage[i] != NULL ) {
              //Reduce all elements
              for( size_t j=0; j<_num_elements; j++ )
              {
                 _reducer( &((char*)_original)[j*_size_element] ,& ((char*)(_storage[i]))[j*_size_element]);
              }
              free(_storage[i]);
           }
        }
   	NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 0 ); )
	return result;
}

inline  void * TaskReduction::init( size_t id )
{
	NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 1 ); )
	_storage[id] = calloc (_num_elements, _size_element);
	//make sure malloc succeeded

	for( size_t j=0; j<_num_elements; j++ ) {
		_initializer( & ((char*)(_storage)[id])[j*_size_element], _original );
	}
	NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 0 ); )
	return _storage[id];
}

inline unsigned TaskReduction::getDepth( void ) const
{
   return _depth;
}

#endif
