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

namespace nanos {

inline bool TaskReduction::has( const void *ptr)
{
	return ( ptr == _dependence ) || ( (ptr >= _min) && (ptr < _max) );
}

inline void * TaskReduction::get( size_t id )
{
   return _storage[id].data;
}

inline void TaskReduction::allocate( size_t id )
{
	_storage[id].data = (void *) malloc (_size);
}

inline bool TaskReduction::isInitialized( size_t id )
{
	return _storage[id].isInitialized;
}

inline unsigned TaskReduction::getDepth( void ) const
{
   return _depth;
}

inline void * TaskReduction::finalize( bool deallocate )
{
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 2) );
   void * result = _original;

   //find first private copy that was allocated during execution
   size_t masterId = 0;
   for ( size_t i=0; i<_num_threads; i++) {
      if ( _storage[i].isInitialized ){
         masterId = i;
         break;
      }
   }

   //reduce all to masterId
   for ( size_t i = masterId + 1; i<_num_threads; i++) {
      if ( _storage[i].isInitialized ) {

         if( _isFortranReduction ) {
            _reducer((char*)_storage[masterId].data ,(_storage[i].data));
         } else {
            for( size_t j=0; j<_num_elements; j++ ) {
               _reducer( &((char*)_storage[masterId].data)[j*_size_element] ,& ((char*)(_storage[i].data))[j*_size_element]);
            }
         }
         if( _isLazyPriv && deallocate ) free(_storage[i].data);
         else initialize(i);
      }
   }

   //reduce masterId to global
   if( _storage[masterId].isInitialized ) {
      if( _isFortranReduction ) {
         _reducer_orig_var(_original ,_storage[masterId].data);
      } else {
         for( size_t j=0; j<_num_elements; j++ ){
            _reducer_orig_var( &((char*)_original)[j*_size_element] ,& ((char*)(_storage[masterId].data))[j*_size_element]);
         }
      }

      if( _isLazyPriv && deallocate ) free(_storage[masterId].data);
      else initialize(masterId);
   }

   if( !_isLazyPriv && deallocate ) free(_storage[0].data);

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 0 ) );
   return result;
}

inline  void * TaskReduction::initialize( size_t id )
{
	NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 1 ) );
	if( _isFortranReduction ) {
		_initializer(_storage[id].data, _original );
	} else {
		for( size_t j=0; j < _num_elements; j++ ) {
			_initializer( & ((char*)_storage[id].data)[j*_size_element], _original );
		}
	}

	_storage[id].isInitialized = true;

	NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "reduction" ), 0 ); )
	return _storage[id].data;
}

} // namespace nanos

#endif
