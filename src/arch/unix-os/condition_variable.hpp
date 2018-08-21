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

#ifndef CONDITION_VARIABLE_HPP
#define CONDITION_VARIABLE_HPP

#include "mutex.hpp"

#include <pthread.h>

namespace nanos {

class ConditionVariable {
	private:
		pthread_cond_t _handle;

		ConditionVariable( const ConditionVariable & ); // Non copyable

		ConditionVariable& operator=( const ConditionVariable& ); // Non assignable

	public:
#ifdef HAVE_CXX11
		ConditionVariable() :
			_handle(PTHREAD_COND_INITIALIZER)
		{
		}
#else
		ConditionVariable()
		{
			// Alternative to PTHREAD_COND_INITIALIZER
			pthread_cond_init( &_handle, NULL );
		}
#endif
		~ConditionVariable()
		{
			pthread_cond_destroy( &_handle );
		}

		void notify_one() throw()
		{
			pthread_cond_signal( &_handle );
		}

		void notify_all() throw()
		{
			pthread_cond_broadcast( &_handle );
		}
		
		void wait( UniqueLock<Mutex>& lock )
		{
			pthread_cond_wait( &_handle, lock.mutex()->native_handle() );
		}

		template < class Predicate >
		void wait( UniqueLock<Mutex>& lock, Predicate pred )
		{
			while (!pred()) {
				wait(lock);
			}
		}

		pthread_cond_t* native_handle()
		{
			return &_handle;
		}
};

} // namespace nanos

#endif // CONDITION_VARIABLE_HPP
