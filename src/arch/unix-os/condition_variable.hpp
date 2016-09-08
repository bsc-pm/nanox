
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
