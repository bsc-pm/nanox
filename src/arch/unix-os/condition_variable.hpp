
#ifndef CONDITION_VARIABLE_HPP
#define CONDITION_VARIABLE_HPP

#include "mutex.hpp"

namespace nanos {

class ConditionVariable {
	private:
		pthread_cond_t _handle;

		ConditionVariable( const ConditionVariable & ); // Non copyable

		ConditionVariable& operator=( const ConditionVariable& ); // Non assignable

	public:
		ConditionVariable() :
			_handle(PTHREAD_COND_INITIALIZER)
		{
		}

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
			while (!preD()) {
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
