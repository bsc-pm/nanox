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

#ifndef MUTEX_HPP
#define MUTEX_HPP

#include <pthread.h>

namespace nanos {

class Mutex {
	private:
		pthread_mutex_t _handle;

		// Non copyable
		Mutex( const Mutex & );

		// Non assignable
		Mutex& operator=( const Mutex & );

	public:
#ifdef HAVE_CXX11
		Mutex() :
			_handle(PTHREAD_MUTEX_INITIALIZER)
		{
		}
#else
		Mutex()
		{
			// Alternative for PTHREAD_MUTEX_INITIALIZER
			pthread_mutex_init( &_handle, NULL );
		}
#endif

		~Mutex()
		{
			pthread_mutex_destroy(&_handle);
		}

		void lock()
		{
			int error __attribute__((unused));
			error = pthread_mutex_lock(&_handle);
			ensure0( error == 0, "Failed to unlock mutex" );
		}

		void unlock()
		{
			int error __attribute__((unused));
			error = pthread_mutex_unlock(&_handle);
			ensure0( error == 0, "Failed to unlock mutex" );
		}

		bool try_lock() throw()
		{
			return !pthread_mutex_trylock(&_handle);
		}

		pthread_mutex_t* native_handle()
		{
			return &_handle;
		}
};

// Empty structures used to disambiguate
// LockGuard construction variants
struct defer_lock_t {};
struct try_to_lock_t {};
struct adopt_lock_t {};

defer_lock_t defer_lock;
try_to_lock_t try_to_lock;
adopt_lock_t adopt_lock;

template < class MutexType >
class LockGuard {
	private:
		MutexType& _mutex;

		LockGuard( const LockGuard & ); // Non copyable

		LockGuard& operator=( const LockGuard & ); // Non assignable

	public:
		LockGuard( MutexType &m ) :
			_mutex(m)
		{
			_mutex.lock();
		}

		LockGuard( MutexType &m, adopt_lock_t t ) :
			_mutex(m)
		{
		}

		~LockGuard()
		{
			_mutex.unlock();
		}
};

template < class MutexType >
class UniqueLock {
	private:
		MutexType& _mutex;
		bool       _isOwner;

		UniqueLock( const UniqueLock & ); // Non copyable

		UniqueLock& operator=( const UniqueLock & ); // Non assignable
	public:
		UniqueLock( MutexType &m ) :
			_mutex(m), _isOwner(false)
		{
			_mutex.lock();
			_isOwner = true;
		}

		UniqueLock( MutexType &m, defer_lock_t t ) :
			_mutex(m), _isOwner(false)
		{
		}

		UniqueLock( MutexType &m, try_to_lock_t t ) :
			_mutex(m), _isOwner(false)
		{
			_isOwner = _mutex.try_lock();
		}

		UniqueLock( MutexType &m, adopt_lock_t t ) :
			_mutex(m), _isOwner(true)
		{
		}

		~UniqueLock()
		{
			if( owns_lock() )
				unlock();
		}

		void lock()
		{
			_mutex.lock();
			_isOwner = true;
		}

		bool try_lock()
		{
			_isOwner = _mutex.try_lock();
			return _isOwner;
		}

		void unlock()
		{
			_isOwner = false;
			_mutex.unlock();
		}

		bool owns_lock() const throw()
		{
			return _isOwner;
		}

		MutexType* mutex() const throw()
		{
			return &_mutex;
		}
};

} // namespace nanos

#endif // MUTEX_HPP

