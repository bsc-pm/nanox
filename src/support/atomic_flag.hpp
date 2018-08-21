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

#ifndef _NANOS_ATOMIC_FLAG
#define _NANOS_ATOMIC_FLAG

namespace nanos {

/**
 * \class atomic_flag is an atomic boolean type.
 * Unlike all specializations of Atomic, it is guaranteed to be lock-free.
 * Unlike Atomic<bool>, atomic_flag does not provide load or store operations.
 * It is available in libstdc++ since C++11 standard.
 * See http://en.cppreference.com/w/cpp/atomic/atomic_flag
 */
class atomic_flag {
	private:
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
		bool          _value;
#else
		volatile bool _value;
#endif

	public:
		/**
		 * Default constructor for atomic_flag
		 */
		atomic_flag() : _value(false) { clear(); }

		/**
		 * Destructor for class atomic_flag
		 */
		~atomic_flag() {}

	private:
		/**
		 * Atomic flag is not copyable
		 */
		atomic_flag( const atomic_flag& );

		/**
		 * Atomic flag is not copyable
		 */
		atomic_flag& operator=( const atomic_flag& );

		/**
		 * Atomic flag is not copyable
		 */
		atomic_flag& operator=( const atomic_flag& ) volatile;

	public:
		bool test_and_set();

		bool load();

		void clear();
};

inline bool atomic_flag::test_and_set()
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
	return __atomic_test_and_set(&_value,__ATOMIC_ACQ_REL);
#else
	return __sync_lock_test_and_set(&_value, true);
#endif
}

inline bool atomic_flag::load()
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
	return __sync_load_n(&_value,__ATOMIC_ACQUIRE);
#else
	return _value;
#endif
}

inline void atomic_flag::clear()
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
	__atomic_clear(&_value,__ATOMIC_RELEASE);
#else
	__sync_lock_release(&_value);
#endif
}

} // namespace nanos

#endif // _NANOS_ATOMIC_FLAG

