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

#ifndef CONCURRENT_QUEUE_HPP
#define CONCURRENT_QUEUE_HPP

#include "mutex.hpp"
#include "condition_variable.hpp"

namespace nanos {

namespace detail {

/**
 * \file concurrent_queue.hpp
 * Thread safe FIFO queue implementation based on the article by
 * M. Michael and M. Scott
 * "Nonblocking algorithms and preemption-safe locking on
 *  multiprogrammed shared - memory multiprocessors."
 * Journal of Parallel and Distributed Computing, 51(1):1-26, 1998.
 */

template < typename T >
class ConcurrentQueueNode {
	private:
		typedef T        value_type;
		typedef T*       pointer_type;
		typedef const T* const_pointer_type;
		typedef T&       reference_type;
		typedef const T& const_reference_type;

		value_type           _value;
		ConcurrentQueueNode* _next;
		
	public:
		ConcurrentQueueNode() :
			_value(), _next()
		{
		}

		ConcurrentQueueNode( const_reference_type element_value ) :
			_value( element_value ), _next( NULL )
		{
		}

		ConcurrentQueueNode( const_reference_type element_value, ConcurrentQueueNode const* next ) :
			_value( element_value ), _next( next )
		{
		}

		bool hasNext() const
		{
			return _next != NULL;
		}

		const ConcurrentQueueNode* getNext() const
		{
			return _next;
		}

		ConcurrentQueueNode* getNext()
		{
			return _next;
		}

		void setNext( ConcurrentQueueNode* next )
		{
			_next = next;
		}

		const_reference_type value() const
		{
			return _value;
		}
};

} // namespace detail

template < typename T >
class ConcurrentQueue {
	private:
		typedef detail::ConcurrentQueueNode<T> node_type;
		typedef T                              value_type;
		typedef T*                             pointer_type;
		typedef const T*                       const_pointer_type;
		typedef T&                             reference_type;
		typedef const T&                       const_reference_type;

		node_type* _head;
		node_type* _tail;

		Lock       _head_lock;
		Lock       _tail_lock;

	public:
		ConcurrentQueue() :
			_head( new node_type() ), _tail(_head),
			_head_lock(), _tail_lock()
		{
		}

		void push( const_reference_type value )
		{
			node_type* newTail = new node_type(value);
			{
				LockBlock guard(_tail_lock);
				_tail->setNext(newTail);
				_tail = newTail;
			}
		}

		bool pop( reference_type value )
		{
			_head_lock.acquire();

			if( !_head->hasNext() ) {
				_head_lock.release();
				return false;
			} else {
				node_type* newHead = _head->getNext();
				node_type* node = _head;

				_head = newHead;
				value = newHead->value();
				_head_lock.release();

				delete node;
				return true;
			}
		}
};

template < typename T >
class ProducerConsumerQueue {
	private:
		typedef detail::ConcurrentQueueNode<T> node_type;
		typedef T                              value_type;
		typedef T*                             pointer_type;
		typedef const T*                       const_pointer_type;
		typedef T&                             reference_type;
		typedef const T&                       const_reference_type;

		node_type*        _head;
		node_type*        _tail;

		// TODO: consider using Lock (spinlock) for tail (producer) and
		// regular Mutex for head (consumer)
		Mutex             _head_lock;
		Mutex             _tail_lock;
		ConditionVariable _empty_condition;

	public:
		ProducerConsumerQueue() :
			_head( new node_type() ), _tail(_head),
			_head_lock(), _tail_lock()
		{
		}

		void push( const_reference_type value )
		{
			node_type* newTail = new node_type(value);
			LockGuard<Mutex> guard( _tail_lock );
			_tail->setNext(newTail);
			_tail = newTail;
			_empty_condition.notify_one();
		}

		value_type pop()
		{
			UniqueLock<Mutex> guard( _head_lock );

			while( !_head->hasNext() ) {
				_empty_condition.wait(guard);
			}

			node_type* newHead = _head->getNext();
			node_type* node = _head;

			_head = newHead;
			value_type value = newHead->value();

			guard.unlock();

			delete node;
			return value;
		}

		bool empty()
		{
			UniqueLock<Mutex> guard( _head_lock );
			return !_head->hasNext();
		}
};

} // namespace nanos

#endif // CONCURRENT_QUEUE

