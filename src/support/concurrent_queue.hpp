
#ifndef CONCURRENT_QUEUE_HPP
#define CONCURRENT_QUEUE_HPP

namespace nanos {

namespace detail {

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

		ConcurrentQueueNode( const_reference_type value ) :
			_value( value ), _next( NULL )
		{
		}

		ConcurrentQueueNode( const_reference_type value, ConcurrentQueueNode const* next ) :
			_value( value ), _next( next )
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
			_tail_lock.acquire();
			_tail->setNext(newTail);
			_tail = newTail;
			_tail_lock.release();
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

} // namespace nanos

#endif // CONCURRENT_QUEUE

