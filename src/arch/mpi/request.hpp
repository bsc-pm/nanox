
#ifndef REQUEST_H
#define REQUEST_H

#include <mpi.h>

#include "status.hpp"

namespace nanos {
namespace mpi {

class request;

inline bool test_impl( request &req );

template < StatusKind kind >
inline bool test_impl( request &req, status<kind> &st );

class request
{
	private:
		MPI_Request _value;

	public:
		typedef MPI_Request value_type;
	
		request() :
			_value( MPI_REQUEST_NULL )
		{
		}
	
		request( request &o ) :
			_value( o._value )
		{
		}

		request( MPI_Request value ) :
			_value( value )
		{
		}

		virtual ~request()
		{
		}
	
		request& operator=( value_type other )
		{
			_value = other;
			return *this;
		}
	
		/**
		 * Transfers MPI_Request ownership
		 */
		request& operator=( request &other )
		{
			_value = other._value;
			other._value = MPI_REQUEST_NULL;
			return *this;
		}

		bool test()
		{
			return test_impl( *this );
		}
	
		template < StatusKind kind >
		bool test( status<kind> &st )
		{
			return test_impl( *this, st );
		}

		void free()
		{
			if( _value != MPI_REQUEST_NULL )
				MPI_Request_free( &_value );
		}
	
		value_type* data()
		{
			return &_value;
		}
	
		operator value_type () const
		{
			return _value;
		}

		operator value_type& ()
		{
			return _value;
		}

		operator value_type* ()
		{
			return &_value;
		}
};

class scoped_request : public request
{
	public:
		scoped_request() :
			request()
		{
		}

		scoped_request( request const& o ) :
			request( o )
		{
		}

		virtual ~scoped_request()
		{
			free();
		}

	private:
		/** Prior to C++11, deleting
		 * a copy constructor is only
		 * possible with privatization
		 */
		scoped_request( scoped_request const& o ) :
			request( o )
		{
		}
};

inline bool test_impl( request &req )
{
	int flag;
	MPI_Test(
				req.data(),
				&flag,
				MPI_STATUS_IGNORE
			);
	return flag;
}

template < StatusKind kind >
inline bool test_impl( request &req, status<kind> &st )
{
	int flag;
	MPI_Test(
				req.data(),
				&flag,
				static_cast<typename status<kind>::value_type*>(st)
			);
	return flag;
}

template < typename Iterator >
inline void wait_all( Iterator begin, Iterator end )
{
	std::vector<MPI_Request> requests( begin, end );
	MPI_Waitall( requests.size(), &requests[0], MPI_STATUSES_IGNORE );
}

template < typename Iterator >
inline bool test_all( Iterator begin, Iterator end )
{
	int flag;
	std::vector<MPI_Request> requests( begin, end );
	MPI_Testall( requests.size(), &requests[0], &flag, MPI_STATUSES_IGNORE );
	return flag == 1;
}

} // namespace mpi
} // namespace nanos

#endif // REQUEST_H

