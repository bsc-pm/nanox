
#ifndef STATUS_H
#define STATUS_H

#include <cassert>
#include <mpi.h>

#include <algorithm>

namespace nanos {
namespace mpi {

enum StatusKind
{
	ignore = 1,
	attend = 0
};

template < StatusKind kind >
class status;

template <>
class status<attend>
{
	public:
		typedef MPI_Status value_type;
		typedef MPI_Status base_type;

	private:
		value_type _value;
	
	public:
		status() :
			_value()
		{
		}
	
		status( value_type const& value ) :
			_value(value)
		{
		}
	
		status( status const& other ) :
			_value( other._value )
		{
		}

		status& operator=( status const& other )
		{
			_value = other._value;
			return *this;
		}

		status& operator=( value_type const& other )
		{
			_value = other;
			return *this;
		}

		void copy( value_type *other ) const
		{
				*other = _value;
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

template <>
class status<ignore>
{
	public:
		typedef MPI_Status value_type;
		typedef MPI_Status base_type;

		operator value_type* ()
		{
			return MPI_STATUS_IGNORE;
		}
};

} // namespace mpi
} // namespace nanos

#endif // STATUS_H
