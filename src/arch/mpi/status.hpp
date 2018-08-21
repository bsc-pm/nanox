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
