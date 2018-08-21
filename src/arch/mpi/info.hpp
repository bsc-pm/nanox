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

#ifndef INFO_HPP
#define INFO_HPP

#include <mpi.h>
#include <string>

#ifndef HAVE_CXX11
#   include <sstream>
#endif

#if MPI_VERSION >=3
#define NANOS_MPI2_CONST const
#else
#define NANOS_MPI2_CONST
#endif

// TODO: replace by debug.hpp
// TODO: replace assert for ensure
#include <cassert>

namespace nanos {
namespace mpi {

// MPI_Info wrapper class
// Avoids manual allocation/deallocation
// Enable transparent copies easily
// Can return a MPI_Info handle or be casted into it.
class Info {
	MPI_Info info_;

public:
	Info() : info_() {
		int result = MPI_Info_create( &info_ );
		assert( result == MPI_SUCCESS );
	}

	Info( MPI_Info info ) : info_(info)
	{
	}

	Info( Info const& o ) : info_() {
		int result = MPI_Info_dup( o.info_, &info_ );
		assert( result == MPI_SUCCESS );
	}

	virtual ~Info() {
		int result = MPI_Info_free( &info_ );
		assert( result == MPI_SUCCESS);
	}

	static Info& env()
	{
		static Info envInfo( MPI_INFO_ENV );
		return envInfo;
	}

	// Cast operator. Should be const to allow 
	// implicit vector copies to be performed 
	// without errors.
	operator MPI_Info() const {
		return info_;
	}

	MPI_Info const& get() const {
		return info_;
	}

	MPI_Info& get() {
		return info_;
	}

	bool containsKey( const char* key ) {
		int flag;
		std::string value;
		MPI_Info_get( info_, key, 0, &value[0], &flag );

		return flag == 1;
	}

	std::string get( const char* key ) {
		int flag;
		std::string value( "\0", MPI_MAX_INFO_VAL );
		MPI_Info_get( info_, key, value.size(), &value[0], &flag );
		if( flag ) {
#ifdef HAVE_CXX11
			value.shrink_to_fit();
#endif
			return value;
		} else {
			return std::string();
		}
	}

	void set( NANOS_MPI2_CONST char* key, NANOS_MPI2_CONST std::string& value ) {
		if( !value.empty() )
		MPI_Info_set( info_, key, value.c_str() );
	}

	void set( NANOS_MPI2_CONST char* key, NANOS_MPI2_CONST char* value ) {
		MPI_Info_set( info_, key, value );
	}

	template<typename T>
	void set( NANOS_MPI2_CONST char* key, const T& value ) {
#ifdef HAVE_CXX11
		std::string svalue( std::to_string(value) );
#else
		std::stringstream ss;
		ss << value;
		std::string svalue( ss.str() );
#endif
		set( key, svalue );
	}
};

} // namespace mpi
} // namespace nanos

#endif //INFO_HPP

