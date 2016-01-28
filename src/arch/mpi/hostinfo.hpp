
#ifndef HOSTINFO_HPP
#define HOSTINFO_HPP

#include <mpi.h>
#include <string>

#ifndef HAVE_CXX11
#   include <sstream>
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
class HostInfo {
	MPI_Info info_;
public:
	HostInfo( MPI_Info info ) : info_(info)
	{
	}

	HostInfo( HostInfo const& o ) : info_() {
		int result = MPI_Info_dup( o.info_, &info_ );
		assert( result == MPI_SUCCESS );
	}

	HostInfo() : info_() {
		int result = MPI_Info_create( &info_ );
		assert( result == MPI_SUCCESS );
	}

	virtual ~HostInfo() {
		int result = MPI_Info_free( &info_ );
		assert( result == MPI_SUCCESS);
	}

	static HostInfo defaultSettings();

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

	void set( const char* key, std::string const& value ) {
		if( !value.empty() )
		MPI_Info_set( info_, key, value.c_str() );
	}

	void set( const char* key, const char* value ) {
		MPI_Info_set( info_, key, value );
	}

	template<typename T>
	void set( const char* key, T const& value ) {
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

#endif //HOSTINFO_HPP
