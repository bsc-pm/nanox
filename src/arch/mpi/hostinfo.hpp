
#ifndef HOSTINFO_HPP
#define HOSTINFO_HPP

#include "info.hpp"

#include <mpi.h>
#include <string>

namespace nanos {
namespace mpi {

// MPI_Info wrapper class
// Avoids manual allocation/deallocation
// Enable transparent copies easily
// Can return a MPI_Info handle or be casted into it.
class HostInfo : public Info {
private:
	void defaultSettings();

public:
	HostInfo( MPI_Info info ) : Info(info)
	{
	}

	HostInfo( HostInfo const& o ) : Info(o)
	{
	}

	HostInfo() : Info()
	{
		defaultSettings();
	}

	virtual ~HostInfo()
	{
	}
};

} // namespace mpi
} // namespace nanos

#endif //HOSTINFO_HPP
