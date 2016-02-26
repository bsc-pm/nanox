
#ifndef COMMAND_HPP
#define COMMAND_HPP

#include "memoryaddress.hpp"
#include "mpidevice_decl.hpp"
#include "mpiprocessor_decl.hpp"

namespace nanos {
namespace mpi {
namespace command {

using namespace ext;

class GenericCommand {
	private:
		int _id;   //!< Used to identify which command is being received
		int _code; //!< Additional code number. Its usage is optional.
		MPIProcessor const& _destination; // Only works from the point of view of the sender

		/**
		 * Custom MPI datatype for GenericCommand
		 * This is necessary since we must compute
		 * the displacement of the attributes
		 * with respect to the object's base address
		 */
		static MPI_Datatype _type;

	public:
		GenericCommand( int id, MPIProcessor const& destination ) :
			_id(id), _code(0), _destination( destination )
		{
		}

		GenericCommand( int id, int code, MPIProcessor const& destination ) :
			_id(id), _code(code), _destination( destination )
		{
		}

		virtual ~GenericCommand()
		{
		}

		int getDestinationRank() const
		{
			return _destination.getRank();
		}

		MPI_Comm getCommunicator() const
		{
			return _destination.getCommunicator();
		}

		int getId() const
		{
			return _id;
		}

		int getCode() const
		{
			return _code;
		}

		static void initDataType()
		{
			int blocklen  = 1;
			MPI_Aint disp = offsetof(GenericCommand,_id);

			MPI_Type_create_hindexed(1, &blocklen, &disp, MPI_INT, &_type );
			MPI_Type_commit( &_type );
		}

		static void freeDataType()
		{
			MPI_Type_free( &_type );
		}

		static MPI_Datatype getDataType()
		{
			return _type;
		}

		static GenericCommand* createSpecific( GenericCommand &command );
};

/**
 * Pairs Requestor and Servant types for each operation id
 */
template < int op_id >
struct Command {
	static id = op_id;

   /* 
	 * Base Requestor/Servant should only be used on
	 * partial template specializations of Requestor/Servant
	 */
	typedef RequestorBase< op_id, GenericCommand > RequestorBase;
	typedef ServantBase< op_id, GenericCommand > ServantBase;

	typedef Requestor< op_id, GenericCommand > Requestor;
	typedef Servant  < op_id, GenericCommand > Servant;
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_HPP

