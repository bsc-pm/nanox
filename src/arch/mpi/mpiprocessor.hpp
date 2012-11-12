/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef _NANOS_MPI_PROCESSOR
#define _NANOS_MPI_PROCESSOR

#include "mpi.h"
#include "config.hpp"
#include "mpidevice_decl.hpp"
#include "mpithread.hpp"
#include "cachedaccelerator.hpp"
#include "copydescriptor_decl.hpp"
#include "processingelement.hpp"

namespace nanos {
namespace ext
{
   class MPIProcessor : public CachedAccelerator<MPIDevice>
   {


      private:
         // config variables
         static bool _useUserThreads;
         static size_t _threadsStackSize;
         static System::CachePolicyType _cachePolicy;
         
         // disable copy constructor and assignment operator
         MPIProcessor( const MPIProcessor &pe );
         const MPIProcessor & operator= ( const MPIProcessor &pe );


      public:         
         //MPI Node data
         static size_t _cacheDefaultSize;
         MPI_Comm _communicator;
         int _rank;
         
         //MPIProcessor( int id ) : PE( id, &MPI ) {}
         MPIProcessor( int id , MPI_Comm communicator, int rank ) ;

         virtual ~MPIProcessor() {}

         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd );

         static void prepareConfig ( Config &config );
         // capability query functions
#ifdef MPI_SUPPORTS_ULT
         virtual bool supportsUserLevelThreads () const { return _useUserThreads; }
#else
         virtual bool supportsUserLevelThreads () const { return false; }
#endif
   };

}
}

#endif
