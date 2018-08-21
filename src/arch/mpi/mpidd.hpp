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

#ifndef _NANOS_MPI_WD
#define _NANOS_MPI_WD

#include "mpidevice.hpp"
#include "workdescriptor.hpp"
#include "config.hpp"
#include "mpiprocessor_decl.hpp"

#include <mpi.h>
#include <stdint.h>

namespace nanos {
namespace ext {

   extern MPIDevice MPI;
   
   class MPIDD : public DD
   {
      private:
         MPI_Comm _assignedComm;
         int _assignedRank;
         size_t uid;
         static Atomic<int> uidGen;
         static bool _spawnDone;
         
      public:
       
         // constructors
         MPIDD( work_fct w ) : DD( &MPI, w ), _assignedComm(MPI_COMM_WORLD),_assignedRank( CACHETHREADRANK ) {}

         MPIDD( work_fct w , MPI_Comm assignedComm, int assignedRank) : DD( &MPI, w ), _assignedComm(assignedComm) , _assignedRank(assignedRank) {
             //if (_assignedRank==CACHETHREADRANK) fatal0("Error, rank value (-1) reserved for nanox");
             if (_assignedComm==MPI_COMM_NULL) fatal0("Error, you are trying to use a (MPI_COMM_NULL) commnicator, make sure that your deep_booster_alloc"
                     " call could allocate/had enough hosts to allocate the requested amount of nodes, if you application can work with"
                     " any number of nodes, you can use deep_booster_alloc_nostrict(...,int provided) call which returns in provided the number of hosts which were allocated");
//             if ((uintptr_t)_assignedComm==0) fatal0("Passed NULL/0 communicator in onto clause, make sure you initialize your communicator with deep_booster_alloc before using"
//                     "it on onto clauses");
             if (!_spawnDone) fatal0("Tried to Offload (onto clause/mpi device) code before any remote node allocation was performed."
                     " Please, allocate remote nodes using deep_booster_alloc API call (check OmpSs docs) before trying to offload or check if your call failed");
             uid=uidGen++;
         }

         MPIDD() : DD( &MPI, NULL ), _assignedRank( CACHETHREADRANK ) {}

         // copy constructors
         MPIDD( const MPIDD &dd ) : DD( dd ), _assignedComm(dd._assignedComm) , _assignedRank(dd._assignedRank) {}

         // assignment operator
         const MPIDD & operator= ( const MPIDD &wd );
         // destructor

         virtual ~MPIDD() { }

         MPI_Comm getAssignedComm() const { return _assignedComm; }
         int getAssignedRank() const { return _assignedRank; }
         static void setSpawnDone(bool _spawnDone);
         static bool getSpawnDone();

         virtual void lazyInit (WD &wd, bool isUserLevelThread, WD *previous){}
         virtual size_t size ( void ) { return sizeof(MPIDD); }
         virtual bool isCompatibleWithPE ( const ProcessingElement *pe=NULL );
         virtual MPIDD *copyTo ( void *toAddr );
         

         virtual MPIDD *clone () const { return NEW MPIDD ( *this); }
      };

   inline const MPIDD & MPIDD::operator= ( const MPIDD &dd )
   {
      // self-assignment: ok
      if ( &dd == this ) return *this;

      DD::operator= ( dd );

      return *this;
   }

} // namespace ext
} // namespace nanos

#endif
