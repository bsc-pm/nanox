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

#ifndef _NANOS_MPI_WD
#define _NANOS_MPI_WD

#include "mpi.h"
#include <stdint.h>
#include "mpidevice.hpp"
#include "workdescriptor.hpp"
#include "config.hpp"

namespace nanos {
namespace ext
{

   extern MPIDevice MPI;
   
   class MPIDD : public DD
   {

      public:
         typedef void ( *work_fct ) ( void *self );

      private:
         work_fct       _work;
         int _assignedRank;
         MPI_Comm _assignedComm;

      public:
       
         // constructors
          MPIDD( work_fct w ) : DD( &MPI ),_work( w ),_assignedRank( -1 ) {}

         MPIDD( work_fct w , MPI_Comm assignedComm, int assignedRank) : DD( &MPI ),_work( w ), _assignedComm(assignedComm) , _assignedRank(assignedRank) {
             if (_assignedRank<0) fatal0("Tried to setup an mpi device with negative rank");
         }

         MPIDD() : DD( &MPI ),_work( 0 ), _assignedRank( -1 ) {}

         // copy constructors
         MPIDD( const MPIDD &dd ) : DD( dd ), _work( dd._work ) , _assignedRank(dd._assignedRank), _assignedComm(dd._assignedComm) {}

         // assignment operator
         const MPIDD & operator= ( const MPIDD &wd );
         // destructor

         virtual ~MPIDD() { }

         work_fct getWorkFct() const { return _work; }
         MPI_Comm getAssignedComm() const { return _assignedComm; }
         int getAssignedRank() const { return _assignedRank; }
         
         


         virtual void lazyInit (WD &wd, bool isUserLevelThread, WD *previous){}
         virtual size_t size ( void ) { return sizeof(MPIDD); }
         virtual bool isCompatible ( const Device &arch, const ProcessingElement *pe=NULL);
         virtual MPIDD *copyTo ( void *toAddr );
         

         virtual MPIDD *clone () const { return NEW MPIDD ( *this); }
      };

   inline const MPIDD & MPIDD::operator= ( const MPIDD &dd )
   {
      // self-assignment: ok
      if ( &dd == this ) return *this;

      DD::operator= ( dd );

      _work = dd._work;

      return *this;
   }

}
}

#endif
