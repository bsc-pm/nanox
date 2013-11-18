/*************************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                               */
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

#ifndef _NANOX_OMP_INIT
#define _NANOX_OMP_INIT

#include <sched.h>
#include "system.hpp"
#include <cstdlib>
#include "config.hpp"
#include "omp_wd_data.hpp"
#include "omp_threadteam_data.hpp"
#include "nanos_omp.h"

namespace nanos
{
   namespace OpenMP {

      class OpenMPInterface : public PMInterface
      {
         protected:
            std::string ws_names[NANOS_OMP_WS_TSIZE];
            nanos_ws_t  ws_plugins[NANOS_OMP_WS_TSIZE];
            virtual void start () ;

         private:
            virtual void config ( Config & cfg ) ;


            virtual void finish() ;

            virtual int getInternalDataSize() const ;
            virtual int getInternalDataAlignment() const ;
            virtual void initInternalData( void *data ) ;
            virtual void setupWD( WD &wd ) ;

            virtual void wdStarted( WD &wd ) ;
            virtual void wdFinished( WD &wd ) ;

            virtual ThreadTeamData * getThreadTeamData();

            virtual void setNumThreads( int nthreads );
            virtual void getCpuMask( cpu_set_t *cpu_set );
            virtual void setCpuMask( const cpu_set_t *cpu_set );
            virtual void addCpuMask( const cpu_set_t *cpu_set );

         public:
            nanos_ws_t findWorksharing( nanos_omp_sched_t kind ) ;
      };

      class OmpSsInterface : public OpenMPInterface
      {
         private:
            Lock _lock;

            virtual void start () ;
            virtual int getInternalDataSize() const ;
            virtual int getInternalDataAlignment() const ;
            virtual void initInternalData( void *data ) ;
            virtual void setupWD( WD &wd ) ;
            virtual void setNumThreads( int nthreads );
            virtual void setCpuMask( const cpu_set_t *cpu_set );
            virtual void addCpuMask( const cpu_set_t *cpu_set );
      };
   }
}

#endif
