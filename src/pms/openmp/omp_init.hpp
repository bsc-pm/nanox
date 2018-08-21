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

#ifndef _NANOX_OMP_INIT
#define _NANOX_OMP_INIT

#include "system.hpp"
#include <cstdlib>
#include "config.hpp"
#include "omp_wd_data.hpp"
#include "omp_threadteam_data.hpp"
#include "nanos_omp.h"
#include "cpuset.hpp"

namespace nanos {

   namespace OpenMP {

      class OpenMPInterface : public PMInterface
      {
         protected:
            std::string ws_names[NANOS_OMP_WS_TSIZE];
            nanos_ws_t  ws_plugins[NANOS_OMP_WS_TSIZE];
            int _numThreads;
            int _numThreadsOMP;
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

            virtual int getMaxThreads() const;
            virtual void setNumThreads( int nthreads );
            virtual const CpuSet& getCpuProcessMask() const;
            virtual bool setCpuProcessMask( const CpuSet& cpu_set );
            virtual void addCpuProcessMask( const CpuSet& cpu_set );
            virtual const CpuSet& getCpuActiveMask() const;
            virtual bool setCpuActiveMask( const CpuSet& cpu_set );
            virtual void addCpuActiveMask( const CpuSet& cpu_set );
            virtual void enableCpu( int cpuid );
            virtual void disableCpu( int cpuid );
#ifdef DLB
            virtual void registerCallbacks() const;
#endif

         public:
            nanos_ws_t findWorksharing( nanos_omp_sched_t kind ) ;

            virtual PMInterface::Interfaces getInterface() const;
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
            virtual int getMaxThreads() const;
            virtual void setNumThreads( int nthreads );
            virtual void setNumThreads_globalState ( int nthreads );
            virtual bool setCpuProcessMask( const CpuSet& cpu_set );
            virtual void addCpuProcessMask( const CpuSet& cpu_set );
            virtual bool setCpuActiveMask( const CpuSet& cpu_set );
            virtual void addCpuActiveMask( const CpuSet& cpu_set );
            virtual void enableCpu( int cpuid );
            virtual void disableCpu( int cpuid );
         public:
            virtual PMInterface::Interfaces getInterface() const;
      };
   }

} // namespace nanos

#endif
