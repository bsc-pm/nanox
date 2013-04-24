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

#include <sched.h>
#include "system.hpp"
#include <cstdlib>
#include "config.hpp"
#include "omp_init.hpp"
#include "omp_wd_data.hpp"
#include "omp_threadteam_data.hpp"
#include "nanos_omp.h"
#include "plugin.hpp"

extern "C" {
   void DLB_UpdateResources( int max_resources ) __attribute__ ((weak));
}

using namespace nanos;
//using namespace nanos::OpenMP;

namespace nanos
{
   namespace OpenMP {
      int * ssCompatibility __attribute__( ( weak ) );
      OmpState *globalState;

      nanos_ws_t OpenMPInterface::findWorksharing( omp_sched_t kind ) { return ws_plugins[kind]; }

      void OpenMPInterface::config ( Config & cfg )
      {
         cfg.setOptionsSection("OpenMP specific","OpenMP related options");

         // OMP_NUM_THREADS
         cfg.registerAlias("num_threads","omp-threads","Configures the number of OpenMP Threads to use");
         cfg.registerEnvOption("omp-threads","OMP_NUM_THREADS");

         // OMP_SCHEDULE
         // OMP_DYNAMIC
         // OMP_NESTED
         // OMP_STACKSIZE
         // OMP_WAIT_POLICY
         // OMP_MAX_ACTIVE_LEVELS
         // OMP_THREAD_LIMIT

         // Initializing names for OpenMP worksharing policies
         ws_names[omp_sched_static] = std::string("static_for");
         ws_names[omp_sched_dynamic] = std::string("dynamic_for");
         ws_names[omp_sched_guided] = std::string("guided_for");
         ws_names[omp_sched_auto] = std::string("static_for");
      }

      void OpenMPInterface::start ()
      {
         // Must be allocated through new to avoid problems with the order of
         // initialization of global objects
         globalState = NEW OmpState();

         TaskICVs & icvs = globalState->getICVs();
         icvs.setSchedule(LoopSchedule(omp_sched_static));
         icvs.setNumThreads(sys.getNumThreads());

         _description = std::string("OpenMP");
         sys.setInitialMode( System::ONE_THREAD );
         sys.setUntieMaster(false);

         // Loading plugins for OpenMP worksharing policies
         for (int i = omp_sched_static; i <= omp_sched_auto; i++) {
            ws_plugins[i] = sys.getWorkSharing ( ws_names[i] );
            if ( ws_plugins[i] == NULL ){
               if ( !sys.loadPlugin( "worksharing-" + ws_names[i]) ) fatal0( "Could not load " + ws_names[i] + "worksharing" );
               ws_plugins[i] = sys.getWorkSharing ( ws_names[i] );
            }
         }
      }

       void OpenMPInterface::finish()
      {
         delete globalState;
      }

      int OpenMPInterface::getInternalDataSize() const { return sizeof(OpenMPData); }
      int OpenMPInterface::getInternalDataAlignment() const { return __alignof__(OpenMPData); }

      void OpenMPInterface::initInternalData( void * data )
      {
         new (data) OpenMPData();
      }

      void OpenMPInterface::setupWD( WD &wd )
      {
         OpenMPData *data = (OpenMPData *) wd.getInternalData();
         ensure(data,"OpenMP data is missing!");
         WD *parent = wd.getParent();

         if ( parent != NULL ) {
            OpenMPData *parentData = (OpenMPData *) parent->getInternalData();
            ensure(data,"parent OpenMP data is missing!");

            *data = *parentData;
         } else {
            data->setICVs( &globalState->getICVs() );
            data->setFinal(false);
         }
      }

      void OpenMPInterface::wdStarted( WD &wd ) {}
      void OpenMPInterface::wdFinished( WD &wd ) { }

      ThreadTeamData * OpenMPInterface::getThreadTeamData()
      {
         return (ThreadTeamData *) NEW OmpThreadTeamData();
      }

      void OpenMPInterface::setNumThreads ( int nthreads )
      {
         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( nthreads );
      }

      void OpenMPInterface::getCpuMask( cpu_set_t *cpu_set )
      {
         sys.getCpuMask( cpu_set );
      }

      void OpenMPInterface::setCpuMask( const cpu_set_t *cpu_set )
      {
         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( CPU_COUNT(cpu_set) );

         sys.setCpuMask( cpu_set, /* apply */ false );
      }

      void OpenMPInterface::addCpuMask( const cpu_set_t *cpu_set )
      {
         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         int old_nthreads = data->icvs()->getNumThreads();
         data->icvs()->setNumThreads( old_nthreads + CPU_COUNT(cpu_set) );

         sys.addCpuMask( cpu_set, /* apply */ false );
      }

      /* OmpSs Interface */
      void OmpSsInterface::start ()
      {
         // Base class start()
         OpenMPInterface::start();

         // Overwrite custom values
         _description = std::string("OmpSs");
         sys.setInitialMode( System::POOL );
         sys.setUntieMaster(true);

         if ( sys.dlbEnabled() && DLB_UpdateResources ) sys.setUntieMaster(false);
      }

      int OmpSsInterface::getInternalDataSize() const { return sizeof(OmpSsData); }
      int OmpSsInterface::getInternalDataAlignment() const { return __alignof__(OmpSsData); }

      void OmpSsInterface::initInternalData( void * data )
      {
         new (data) OmpSsData();
      }

      void OmpSsInterface::setupWD( WD &wd )
      {
         OmpSsData *data = (OmpSsData *) wd.getInternalData();
         ensure(data,"OmpSs data is missing!");
         WD *parent = wd.getParent();

         if ( parent != NULL ) {
            OmpSsData *parentData = (OmpSsData *) parent->getInternalData();
            ensure(data,"parent OmpSs data is missing!");

            *data = *parentData;
         } else {
            data->setICVs( &globalState->getICVs() );
         }
         data->setFinal(false);
      }

      // update the system threads after the API omp_set_num_threads
      void OmpSsInterface::setNumThreads ( int nthreads )
      {
         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( nthreads );

         sys.updateActiveWorkers( nthreads );

         ensure( sys.getNumThreads() == nthreads, "Update Number of Threads failed " );
      }

      void OmpSsInterface::getCpuMask( cpu_set_t *cpu_set )
      {
         sys.getCpuMask( cpu_set );
      }

      void OmpSsInterface::setCpuMask( const cpu_set_t *cpu_set )
      {
         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( CPU_COUNT(cpu_set) );

         sys.setCpuMask( cpu_set, /* apply */ true );
      }

      void OmpSsInterface::addCpuMask( const cpu_set_t *cpu_set )
      {
         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         int old_nthreads = data->icvs()->getNumThreads();
         data->icvs()->setNumThreads( old_nthreads + CPU_COUNT(cpu_set) );

         sys.addCpuMask( cpu_set, /* apply */ true );
      }
   };
}

/*
   This function must have C linkage to avoid that C applications need to link against the C++ library
*/   
extern "C" {
   void nanos_omp_set_interface( void * )
   {
      if ( nanos::OpenMP::ssCompatibility != NULL ) {
         sys.setPMInterface(NEW nanos::OpenMP::OmpSsInterface());
      } else {
         sys.setPMInterface(NEW nanos::OpenMP::OpenMPInterface());
      }
   }
}
