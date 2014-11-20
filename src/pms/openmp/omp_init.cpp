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

using namespace nanos;
//using namespace nanos::OpenMP;

namespace nanos
{
   namespace OpenMP {
      int * ssCompatibility __attribute__( ( weak ) );
      OmpState *globalState;

      nanos_ws_t OpenMPInterface::findWorksharing( nanos_omp_sched_t kind ) { return ws_plugins[kind]; }

      void OpenMPInterface::config ( Config & cfg )
      {
         cfg.setOptionsSection("OpenMP specific","OpenMP related options");

         // OMP_NUM_THREADS
         _numThreads = -1;
         _numThreadsOMP = -1;
         cfg.registerConfigOption( "omp-threads", NEW Config::PositiveVar( _numThreadsOMP ),
                             "Configures the number of OpenMP Threads to use" );
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


      /*** OpenMP Interface ***/

      /*!
       * \brief OpenMP Interface initialization
       */
      void OpenMPInterface::start ()
      {
         // Must be allocated through new to avoid problems with the order of
         // initialization of global objects
         globalState = NEW OmpState();

         TaskICVs & icvs = globalState->getICVs();
         icvs.setSchedule(LoopSchedule(omp_sched_static));

         _numThreads = _numThreadsOMP == -1 ? CPU_COUNT(&sys.getCpuActiveMask()) : _numThreadsOMP;

         icvs.setNumThreads(_numThreads);
         sys.getSMPPlugin()->setRequestedWorkers( _numThreads );

         _description = std::string("OpenMP");
         _malleable = false;
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

      /*!
       * \brief Clean up PM data
       */
      void OpenMPInterface::finish()
      {
         delete globalState;
      }

      /*! \brief Get the size of OpenMPData */
      int OpenMPInterface::getInternalDataSize() const { return sizeof(OpenMPData); }

      /*! \brief Get the aligment of OpenMPData*/
      int OpenMPInterface::getInternalDataAlignment() const { return __alignof__(OpenMPData); }

      /*!
       * \brief Initialize WD internal data allocating a new OpenMPData
       * \param[out] data The pointer where to allocate
       */
      void OpenMPInterface::initInternalData( void * data )
      {
         new (data) OpenMPData();
      }

      /*!
       * \brief Fill WD internal data information from either the WD parent or the Global State
       * \param[out] wd The WD to set up
       */
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

      /*!
       * \brief specific setNumThreads implementation for OpenMP model
       * \param[in] nthreads Number of threads
       */
      void OpenMPInterface::setNumThreads ( int nthreads )
      {
         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( nthreads );
      }

      /*!
       * \brief Get the current mask of the process
       * \param[out] cpu_set cpu_set_t that will containt the process mask
       */
      void OpenMPInterface::getCpuProcessMask( cpu_set_t *cpu_set ) const
      {
         sys.getCpuProcessMask( cpu_set );
      }

      /*!
       * \brief Set a new mask for the process
       * \param[in] cpu_set cpu_set_t that containts the mask to set
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      void OpenMPInterface::setCpuProcessMask( const cpu_set_t *cpu_set )
      {
         sys.setCpuProcessMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Add a new mask to be merged with the process mask
       * \param[in] cpu_set cpu_set_t that containts the mask to add
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      void OpenMPInterface::addCpuProcessMask( const cpu_set_t *cpu_set )
      {
         sys.addCpuProcessMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Get the current mask of used cpus
       * \param[out] cpu_set cpu_set_t that will containt the current mask
       */
      void OpenMPInterface::getCpuActiveMask( cpu_set_t *cpu_set ) const
      {
         sys.getCpuActiveMask( cpu_set );
      }

      /*!
       * \brief Set a new mask of active cpus
       * \param[in] cpu_set cpu_set_t that containts the mask to set
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      void OpenMPInterface::setCpuActiveMask( const cpu_set_t *cpu_set )
      {
         sys.setCpuActiveMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Add a new mask to be merged with active cpus
       * \param[in] cpu_set cpu_set_t that containts the mask to add
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      void OpenMPInterface::addCpuActiveMask( const cpu_set_t *cpu_set )
      {
         sys.addCpuActiveMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Returns the identifier of the interface, OpenMP
       */
      PMInterface::Interfaces OpenMPInterface::getInterface() const
      {
         return PMInterface::OpenMP;
      }


      /*** OmpSs Interface ***/

      /*!
       * \brief OmpSs Interface initialization
       */
      void OmpSsInterface::start ()
      {
         // Base class start()
         OpenMPInterface::start();

         int num_threads = sys.getSMPPlugin()->getRequestedWorkers();
         if ( _numThreadsOMP != -1 ) {
            std::cerr << "Using OMP_NUM_THREADS in an OmpSs applications is discouraged, the recomended way to set the number of worker smp threads is using the flag --smp-workers." << std::endl;
            if ( num_threads == -1 ) {
               std::cerr << "Option --smp-workers not set, will use OMP_NUM_THREADS instead, value: " << _numThreads << "." << std::endl;
               num_threads = _numThreadsOMP;
            } else if ( num_threads != _numThreadsOMP ) {
               std::cerr << "Option --smp-workers set (value: " << num_threads << "), and OMP_NUM_THREADS is also set (value: " << _numThreads << "), will use the value of --smp-workers." << std::endl;
            }

         }
         _numThreads = num_threads;

         TaskICVs & icvs = globalState->getICVs();
         icvs.setNumThreads( _numThreads );
         sys.getSMPPlugin()->setRequestedWorkers( _numThreads );

         // Overwrite custom values
         _description = std::string("OmpSs");
         _malleable = true;
         sys.setInitialMode( System::POOL );
         sys.setUntieMaster(true);
      }

      /*! \brief Get the size of OmpSsData */
      int OmpSsInterface::getInternalDataSize() const { return sizeof(OmpSsData); }

      /*! \brief Get the aligment of OmpSsData*/
      int OmpSsInterface::getInternalDataAlignment() const { return __alignof__(OmpSsData); }

      /*!
       * \brief Initialize WD internal data allocating a new OmpSsData
       * \param[out] data The pointer where to allocate
       */
      void OmpSsInterface::initInternalData( void * data )
      {
         new (data) OmpSsData();
      }

      /*!
       * \brief Fill WD internal data information from either the WD parent or the Global State
       * \param[out] wd The WD to set up
       */
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

      /*!
       * \brief specific setNumThreads implementation for OmpSs model
       * \param[in] nthreads Number of threads
       */
      void OmpSsInterface::setNumThreads ( int nthreads )
      {
         LockBlock Lock( _lock );
         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( nthreads );

         sys.updateActiveWorkers( nthreads );
      }

      void OmpSsInterface::setNumThreads_globalState ( int nthreads )
      {
         TaskICVs & icvs = globalState->getICVs();
         icvs.setNumThreads( nthreads );
      }

      /*!
       * \brief Set a new mask for the process
       * \param[in] cpu_set cpu_set_t that containts the mask to set
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      void OmpSsInterface::setCpuProcessMask( const cpu_set_t *cpu_set )
      {
         LockBlock Lock( _lock );

         sys.setCpuProcessMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Add a new mask to be merged with the process mask
       * \param[in] cpu_set cpu_set_t that containts the mask to add
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      void OmpSsInterface::addCpuProcessMask( const cpu_set_t *cpu_set )
      {
         LockBlock Lock( _lock );

         sys.addCpuProcessMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Set a new mask of active cpus
       * \param[in] cpu_set cpu_set_t that containts the mask to set
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      void OmpSsInterface::setCpuActiveMask( const cpu_set_t *cpu_set )
      {
         LockBlock Lock( _lock );

         sys.setCpuActiveMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Add a new mask to be merged with active cpus
       * \param[in] cpu_set cpu_set_t that containts the mask to add
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      void OmpSsInterface::addCpuActiveMask( const cpu_set_t *cpu_set )
      {
         LockBlock Lock( _lock );

         sys.addCpuActiveMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Returns the identifier of the interface, OpenMP
       */
      PMInterface::Interfaces OmpSsInterface::getInterface() const
      {
         return PMInterface::OmpSs;
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
