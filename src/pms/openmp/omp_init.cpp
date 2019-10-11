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

#include <cstdlib>
#include "system.hpp"
#include "config.hpp"
#include "omp_init.hpp"
#include "omp_wd_data.hpp"
#include "omp_threadteam_data.hpp"
#include "nanos_omp.h"
#include "plugin.hpp"

#ifdef DLB
#include <dlb.h>
#endif

using namespace nanos;
//using namespace nanos::OpenMP;

namespace nanos
{
   namespace PMInterfaceType {
#ifdef NANOX_SS_SUPPORT
      int * ssCompatibility = (int *) 1;
#else
      int * ssCompatibility = (int *) 0;
#endif
   void set_interface_cb( void * );
   void set_interface_cb( void * p  )
   {
      if ( nanos::PMInterfaceType::ssCompatibility != NULL ) {
         sys.setPMInterface(NEW nanos::OpenMP::OmpSsInterface());
      } else {
         sys.setPMInterface(NEW nanos::OpenMP::OpenMPInterface());
      }
   }
   void (*set_interface)( void * ) = set_interface_cb;
   }

   namespace OpenMP {
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

         int requested_workers = sys.getSMPPlugin()->getRequestedWorkers();
         int max_workers = sys.getSMPPlugin()->getNumWorkers();

         if ( requested_workers > 0 && _numThreadsOMP > 0 && requested_workers != _numThreadsOMP ) {
            warning0( "Option --smp-workers value (" << requested_workers << "), and "
                  "OMP_NUM_THREADS value (" << _numThreadsOMP << ") differ. "
                  "The value of OMP_NUM_THREADS will be used.");
         }

         // In OpenMP, OMP_NUM_THREADS takes precedence over --smp-workers
         if ( _numThreadsOMP > 0 ) {
            _numThreads = _numThreadsOMP;
         } else if ( requested_workers > 0 ) {
            _numThreads = requested_workers;
         } else {
            _numThreads = max_workers;
         }

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

      int OpenMPInterface::getMaxThreads() const
      {
         int max_threads = 0;
         if ( myThread ) {
            OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
            max_threads = data->icvs()->getNumThreads();
         } else {
            // This function can be called at initialization when myThread == NULL
            max_threads = globalState->getICVs().getNumThreads();
         }
         return max_threads;
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
       * \param[out] cpu_set CpuSet that will containt the process mask
       */
      const CpuSet& OpenMPInterface::getCpuProcessMask() const
      {
         return sys.getCpuProcessMask();
      }

      /*!
       * \brief Set a new mask for the process
       * \param[in] cpu_set CpuSet that containts the mask to set
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      bool OpenMPInterface::setCpuProcessMask( const CpuSet& cpu_set )
      {
         bool success = sys.setCpuProcessMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );

         return success;
      }

      /*!
       * \brief Add a new mask to be merged with the process mask
       * \param[in] cpu_set CpuSet that containts the mask to add
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      void OpenMPInterface::addCpuProcessMask( const CpuSet& cpu_set )
      {
         sys.addCpuProcessMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Get the current mask of used cpus
       * \param[out] cpu_set CpuSet that will containt the current mask
       */
      const CpuSet& OpenMPInterface::getCpuActiveMask() const
      {
         return sys.getCpuActiveMask();
      }

      /*!
       * \brief Set a new mask of active cpus
       * \param[in] cpu_set CpuSet that containts the mask to set
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      bool OpenMPInterface::setCpuActiveMask( const CpuSet& cpu_set )
      {
         bool success = sys.setCpuActiveMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );

         return success;
      }

      /*!
       * \brief Add a new mask to be merged with active cpus
       * \param[in] cpu_set CpuSet that containts the mask to add
       * \note New threads are not inmediately created nor added to the team in the OpenMP model
       */
      void OpenMPInterface::addCpuActiveMask( const CpuSet& cpu_set )
      {
         sys.addCpuActiveMask( cpu_set );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Enable one CPU in the active cpu mask
       * \param[in] cpuid CPU id to enable
       */
      void OpenMPInterface::enableCpu( int cpuid )
      {
         sys.enableCpu( cpuid );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Disable one CPU in the active cpu mask
       * \param[in] cpuid CPU id to disable
       */
      void OpenMPInterface::disableCpu( int cpuid )
      {
         sys.disableCpu( cpuid );

         OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

#ifdef DLB
      /*!
       * \brief Register Nanos++ interface in DLB
       */
      void OpenMPInterface::registerCallbacks() const
      {
         DLB_CallbackSet( dlb_callback_set_num_threads,
               (dlb_callback_t)nanos_omp_set_num_threads, NULL );
         DLB_CallbackSet( dlb_callback_set_active_mask,
               (dlb_callback_t)nanos_omp_set_active_mask, NULL );
         DLB_CallbackSet( dlb_callback_set_process_mask,
               (dlb_callback_t)nanos_omp_set_process_mask, NULL );
         DLB_CallbackSet( dlb_callback_add_active_mask,
               (dlb_callback_t)nanos_omp_add_active_mask, NULL );
         DLB_CallbackSet( dlb_callback_add_process_mask,
               (dlb_callback_t)nanos_omp_add_process_mask, NULL );
         DLB_CallbackSet( dlb_callback_enable_cpu,
               (dlb_callback_t)nanos_omp_enable_cpu, NULL );
         DLB_CallbackSet( dlb_callback_disable_cpu,
               (dlb_callback_t)nanos_omp_disable_cpu, NULL );
      }
#endif

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
         // Must be allocated through new to avoid problems with the order of
         // initialization of global objects
         globalState = NEW OmpState();
         TaskICVs & icvs = globalState->getICVs();

         int requested_workers = sys.getSMPPlugin()->getRequestedWorkers();
         int max_workers = sys.getSMPPlugin()->getNumWorkers();

         if ( requested_workers > 0 && _numThreadsOMP > 0 && requested_workers != _numThreadsOMP ) {
            warning0( "Option --smp-workers value (" << requested_workers << "), "
                  "and OMP_NUM_THREADS value (" << _numThreadsOMP << ") differ. "
                  "The value of --smp-workers will be used.");
         }

         // In OmpSs, --smp-workers takes precedence over OMP_NUM_THREADS
         if ( requested_workers > 0 ) {
            _numThreads = requested_workers;
         } else if ( _numThreadsOMP > 0 ) {
            _numThreads = _numThreadsOMP;
         } else {
            _numThreads = max_workers;
         }

         icvs.setNumThreads( _numThreads );
         sys.getSMPPlugin()->setRequestedWorkers( _numThreads );

         _description = std::string("OmpSs");
         _malleable = true;
         sys.setInitialMode( System::POOL );
         sys.setUntieMaster( sys.getThreadManagerConf().canUntieMaster() );

         // Loading plugins for OpenMP worksharing policies
         for (int i = omp_sched_static; i <= omp_sched_auto; i++) {
            ws_plugins[i] = sys.getWorkSharing ( ws_names[i] );
            if ( ws_plugins[i] == NULL ){
               if ( !sys.loadPlugin( "worksharing-" + ws_names[i]) ) fatal0( "Could not load " + ws_names[i] + "worksharing" );
               ws_plugins[i] = sys.getWorkSharing ( ws_names[i] );
            }
         }
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
            ensure(parentData,"parent OmpSs data is missing!");

            *data = *parentData;
         } else {
            data->setICVs( &globalState->getICVs() );
         }
         data->setFinal(false);
      }

      int OmpSsInterface::getMaxThreads() const
      {
         int max_threads = 0;
         if ( myThread ) {
            OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
            max_threads = data->icvs()->getNumThreads();
         } else {
            // This function can be called at initialization when myThread == NULL
            max_threads = globalState->getICVs().getNumThreads();
         }
         return max_threads;
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
       * \param[in] cpu_set CpuSet that containts the mask to set
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      bool OmpSsInterface::setCpuProcessMask( const CpuSet& cpu_set )
      {
         LockBlock Lock( _lock );

         bool success = sys.setCpuProcessMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );

         return success;
      }

      /*!
       * \brief Add a new mask to be merged with the process mask
       * \param[in] cpu_set CpuSet that containts the mask to add
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      void OmpSsInterface::addCpuProcessMask( const CpuSet& cpu_set )
      {
         LockBlock Lock( _lock );

         sys.addCpuProcessMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Set a new mask of active cpus
       * \param[in] cpu_set CpuSet that containts the mask to set
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      bool OmpSsInterface::setCpuActiveMask( const CpuSet& cpu_set )
      {
         LockBlock Lock( _lock );

         bool success = sys.setCpuActiveMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );

         return success;
      }

      /*!
       * \brief Add a new mask to be merged with active cpus
       * \param[in] cpu_set CpuSet that containts the mask to add
       * \note New threads are created and/or added to the team ASAP in the OmpSs model
       */
      void OmpSsInterface::addCpuActiveMask( const CpuSet& cpu_set )
      {
         LockBlock Lock( _lock );

         sys.addCpuActiveMask( cpu_set );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Enable one CPU in the active cpu mask
       * \param[in] cpuid CPU id to enable
       */
      void OmpSsInterface::enableCpu( int cpuid )
      {
         LockBlock Lock( _lock );

         sys.enableCpu( cpuid );

         OmpSsData *data = (OmpSsData *) myThread->getCurrentWD()->getInternalData();
         data->icvs()->setNumThreads( sys.getSMPPlugin()->getMaxWorkers() );
      }

      /*!
       * \brief Disable one CPU in the active cpu mask
       * \param[in] cpuid CPU id to disable
       */
      void OmpSsInterface::disableCpu( int cpuid )
      {
         LockBlock Lock( _lock );

         sys.disableCpu( cpuid );

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
