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

#ifndef _NANOS_FPGA_PROCESSOR
#define _NANOS_FPGA_PROCESSOR

#include "atomic.hpp"
#include "compatibility.hpp"
#include "copydescriptor_decl.hpp"

#include "fpgadevice.hpp"
#include "fpgaconfig.hpp"
#include "cachedaccelerator.hpp"
#include "fpgapinnedallocator.hpp"

namespace nanos {
namespace ext {

//As in gpu, we could keep track of copied data

      //forward declaration of transfer list
      class FPGAMemoryTransferList;
      class FPGAProcessor: public ProcessingElement
      {
         public:
            class FPGAProcessorInfo;
         private:
            //! FPGA device ID
            Atomic<int> _fpgaDevice;
            static Lock _initLock;  ///Initialization lock (may only be needed if we dynamically spawn fpga helper threads)
            FPGAProcessorInfo *_fpgaProcessorInfo;
            static int _accelSeed;  ///Keeps track of the created accelerators
            int _accelBase;         ///Base of the range of assigned accelerators
            int _numAcc;            ///Number of accelerators managed by this thread
            int _activeAcc;         ///Currently active accelerator. This is a local acc ID/index
            bool _update;           ///Update after submitting data for a task
            int _fallBackAcc;       ///Active accelerator when current task can run anywhere
            SMPProcessor *_core;

            FPGAMemoryTransferList *_inputTransfers;
            FPGAMemoryTransferList *_outputTransfers;

            static FPGAPinnedAllocator _allocator;

         public:

            /*!
             * Constructor:
             * \param id        Processing element ID
             * \param fpgaId    ID of the fpga device
             */
            //FPGAProcessor(int id, int fpgaId, unsigned int uid, memory_space_id_t memSpaceId);
            FPGAProcessor(memory_space_id_t memSpaceId, SMPProcessor *core);
            ~FPGAProcessor();

            FPGAProcessorInfo* getFPGAProcessorInfo() const {
               return _fpgaProcessorInfo;
            }
            FPGAMemoryTransferList *getInTransferList() const {
               return _inputTransfers;
            }
            FPGAMemoryTransferList* getOutTransferList() const {
               return _outputTransfers;
            }

            /*! \brief Initialize hardware:
             *   * Open device
             *   * Get channels
             */
            void init();

            /*! \brief Deinit hardware
             *      Close channels and device
             */
            void cleanUp();

            //Inherted from ProcessingElement
            WD & getWorkerWD () const;
            WD & getMasterWD () const;

            virtual WD & getMultiWorkerWD() const {
               fatal( "getMasterWD(): FPGA processor is not allowed to create MultiThreads" );
            }

            BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread* parent );
            BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, ProcessingElement **repPEs ) {
               fatal( "ClusterNode is not allowed to create FPGA MultiThreads" );
            }

            bool supportsUserLevelThreads () const { return false; }
            //virtual void waitInputs(WorkDescriptor& wd);
            int getAccelBase() const { return _accelBase; }

            void setActiveAcc( int activeAcc ) { _activeAcc = activeAcc; }
            int getActiveAcc() const { return _activeAcc < 0 ? _fallBackAcc : _activeAcc; }

            int getNumAcc() const { return _numAcc; }

            void setNextAccelerator() {
               if ( _update ) {
                  _fallBackAcc = (_fallBackAcc+1)%_numAcc;
                  _update = false;
               }
            }
            void setUpdate(bool update) { _update = update; }

            /// \brief Override (disable) getAddres as this device does not have a dedicated memory nor separated address space
            // This avoids accessing the cache to retrieve a (null) address
            virtual void* getAddress(WorkDescriptor &wd, uint64_t tag, nanos_sharing_t sharing ) {return NULL;}

            BaseThread &startFPGAThread();
            static  FPGAPinnedAllocator& getPinnedAllocator() { return _allocator; }
      };
} // namespace ext
} // namespace nanos

#endif
