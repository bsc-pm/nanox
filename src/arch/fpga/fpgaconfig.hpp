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

#ifndef _NANOS_FPGA_CFG
#define _NANOS_FPGA_CFG
#include "config.hpp"

#include "system_decl.hpp"

namespace nanos {
namespace ext {

      class FPGAConfig
      {
            friend class FPGAPlugin;
         private:
            //! Defines the cache policy used by FPGA devices
            //! Data transfer's mode (synchronous / asynchronous, ...)
            //Basically determines where the waits will be placed for now we will only support sync
            static int                       _numAccelerators;
            static bool                      _disableFPGA;
            static int                       _numFPGAThreads;
            static const int                 _maxAccelerators;

            static unsigned int              _burst;
            static int                       _maxTransfers;
            static Atomic<int>               _accelID; ///ID assigned to each individual accelerator

            /*! Parses the GPU user options */
            static void prepare ( Config &config );
            /*! Applies the configuration options and retrieves the information of the GPUs of the system */
            static void apply ( void );
            static Lock                      _dmaLock;
            static int                       _idleSyncBurst;
            static bool                      _syncTransfers;

         public:
            static void printConfiguration( void );
            static int getFPGACount ( void ) { return _numAccelerators; }
            static inline Lock& getDMALock() { return _dmaLock; }
            static void acquireDMALock();
            static void releaseDMALock();
            static inline unsigned int getBurst() { return _burst; }
            static inline unsigned int getMaxTransfers() { return _maxTransfers; }
            static inline int getAccPerThread() { return _numAccelerators/_numFPGAThreads; }
            static inline int getNumFPGAThreads() { return _numFPGAThreads; }
            static int getAcceleratorID() {
               int t = _accelID.value();
               _accelID++;
               return t;
            }
            static inline bool isDisabled() {return _disableFPGA; }
            //should be areSyncTransfersDisabled() but is for consistency with other bool getters
            static inline int getIdleSyncBurst() { return _idleSyncBurst; }
            static bool getSyncTransfersEnabled() { return _syncTransfers; }

      };
       //create instrumentation macros (as gpu) to make code cleaner
#define NANOS_FPGA_CREATE_RUNTIME_EVENT(x)    NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseOpenBurstEvent (    \
         sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-xdma" ), (x) ); )

#define NANOS_FPGA_CLOSE_RUNTIME_EVENT       NANOS_INSTRUMENT( \
      sys.getInstrumentation()->raiseCloseBurstEvent (   \
         sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-xdma" ), 0 ); )

      typedef enum {
         NANOS_FPGA_NULL_EVENT = 0,      // 0
         NANOS_FPGA_OPEN_EVENT,
         NANOS_FPGA_CLOSE_EVENT,
         NANOS_FPGA_REQ_CHANNEL_EVENT,
         NANOS_FPGA_REL_CHANNEL_EVENT,
         NANOS_FPGA_SUBMIT_IN_DMA_EVENT,    // 5
         NANOS_FPGA_SUBMIT_OUT_DMA_EVENT,
         NANOS_FPGA_WAIT_INPUT_DMA_EVENT,
         NANOS_FPGA_WAIT_OUTPUT_DMA_EVENT
      }in_xdma_event_value;

} // namespace ext
} // namespace nanos
#endif
