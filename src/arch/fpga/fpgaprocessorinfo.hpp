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

#ifndef _NANOS_FPGA_PROCESSOR_INFO
#define _NANOS_FPGA_PROCESSOR_INFO


#include "fpgaprocessor.hpp"
#include "libxdma.h"

//xdma config definitions, most of these are magic :/
#define TID     0x0             /* Stream identifier 0 */
#define TDEST   0x0             /* Coarse Routing info for stream 0 */
#define TUSER   0x0             /* User defined sideband signaling */
#define ARCACHE 0x3             /* Cache type */
#define ARUSER  0x0             /* Sideband signals */
#define VSIZE   0x1             /* Vsize */
#define STRIDE  0x0             /* Stride control */


namespace nanos {
namespace ext {
      /*!
       * The only purpose of this class is to wrap some device dependent info dependeing
       * on an external library ir order to keep the system as clean as possible
       * (not having to include/define xdma types in system, etc)
       */
      class FPGAProcessor::FPGAProcessorInfo
      {
         private:
            xdma_channel _inChannel, _outChannel;
            xdma_device _deviceHandle;   /// Low level device handle

         public:
            //FPGAProcessorInfo(): _configured( false ) {}
            FPGAProcessorInfo() {}

            //! \brief get handle to the input channel of the device
            xdma_channel getInputChannel() const {
               return _inChannel;
            }
            void setInputChannel( xdma_channel ic ) {
               _inChannel = ic;
            }

            //! \brief get handle to the output channel of the device
            // TODO: Multiple devices (we may need a deviceID or similar)
            xdma_channel getOutputChannel() const {
               return _outChannel;
            }

            inline void setOutputChannel( xdma_channel oc ) {
               _outChannel = oc;
            }

            void setDeviceHandle( xdma_device dev ) {
               _deviceHandle = dev;
            }

            xdma_device getDeviceHandle() const {
               return _deviceHandle;
            }
      };
} // namespace ext
} // namespace nanos

#endif
