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

#ifndef _NANOS_FPGA_MEM_TRANSFER
#define _NANOS_FPGA_MEM_TRANSFER

#include "atomic.hpp"
#include "copydescriptor.hpp"
#include "fpgaprocessor.hpp"
#include "fpgaprocessor.hpp"
#include "deviceops.hpp"

#include "libxdma.h"

namespace nanos {
namespace ext {

      class FPGAMemoryTransfer {
         public:
            CopyDescriptor _copyDescriptor;
            xdma_transfer_handle _dmaHandle;
            size_t _size;  //XXX: size is only needed to really sync transfers

            FPGAMemoryTransfer(const FPGAMemoryTransfer &transfer) :
               _copyDescriptor(transfer._copyDescriptor), _dmaHandle(transfer._dmaHandle) {}
            FPGAMemoryTransfer(CopyDescriptor &cd, xdma_transfer_handle handle) :
                _copyDescriptor(cd), _dmaHandle(handle) {}


      };
      //TODO: Limit number of active transfers
      class FPGAMemoryTransferList {
          public:
              //class FPGAMemoryTransfer;
              std::deque< FPGAMemoryTransfer* > _transfers;
              Lock _lock;

              FPGAMemoryTransferList(): _burst(FPGAConfig::getBurst()), _maxActiveTransfers(FPGAConfig::getMaxTransfers()) {}
              virtual ~FPGAMemoryTransferList() {}

              //! \brief Add a pending transfer to the list
              virtual FPGAMemoryTransfer* addTransfer(CopyDescriptor copyDesc, xdma_transfer_handle handle);
              //We may not need this
              //! \brief Sync a memory transfer and deletes it from the list
              virtual void syncTransfer(CopyDescriptor copyDesc);
              //! \brief Sync a memory transfer and deletes it from the list
              virtual void syncTransfer(uint64_t hostAddress) = 0;
              //! Sync all memory transfers and remove them from the list
              virtual void syncAll();

              inline unsigned int getMaActiveTransfers() const {
                 return _maxActiveTransfers;
              }
              inline void setMaxActiveTransfers(int transfers) {
                 _maxActiveTransfers = transfers;
              }
              virtual void syncNTransfers(unsigned int n) = 0;
          protected:
              //! Number of transfers synchronized when the limit of active transfers is reached
              unsigned int _burst;
              //! Maximum number of dma transfers that can be active in any given moment
              int _maxActiveTransfers;
              //! Sync old transfers in the transfer list to "make room" for new ones
              virtual void syncOldTransfers() = 0;
      };

      /* The main difference in here is that input transfers should not send synchronize to the cache
       * This list does also not lock on internal list operations as no operations will be executed in parallel
       *   as cache flush does not affect input transfers.
       */
      class FPGAMemoryInTransferList : public FPGAMemoryTransferList {
         public:
            FPGAMemoryInTransferList() : FPGAMemoryTransferList() {};
            virtual void syncTransfer(uint64_t hostAddress);
            //virtual void syncAll();
            virtual void syncNTransfers(unsigned int n);
         protected:
            virtual void syncOldTransfers();
      };

      class FPGAMemoryOutTransferList : public FPGAMemoryTransferList {
         public:
            FPGAMemoryOutTransferList (FPGAProcessor &fpgaProcessor) : FPGAMemoryTransferList() , _myProcessor(fpgaProcessor){};
            virtual void syncTransfer(uint64_t hostAddress);
            virtual void syncNTransfers(unsigned int n);
         protected:
            virtual void syncOldTransfers();
            //Need to know the owner of the list to correctly sync transfers
            //(ie. the smp master thread cannot sync an fpga transfer)
            FPGAProcessor &_myProcessor;

      };

} // namespace ext
} // namespace nanos
#endif
