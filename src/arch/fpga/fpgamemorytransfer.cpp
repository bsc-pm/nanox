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

#include "fpgaconfig.hpp"
#include "fpgamemorytransfer.hpp"
#include "instrumentationmodule_decl.hpp"

using namespace nanos;
using namespace nanos::ext;

FPGAMemoryTransfer* FPGAMemoryTransferList::addTransfer(CopyDescriptor copyDesc, xdma_transfer_handle handle) {
   debug("Add transfer" << copyDesc.getTag());

   FPGAMemoryTransfer *newTransfer = NEW FPGAMemoryTransfer(copyDesc, handle);
   if ( _transfers.size() > (unsigned int)_maxActiveTransfers && _maxActiveTransfers >= 0 ) {
      syncOldTransfers();
   }
   _lock.acquire();
   _transfers.push_back(newTransfer);
   _lock.release();
   return newTransfer;
}

void FPGAMemoryTransferList::syncAll(){
   syncNTransfers( _transfers.size() );
}

void FPGAMemoryTransferList::syncTransfer(CopyDescriptor copyDesc) {
   syncTransfer( copyDesc.getTag() );
}

void FPGAMemoryOutTransferList::syncTransfer(uint64_t hostAddress){
   debug("sync out transfer" << (void*)hostAddress);
   _lock.acquire();
   for (std::deque< FPGAMemoryTransfer* >::iterator it = _transfers.begin();
         it != _transfers.end();
         it++)
   {
      int status;
      if ( (*it)->_copyDescriptor.getTag() == hostAddress ){
         FPGAMemoryTransfer *transfer = *it;
         _transfers.erase(it);
         _lock.release();
         verbose("DMAWait out" << transfer->_dmaHandle);
         {  NANOS_INSTRUMENT( InstrumentBurst i( "in-xdma" ,ext::NANOS_FPGA_WAIT_OUTPUT_DMA_EVENT); )

            //TODO use non blocking finish in orer to clean finished transfers
            status = xdmaWaitTransfer( transfer->_dmaHandle );
            verbose(" waited " << transfer->_dmaHandle);
            if (status) {
               warning( "ERROR on dma out transfer wait #" << transfer->_dmaHandle <<  "status:" << status );
            }
            xdmaReleaseTransfer( &transfer->_dmaHandle );

         }
         transfer->_copyDescriptor._ops->completeOp();

         delete transfer;
         return;
      }
   }
   _lock.release();

}

void FPGAMemoryOutTransferList::syncNTransfers(unsigned int n){
   int status;
   int nx = std::min(n, (unsigned int)_transfers.size());
   if (nx == 0) return; //avoid locking when there is nothing to sync
   debug( "Sync " << n << " out transfers");
   _lock.acquire();
   for (int i=0; i<nx; i++)
   {
      FPGAMemoryTransfer *transfer  = _transfers.front();
      _transfers.pop_front();
      //_lock.release();

      //perform this outside the critical region so we do not lock on long operations
      verbose("DMAWait out" << transfer->_dmaHandle);
      NANOS_FPGA_CREATE_RUNTIME_EVENT( NANOS_FPGA_WAIT_OUTPUT_DMA_EVENT );
      status = xdmaWaitTransfer( transfer->_dmaHandle );
      if (status) {
         warning( "ERROR on dma out transfer wait #" << transfer->_dmaHandle <<  "status:" << status );
      }
      xdmaReleaseTransfer( &transfer->_dmaHandle );

      NANOS_FPGA_CLOSE_RUNTIME_EVENT;


      transfer->_copyDescriptor._ops->completeOp();

      delete transfer;

      //_lock.acquire();//must lock here so we correctly get the next element
   }
   _lock.release();
}

//sync up to burst transfers
void FPGAMemoryOutTransferList::syncOldTransfers(){
   syncNTransfers(_burst);
}

void FPGAMemoryInTransferList::syncTransfer(uint64_t hostAddress){
   debug( "Sync intput transfer " << (void*)hostAddress);
   int status;
   for (std::deque< FPGAMemoryTransfer* >::iterator it = _transfers.begin();
         it != _transfers.end();
         it++)
   {
      if ( (*it)->_copyDescriptor.getTag() == hostAddress ){
         FPGAMemoryTransfer *transfer = *it;
         _transfers.erase(it);
         verbose("DMAWait in" << transfer->_dmaHandle);
         {  NANOS_INSTRUMENT( InstrumentBurst i( "in-xdma" ,ext::NANOS_FPGA_WAIT_INPUT_DMA_EVENT); )
            status = xdmaWaitTransfer( transfer->_dmaHandle );
            xdmaReleaseTransfer( &transfer->_dmaHandle );
         }
         if (status) {
            warning( "ERROR on dma in transfer wait #" << transfer->_dmaHandle <<  "status:" << status );
         }

         return;
      }
   }

}

void FPGAMemoryInTransferList::syncNTransfers(unsigned int n){
   debug( "Sync " << n << " in transfers");
   int status;
   //_lock.acquire();
   int nx = std::min(n, (unsigned int)_transfers.size());
   for (int i=0; i<nx; i++)
   {
      FPGAMemoryTransfer *transfer  = _transfers.front();

      verbose("DMAWait in" << transfer->_dmaHandle);
      NANOS_FPGA_CREATE_RUNTIME_EVENT( NANOS_FPGA_WAIT_INPUT_DMA_EVENT );
      status = xdmaWaitTransfer( transfer->_dmaHandle );
      NANOS_FPGA_CLOSE_RUNTIME_EVENT;
      xdmaReleaseTransfer( &transfer->_dmaHandle );
      if (status) {
         warning( "ERROR on dma in transfer wait #" << transfer->_dmaHandle <<  "status:" << status );
      }
      _transfers.pop_front();
   }
   //_lock.release();
}

void FPGAMemoryInTransferList::syncOldTransfers(){
   syncNTransfers(_burst);
}

