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

#include "libxdma.h"

#include "fpgaprocessor.hpp"
#include "fpgaprocessorinfo.hpp"
#include "fpgamemorytransfer.hpp"
#include "fpgadevice.hpp"
#include "fpgaconfig.hpp"
#include "deviceops.hpp"
#include "fpgapinnedallocator.hpp"

using namespace nanos;
using namespace nanos::ext;

#define DIRTY_SYNC

FPGADevice::FPGADevice ( const char *n ): Device( n ) {}

void FPGADevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {

   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ops->addOp();
   ProcessingElement &pe = mem.getPE();
   bool done = copyIn((void*)devAddr, cd, len, &pe);
   if ( done ) ops->completeOp();
}

/*
 * Allow transfers to be performed synchronously 
 */
inline bool FPGADevice::copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe )
{
   verbose("fpga copy in");

   nanos::ext::FPGAProcessor *fpga = ( nanos::ext::FPGAProcessor* ) pe;

   uint64_t  src_addr = remoteSrc.getTag();
   int status;
   xdma_channel iChan;
   xdma_transfer_handle dmaHandle;
   xdma_device device;

   iChan =  fpga->getFPGAProcessorInfo()[fpga->getActiveAcc()].getInputChannel();
   device = fpga->getFPGAProcessorInfo()[fpga->getActiveAcc()].getDeviceHandle();

   syncTransfer(src_addr, fpga);

   debug("submitting input transfer:" << std::endl
           << "  @:" << std::hex << src_addr << std::dec << " size:" << size
           << "  iChan:" << iChan );

   FPGAPinnedAllocator& allocator = FPGAProcessor::getPinnedAllocator();
   uint64_t baseAddress = (uint64_t)allocator.getBasePointer( (void *)src_addr, size);
   ensure( baseAddress != 0, "Trying to submit a regular (not pinned) buffer to FPGA");
   uint64_t offset;
   xdma_buf_handle bufHandle;
   offset = src_addr - baseAddress;
   bufHandle = allocator.getBufferHandle( (void *)src_addr );

   NANOS_FPGA_CREATE_RUNTIME_EVENT( ext::NANOS_FPGA_SUBMIT_IN_DMA_EVENT );
   //Support synchronous transfers??
   status = xdmaSubmitKBuffer( bufHandle, size, (unsigned int)offset, XDMA_ASYNC,
           device, iChan, &dmaHandle );
   NANOS_FPGA_CLOSE_RUNTIME_EVENT;

   debug ( "  got intput handle: " << dmaHandle );
   if ( status )
      warning("Error submitting output: " << status);

   if ( FPGAConfig::getSyncTransfersEnabled() ) {
      {  NANOS_INSTRUMENT( InstrumentBurst i( "in-xdma" ,ext::NANOS_FPGA_WAIT_INPUT_DMA_EVENT); )
         xdmaWaitTransfer( dmaHandle );
         xdmaReleaseTransfer( &dmaHandle );
      }
   } else {
      fpga->getInTransferList()->addTransfer( remoteSrc, dmaHandle );
   }

   /*
    * This is not actually true because the data is copied asynchronously.
    * As long as the transfer is submitted and 
    */
   return true; // true means sync transfer
}
void FPGADevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len,
      SeparateMemoryAddressSpace &mem, DeviceOps *ops,
      WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId ) {

   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ops->addOp();
   ProcessingElement &pe = mem.getPE();
   bool done = copyOut(cd, (void*)devAddr, len, &pe);
   if ( done ) ops->completeOp();
}

bool FPGADevice::copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe )
{
   verbose("fpga copy out");

   nanos::ext::FPGAProcessor *fpga = ( nanos::ext::FPGAProcessor* ) pe;
   
   //get channel
   xdma_channel oChan;
   xdma_transfer_handle dmaHandle;
   xdma_device device;
   int status;
   uint64_t src_addr = remoteDst.getTag();
   device = fpga->getFPGAProcessorInfo()[fpga->getActiveAcc()].getDeviceHandle();
   oChan = fpga->getFPGAProcessorInfo()[fpga->getActiveAcc()].getOutputChannel();

   debug("submitting output transfer:" << std::endl
           << "  @:" << std::hex <<  src_addr << std::dec << " size:" << size
           << "  oChan:" << oChan );

   //get pinned buffer handle for this address
   //at this point, assume that all buffers to be transferred to fpga are pinned

   FPGAPinnedAllocator& allocator = FPGAProcessor::getPinnedAllocator();
   uint64_t baseAddress = (uint64_t)allocator.getBasePointer( (void*)src_addr, size );
   ensure( baseAddress != 0, "Trying to submit a regular (not pinned) buffer to FPGA");
   uint64_t offset;
   xdma_buf_handle bufHandle;
   offset = src_addr - baseAddress;
   bufHandle = allocator.getBufferHandle( (void *)src_addr );

   NANOS_FPGA_CREATE_RUNTIME_EVENT( ext::NANOS_FPGA_SUBMIT_OUT_DMA_EVENT );
   status = xdmaSubmitKBuffer( bufHandle, size, (unsigned int)offset, XDMA_ASYNC,
           device, oChan, &dmaHandle );
   NANOS_FPGA_CLOSE_RUNTIME_EVENT;

   if ( status )
      warning( "Error submitting output:" << status );

   debug( "  got out handle: " << dmaHandle );
   verbose("add transfer H:" << dmaHandle << " to the transfer list");

   bool finished;

   if ( FPGAConfig::getSyncTransfersEnabled() ) {
      {  NANOS_INSTRUMENT( InstrumentBurst i( "in-xdma" ,ext::NANOS_FPGA_WAIT_OUTPUT_DMA_EVENT); )
         xdmaWaitTransfer( dmaHandle );
         xdmaReleaseTransfer( &dmaHandle );
      }
      finished = true;
   } else {
      ((FPGAMemoryOutTransferList*)fpga->getOutTransferList())->addTransfer( remoteDst, dmaHandle );
      finished = false;
   }

   return finished;
}

void *FPGADevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem,
        WorkDescriptor const *wd, unsigned int copyIdx){
   //empty as we cannot allocate memory inside the fpga
   return (void *) 0xdeadbeef;
}
void FPGADevice::memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ){
   //empty as we cannot allocate memory inside the fpga
}

//this is used to priorize transfers (because someone needs the data)
//In our case this causes this actually means "finish the transfer"
void FPGADevice::syncTransfer( uint64_t hostAddress, ProcessingElement *pe)
{
    //TODO: At this point we only are going to sync output transfers
    // as input transfers do not need to be synchronized
    ((FPGAProcessor *)pe)->getOutTransferList()->syncTransfer(hostAddress);
    //((FPGAProcessor *)pe)->getInTransferList()->syncTransfer(hostAddress);
}
 bool FPGADevice::copyDevToDev( void * addrDst, CopyDescriptor& dstCd, void * addrSrc, std::size_t size,
         ProcessingElement *peDst, ProcessingElement *peSrc )
{
   //sync transfer on the origin
   //This will cause the transfer to be synchronized
   syncTransfer(dstCd.getTag(), peSrc);
   //submit input copy to dest accelerator
   return copyIn(addrDst, dstCd, size, peDst );

}
