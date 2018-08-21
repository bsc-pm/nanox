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

#include "fpgaprocessor.hpp"
#include "fpgadd.hpp"
#include "fpgathread.hpp"
#include "fpgaconfig.hpp"
#include "fpgaworker.hpp"
#include "fpgaprocessorinfo.hpp"
#include "fpgamemorytransfer.hpp"
#include "instrumentationmodule_decl.hpp"
#include "smpprocessor.hpp"
#include "fpgapinnedallocator.hpp"

#include "libxdma.h"

using namespace nanos;
using namespace nanos::ext;

int FPGAProcessor::_accelSeed = 0;
Lock FPGAProcessor::_initLock;
FPGAPinnedAllocator FPGAProcessor::_allocator;
/*
 * TODO: Support the case where each thread may manage a different number of accelerators
 */
FPGAProcessor::FPGAProcessor( memory_space_id_t memSpaceId, SMPProcessor *core ) :
   ProcessingElement( &FPGA, memSpaceId, 0, 0, false, 0, false ),
   _fpgaDevice(0), _numAcc(FPGAConfig::getAccPerThread()), _update(true),
   _fallBackAcc(0), _core( core )
{
   //initilalize to _numAcc so if setActiveAcc() kicks in, first accelerator is #0
   _activeAcc = _numAcc-1;

   _initLock.acquire();
   _accelBase = _accelSeed;
   _accelSeed += _numAcc;
   _initLock.release();

   _fpgaProcessorInfo = NEW FPGAProcessorInfo[_numAcc];
   _inputTransfers = NEW FPGAMemoryInTransferList();
   _outputTransfers = NEW FPGAMemoryOutTransferList(*this);
}

FPGAProcessor::~FPGAProcessor(){
    delete[] _fpgaProcessorInfo;
    delete _outputTransfers;
    delete _inputTransfers;
}

void FPGAProcessor::init()
{
   xdma_device *devices = NEW xdma_device[_numAcc];
   xdma_status status;
   status = xdmaGetDevices(FPGAConfig::getFPGACount(), devices, NULL);

   for ( int i=0; i < _numAcc; i++ ) {
      xdma_channel iChan, oChan;
      int devIndex = i + _accelBase;

      _fpgaProcessorInfo[i].setDeviceHandle( devices[devIndex] );

      //open input channel
      NANOS_FPGA_CREATE_RUNTIME_EVENT( ext::NANOS_FPGA_REQ_CHANNEL_EVENT);
      status = xdmaOpenChannel(devices[devIndex], XDMA_TO_DEVICE, XDMA_CH_NONE, &iChan);
      NANOS_FPGA_CLOSE_RUNTIME_EVENT;

      if (status)
         warning("Error opening DMA input channel: " << status);

      debug("got input channel " << iChan );

      _fpgaProcessorInfo[i].setInputChannel( iChan );

      NANOS_FPGA_CREATE_RUNTIME_EVENT( ext::NANOS_FPGA_REQ_CHANNEL_EVENT );
      status = xdmaOpenChannel(devices[devIndex], XDMA_FROM_DEVICE, XDMA_CH_NONE, &oChan);
      NANOS_FPGA_CLOSE_RUNTIME_EVENT;
      if (status || !oChan)
         warning ("Error opening DMA output channel");

      debug("got output channel " << oChan );

      _fpgaProcessorInfo[i].setOutputChannel( oChan );
   }
   delete devices;
}

void FPGAProcessor::cleanUp()
{

   //release channels
   for (int i=0; i<_numAcc; i++) {

      xdma_status status;
      xdma_channel tmpChannel;

      //wait for remaining transfers that could remain
      _inputTransfers->syncAll();
      _outputTransfers->syncAll();

      debug("Release DMA channels");
      NANOS_FPGA_CREATE_RUNTIME_EVENT( ext::NANOS_FPGA_REL_CHANNEL_EVENT );
      tmpChannel = _fpgaProcessorInfo[i].getInputChannel();
      debug("release input channel " << _fpgaProcessorInfo[i].getInputChannel() );
      status = xdmaCloseChannel( &tmpChannel );
      debug("  channel released");
      //Update the new channel as it may be modified by closing the channel
      _fpgaProcessorInfo[i].setInputChannel( tmpChannel );
      NANOS_FPGA_CLOSE_RUNTIME_EVENT;

      if ( status )
         warning ( "Failed to release input dma channel" );

      NANOS_FPGA_CREATE_RUNTIME_EVENT( ext::NANOS_FPGA_REL_CHANNEL_EVENT );
      tmpChannel = _fpgaProcessorInfo[i].getOutputChannel();
      debug("release output channel " << _fpgaProcessorInfo[i].getOutputChannel() );
      status = xdmaCloseChannel( &tmpChannel );
      debug("  channel released");
      _fpgaProcessorInfo[i].setOutputChannel( tmpChannel );
      NANOS_FPGA_CLOSE_RUNTIME_EVENT;

      if ( status )
         warning ( "Failed to release output dma channel" );
   }
}



WorkDescriptor & FPGAProcessor::getWorkerWD () const
{
   //SMPDD *dd = NEW SMPDD( ( SMPDD::work_fct )Scheduler::workerLoop );
   SMPDD *dd = NEW SMPDD( ( SMPDD::work_fct )FPGAWorker::FPGAWorkerLoop );
   WD *wd = NEW WD( dd );
   return *wd;
}

WD & FPGAProcessor::getMasterWD () const
{
   fatal("Attempting to create a FPGA master thread");
}

BaseThread & FPGAProcessor::createThread ( WorkDescriptor &helper, SMPMultiThread *parent )
{
   ensure( helper.canRunIn( getSMPDevice() ), "Incompatible worker thread" );
   FPGAThread &th = *NEW FPGAThread( helper, this, _core, _fpgaDevice );
   return th;
}

BaseThread & FPGAProcessor::startFPGAThread() {

   WD & worker = getWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   return _core->startThread( *this, worker, NULL );
}
