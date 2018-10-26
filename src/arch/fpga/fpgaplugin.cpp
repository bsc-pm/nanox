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

#include "plugin.hpp"
#include "archplugin.hpp"
#include "fpgaconfig.hpp"
#include "system_decl.hpp"
#include "fpgaprocessor.hpp"
#include "fpgathread.hpp"
#include "fpgadd.hpp"
#include "smpprocessor.hpp"

#include "libxdma.h"

#warning "The FPGA arch is no longer maintained here. The development has been moved to a Nanos++ fork"

namespace nanos {
namespace ext {

class FPGAPlugin : public ArchPlugin
{
   private:
      std::vector< FPGAProcessor* > *_fpgas;
      std::vector< FPGAThread* > *_fpgaThreads;

   public:
      FPGAPlugin() : ArchPlugin( "FPGA PE Plugin", 1 ) {}

      void config( Config& cfg )
      {
         FPGAConfig::prepare( cfg );
      }

      /*!
       * \brief Initialize fpga device plugin.
       * Load config and initialize xilinx dma library
       */
      void init()
      {
         fatal0( "=================================================================" );
         fatal0( "== The FPGA support is no longer maintained here.              ==" );
         fatal0( "== The development has been moved to a Nanos++ fork.           ==" );
         fatal0( "== Visit https://pm.bsc.es/ompss-at-fpga for more information. ==" );
         fatal0( "=================================================================" );

         FPGAConfig::apply();
         _fpgas = NEW std::vector< FPGAProcessor* >( FPGAConfig::getNumFPGAThreads(),( FPGAProcessor* )NULL) ;
         _fpgaThreads = NEW std::vector< FPGAThread* >( FPGAConfig::getNumFPGAThreads(),( FPGAThread* )NULL) ;
         /*
          * Initialize dma library here if fpga device is not disabled
          * We must init the DMA lib before any operation using it is performed
          */
         if ( !FPGAConfig::isDisabled() ) {
            if ( sys.getRegionCachePolicyStr().compare( "fpga" ) != 0 ) {
               if ( sys.getRegionCachePolicyStr().compare( "" ) != 0 ) {
                  warning0( "Switching the cache-policy from '" << sys.getRegionCachePolicyStr() << "' to 'fpga'" );
               } else {
                  debug0( "Setting the cache-policy option to 'fpga'" );
               }
               sys.setRegionCachePolicyStr( "fpga" );
            }

            debug0( "xilinx dma initialization" );
            //Instrumentation has not been initialized yet so we cannot trace things yet
            int status = xdmaOpen();
            //Abort if dma library failed to initialize
            //Otherwise this will cause problems (segfaults/hangs) later on the execution
            if (status)
               fatal0( "Error initializing DMA library: Returned status: " << status );
            //get a core to run the helper thread. First one available
            for ( int i = 0; i<FPGAConfig::getNumFPGAThreads(); i++ ) {

//               memory_space_id_t memSpaceId = sys.getNewSeparateMemoryAddressSpaceId();
//               SeparateMemoryAddressSpace *fpgaAddressSpace =
//                  NEW SeparateMemoryAddressSpace( memSpaceId, nanos::ext::FPGA, true );

               memory_space_id_t memSpaceId = sys.addSeparateMemoryAddressSpace(
                     nanos::ext::FPGA, true, 0 );
               SeparateMemoryAddressSpace &fpgaAddressSpace = sys.getSeparateMemory( memSpaceId );
               fpgaAddressSpace.setNodeNumber( 0 ); //there is only 1 node on this machine
               ext::SMPProcessor *core;
               core = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
               if ( !core ) {
                  //TODO: Run fpga threads in the core running least threads
                  warning0( "Unable to get free core to run the FPGA thread, using the first one" );
                  core = sys.getSMPPlugin()->getFirstSMPProcessor();
                  core->setNumFutureThreads( core->getNumFutureThreads()+1 );
               } else {
                  core->setNumFutureThreads( 1 );
               }
               FPGAProcessor* fpga = NEW FPGAProcessor( memSpaceId, core );
               (*_fpgas)[i] = fpga;
            }
         } //!FPGAConfig::isDisabled()
      }
      /*!
       * \brief Finalize plugin and close dma library.
       */
      void finalize() {
         /*
          * After the plugin is unloaded, no more operations regarding the DMA
          * library nor the FPGA device will be performed so it's time to close the dma lib
          */
         if ( !FPGAConfig::isDisabled() ) { //cleanup only if we have initialized
            int status;
            debug0( "Xilinx close dma" );
            status = xdmaClose();
            if ( status ) {
               warning( "Error de-initializing xdma core library: " << status );
            }
         }
      }

      virtual unsigned getNumHelperPEs() const {
         return FPGAConfig::getNumFPGAThreads();
      }

      virtual unsigned getNumPEs() const {
         return getNumHelperPEs();
      }

      virtual unsigned getNumThreads() const {
         return getNumWorkers() /* + getNumHelperThreads() */;
      }

      virtual unsigned getNumWorkers() const {
         return FPGAConfig::getNumFPGAThreads();
      }

      virtual void addPEs( PEMap &pes  ) const {
          for ( std::vector<FPGAProcessor*>::const_iterator it = _fpgas->begin();
                  it != _fpgas->end(); it++ )
          {
              pes.insert( std::make_pair( (*it)->getId(), *it) );
          }
      }

      virtual void addDevices( DeviceList &devices ) const {
         if ( !_fpgas->empty() ) {
            //Insert device type.
            //Any position in _fpgas will work as we only need the device type
            std::vector<const Device*> const &pe_archs = ( *_fpgas->begin() )->getDeviceTypes();
            for ( std::vector<const Device *>::const_iterator it = pe_archs.begin();
                  it != pe_archs.end(); it++ ) {
               devices.insert( *it );
            }
         }
      }

      virtual void startSupportThreads () {
         for ( unsigned int i = 0; i<_fpgas->size(); i++ ) {
            FPGAProcessor *fpga = ( *_fpgas )[i];
            ( *_fpgaThreads )[i] = ( FPGAThread* ) &fpga->startFPGAThread();
         }
      }

      virtual void startWorkerThreads( std::map<unsigned int, BaseThread*> &workers ) {
          for ( std::vector<FPGAThread*>::iterator it = _fpgaThreads->begin();
                  it != _fpgaThreads->end(); it++ )
          {
              workers.insert( std::make_pair( (*it)->getId(), *it ));
          }
      }

      virtual ProcessingElement * createPE( unsigned id , unsigned uid) {
         return NULL;
      }
};

}
}

DECLARE_PLUGIN("arch-fpga",nanos::ext::FPGAPlugin);
