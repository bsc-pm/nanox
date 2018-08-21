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
#include "plugin.hpp"
// We need to include system.hpp (to use verbose0(msg)), as debug.hpp does not include it
#include "system.hpp"

namespace nanos
{
   namespace ext
   {
      int  FPGAConfig::_numAccelerators = -1;
      int  FPGAConfig::_numFPGAThreads = -1;
      bool FPGAConfig::_disableFPGA = false;
      Lock FPGAConfig::_dmaLock;
      Atomic <int> FPGAConfig::_accelID(0);
      const int FPGAConfig::_maxAccelerators = 2;
      //TODO set sensible defaults (disabling transfers when necessary, etc.)
      unsigned int FPGAConfig::_burst = 8;
      int FPGAConfig::_maxTransfers = 32;
      int FPGAConfig::_idleSyncBurst = 4;
      bool FPGAConfig::_syncTransfers = false;

      void FPGAConfig::prepare( Config &config )
      {
         config.setOptionsSection( "FPGA Arch", "FPGA spefific options" );
         config.registerConfigOption( "num-fpga" , NEW Config::IntegerVar( _numAccelerators ),
                                      "Defines de number of FPGA acceleratos to use (defaults to one)" );
         config.registerEnvOption( "num-fpga", "NX_FPGA_NUM" );
         config.registerArgOption( "num-fpga", "fpga-num" );

         config.registerConfigOption( "disable-fpga", NEW Config::FlagOption( _disableFPGA ),
                                      "Disable the use of FPGA accelerators" );
         config.registerEnvOption( "disable-fpga", "NX_DISABLE_FPGA" );
         config.registerArgOption( "disable-fpga", "disable-fpga" );

         config.registerConfigOption( "fpga-burst", NEW Config::UintVar( _burst ),
                 "Defines the number of transfers fo be waited in a row when the maximum active transfer is reached (-1 acts as unlimited)");
         config.registerEnvOption( "fpga-burst", "NX_FPGA_BURST" );
         config.registerArgOption( "fpga-burst", "fpga-burst" );

         config.registerConfigOption( "fpga_helper_threads", NEW Config::IntegerVar( _numFPGAThreads ),
                 "Defines de number of helper threads managing fpga accelerators");
         config.registerEnvOption( "fpga_helper_threads", "NX_FPGA_HELPER_THREADS" );
         config.registerArgOption( "fpga_helper_threads", "fpga-helper-threads" );

         config.registerConfigOption( "fpga_max_transfers", NEW Config::IntegerVar( _maxTransfers ),
                 "Defines the maximum number of active transfers per dma accelerator channel (-1 behaves as unlimited)" );
         config.registerEnvOption( "fpga_max_transfers", "NX_FPGA_MAX_TRANSFERS" );
         config.registerArgOption( "fpga_max_transfers", "fpga-max-transfers" );

         config.registerConfigOption( "fpga_idle_sync_burst", NEW Config::IntegerVar( _idleSyncBurst ),
               "Number of transfers synchronized when calling thread's idle" );
         config.registerEnvOption( "fpga_idle_sync_burst", "NX_FPGA_IDLE_SYNC_BURST" );
         config.registerArgOption( "fpga_idle_sync_burst", "fpga-idle-sync-burst" );

         config.registerConfigOption( "fpga_sync_transfers", NEW Config::FlagOption( _syncTransfers ),
               "Perform fpga transfers synchronously" );
         config.registerEnvOption( "fpga_sync_transfers", "NX_FPGA_SYNC_TRANSFERS" );
         config.registerArgOption( "fpga_sync_transfers", "fpga-sync-transfers" );

      }

      void FPGAConfig::apply()
      {
         verbose0( "Initializing FPGA support component" );

         if ( _disableFPGA ) {
            _numAccelerators = 0; //system won't instanciate accelerators if count=0
         } else if ( _numAccelerators < 0 ) {
            /* if not given, assume we are using one accelerator
             * We should get the number of accelerators on the system
             * and use it as a default
             */
            _numAccelerators = 1;
         } else if ( _numAccelerators == 0 ) {
             _disableFPGA = true;
         }

         if (_numAccelerators > _maxAccelerators) {
             warning0( "The number of accelerators is greater then the accelerators in the system. Using "
                     << _maxAccelerators << " accelerators" );
             _numAccelerators = _maxAccelerators;
         }

         if ( _numFPGAThreads < 0 ) {
            warning0( "Number of fpga threads cannot be negative. Using one thread per accelerator" );
            _numFPGAThreads = _numAccelerators;
         } else if ( _numFPGAThreads > _numAccelerators ) {
            warning0( "Number of helper is greater than the number of accelerators. Using one thread per accelerator" );
            _numFPGAThreads = _numAccelerators;
         }
         _idleSyncBurst = ( _idleSyncBurst < 0 ) ? _burst : _idleSyncBurst;

      }
   }
}


