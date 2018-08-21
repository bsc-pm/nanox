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

#include "fpgadd.hpp"
#include "fpgaprocessor.hpp"

using namespace nanos;
using namespace nanos::ext;

FPGADevice nanos::ext::FPGA( "FPGA" );

FPGADD * FPGADD::copyTo ( void *toAddr )
{
   //Construct into a given address (toAddr) since we are copying
   //we are not allocating anithind, therefore, system allocator cannot be used here
   FPGADD *dd = new ( toAddr ) FPGADD( *this );
   return dd;
}

bool FPGADD::isCompatibleWithPE ( const ProcessingElement *pe ){

   if (pe == NULL) return true;
   nanos::ext::FPGAProcessor* myPE = ( nanos::ext::FPGAProcessor* ) pe;
   int accBase = myPE->getAccelBase();
   int numAcc = myPE->getNumAcc();

   // Currently each fpga processor (helper thread) manages numAcc starting from accBase.
   bool compatible = _accNum < 0 || ( ( _accNum >= accBase ) && ( _accNum < ( accBase + numAcc ) ) );
   //once checked if current PE is compatible with this task(wd) it is going to run it so
   //we set the accelerator on which the task may be run.
   if (compatible) {
      if ( _accNum < 0 ) {
         //If the user has not specified any accelerator (which means that can run in any)
         //assign one of them in a round robin fashion
         myPE->setActiveAcc(-1);
         myPE->setNextAccelerator();
      } else {
         //myPE->setUpdate(false);
         myPE->setActiveAcc(_accNum - accBase);
      }
   }

   return compatible;

}
