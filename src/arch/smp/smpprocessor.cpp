/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include <iostream>

namespace nanos
{

   bool SMPProcessor::useUserThreads = true;

   void SMPProcessor::prepareConfig ( Config &config )
   {
      config.registerArgOption( new Config::FlagOption( "nth-no-ut",useUserThreads,false ) );
   }

   WorkDescriptor & SMPProcessor::getWorkerWD () const
   {
      SMPDD * dd = new SMPDD( ( SMPDD::work_fct )Scheduler::idle );
      WD *wd = new WD( dd );
      return *wd;
   }

   WorkDescriptor & SMPProcessor::getMasterWD () const
   {
      WD * wd = new WD( new SMPDD() );
      return *wd;
   }

   BaseThread &SMPProcessor::createThread ( WorkDescriptor &helper )
   {
      ensure( helper.canRunIn( SMP ),"Incompatible worker thread" );
      SMPThread &th = *new SMPThread( helper,this,useUserThreads );

      return th;
   }

};
