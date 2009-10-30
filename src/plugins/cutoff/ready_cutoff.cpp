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

#include "system.hpp"
#include "cutoff.hpp"
#include "plugin.hpp"


#define DEFAULT_CUTOFF_READY 100

using namespace nanos;


class ready_cutoff: public cutoff
{

   private:
      int max_ready;

   public:
      ready_cutoff() : max_ready( DEFAULT_CUTOFF_READY ) {}

      void init() {}

      void setMaxCutoff( int mr ) { max_ready = mr; }

      bool cutoff_pred();

      ~ready_cutoff() {}
};


bool ready_cutoff::cutoff_pred()
{
   //checking if the number of ready tasks is higher than the allowed maximum
   if ( sys.getReadyNum() > max_ready )  {
      verbose0( "Cutoff Policy: avoiding task creation!" );
      return false;
   }

   return true;
}

//factory
cutoff * createReadyCutoff()
{
   return new ready_cutoff();
}


class ReadyCutOffPlugin : public Plugin
{

   public:
      ReadyCutOffPlugin() : Plugin( "Ready Task CutOff Plugin",1 ) {}

      virtual void init() {
         sys.setCutOffPolicy( createReadyCutoff() );
      }
};

ReadyCutOffPlugin NanosXPlugin;
