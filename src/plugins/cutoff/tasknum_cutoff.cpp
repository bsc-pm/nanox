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


#define DEFAULT_CUTOFF_NUM 1000

using namespace nanos;


class tasknum_cutoff: public cutoff
{

   private:
      int max_cutoff;

   public:
      tasknum_cutoff() : max_cutoff( DEFAULT_CUTOFF_NUM ) {}

      void init() {}

      void setMaxCutoff( int mc ) { max_cutoff = mc; }

      bool cutoff_pred();

      ~tasknum_cutoff() {}
};


bool tasknum_cutoff::cutoff_pred()
{
   if ( sys.getTaskNum() > max_cutoff ) {
      verbose0( "Cutoff Policy: avoiding task creation!" );
      return false;
   }

   return true;
}

//factory
cutoff * createTasknumCutoff()
{
   return new tasknum_cutoff();
}



class TasknumCutOffPlugin : public Plugin
{

   public:
      TasknumCutOffPlugin() : Plugin( "TaskNum CutOff Plugin",1 ) {}

      virtual void init() {
         sys.setCutOffPolicy( createTasknumCutoff() );
      }
};

TasknumCutOffPlugin NanosXPlugin;
