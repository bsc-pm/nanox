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


#define DEFAULT_CUTOFF_LEVEL 2

using namespace nanos;


class level_cutoff: public cutoff
{

   private:
      int max_level;

   public:
      level_cutoff() : max_level( DEFAULT_CUTOFF_LEVEL ) {}

      void init() {}

      void setMaxCutoff( int ml ) { max_level = ml; }

      bool cutoff_pred();

      ~level_cutoff() {}
};


bool level_cutoff::cutoff_pred()
{
   //checking the parent level of the next work to be created (check >)
   if ( ( myThread->getCurrentWD() )->getLevel() > max_level )  {
      verbose0( "Cutoff Policy: avoiding task creation!" );
      return false;
   }

   return true;
}

//factory
cutoff * createLevelCutoff()
{
   return new level_cutoff();
}


class LevelCutOffPlugin : public Plugin
{

   public:
      LevelCutOffPlugin() : Plugin( "Task Tree Level CutOff Plugin",1 ) {}

      virtual void init() {
         sys.setCutOffPolicy( createLevelCutoff() );
      }
};

LevelCutOffPlugin NanosXPlugin;
