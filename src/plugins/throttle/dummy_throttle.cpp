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

#include "throttle_decl.hpp"
#include "plugin.hpp"
#include "system.hpp"

namespace nanos {
namespace ext {

   class DummyThrottle: public ThrottlePolicy
   {

      private:
         /*!
          * we decide once if all new tasks are to be created during the execution
          * if _createTasks is true, then we have the maximum number of tasks else we have only one task (sequential comp.)
          */
         bool _createTasks;

         DummyThrottle ( const DummyThrottle & );
         const DummyThrottle & operator= ( const DummyThrottle & );

      public:
         DummyThrottle( bool ct ) : _createTasks( ct ) {}

         void setCreateTask( bool ct ) { _createTasks = ct; }

         bool throttleIn();

         ~DummyThrottle() {};
   };

   bool DummyThrottle::throttleIn()
   {
      return _createTasks;
   }

   //factory
   DummyThrottle * createDummyThrottle( bool createTasks );

   DummyThrottle * createDummyThrottle( bool createTasks )
   {
      return NEW DummyThrottle( createTasks );
   }

   class DummyThrottlePlugin : public Plugin
   {
      private:
         bool  _createTasks; //<! Default value is 'true'

      public:
         DummyThrottlePlugin() : Plugin( "Simple (all/nothing) Throttle Plugin",1 ), _createTasks( true ) {}

         virtual void config( Config& cfg )
         {
            cfg.setOptionsSection( "Dummy throttle", "Scheduling throttle policy based on fixed behaviour" );
            cfg.registerConfigOption ( "throttle-create-tasks", NEW Config::FlagOption( _createTasks ), "Throttle decides to create all tasks (default: enabled)" );
            cfg.registerArgOption( "throttle-create-tasks", "throttle-create-tasks" );
         }

         virtual void init()
         {
            sys.setThrottlePolicy( createDummyThrottle( _createTasks ) );
         }
   };

} /* namespace ext */
} /* namespace nanos */

DECLARE_PLUGIN("throttle-dummy",nanos::ext::DummyThrottlePlugin);
