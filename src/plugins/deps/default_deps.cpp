/*************************************************************************************/
/*      Copyright 2012 Barcelona Supercomputing Center                               */
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

#include "dependenciesdomain.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

namespace nanos {
   namespace ext {

      class NanosDependencies : public DependenciesManager
      {
         public:
            NanosDependencies() : DependenciesManager("Nanos default dependencies manager") {}
            virtual ~NanosDependencies () {}
      };
  
      class NanosDepsPlugin : public Plugin
      {
         public:
            NanosDepsPlugin() : Plugin( "Nanos++ default dependencies management plugin",1 )
            {
            }

            virtual void config ( Config &cfg )
            {
            }

            virtual void init() {
               sys.setDefaultDependenciesManager(NEW NanosDependencies());
            }
      };

   }
}

DECLARE_PLUGIN("deps-default",nanos::ext::NanosDepsPlugin);
