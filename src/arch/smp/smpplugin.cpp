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

#include "plugin.hpp"
#include "smpprocessor.hpp"
#include "smpdd.hpp"
#include "system.hpp"
#include "archplugin.hpp"

namespace nanos {
namespace ext {

nanos::PE * smpProcessorFactory ( int id, int uid );

nanos::PE * smpProcessorFactory ( int id, int uid )
{
   return NEW SMPProcessor( id, uid );
}

class SMPPlugin : public ArchPlugin
{

   public:
      SMPPlugin() : ArchPlugin( "SMP PE Plugin",1 ) {}

      virtual void config ( Config& cfg )
      {
         cfg.setOptionsSection( "SMP Arch", "SMP specific options" );
         SMPProcessor::prepareConfig( cfg );
         SMPDD::prepareConfig( cfg );
      }

      virtual void init() {
         sys.setHostFactory( smpProcessorFactory );
      }
      
      virtual unsigned getPEsInNode( unsigned node ) const
      {
         // TODO (gmiranda): if HWLOC is available, use it.
         return sys.getCoresPerSocket();
      }

      virtual unsigned getNumPEs() const
      {
         // TODO (gmiranda): create PEs here and not in system
         return 0;
      }

      virtual unsigned getNumThreads() const
      {
         //return sys.getNumThreads();
         return 0;
      }

      virtual ProcessingElement* createPE( unsigned id, unsigned uid )
      {
         return NULL;
      }
};
}
}

DECLARE_PLUGIN("arch-smp",nanos::ext::SMPPlugin);
