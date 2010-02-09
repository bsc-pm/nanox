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

#include "nanos.h"
#include "debug.hpp"
#include "system.hpp"
#include "plugin.hpp"

using namespace nanos;

/*! \brief Find a slicer giving a label id
 *
 *  \sa Slicers
 */
nanos_slicer_t nanos_find_slicer ( const char * label )
{
   try
   {
      nanos_slicer_t slicer;
      std::string plugin = "slicer-" + std::string(label);

      slicer = sys.getSlicer ( std::string(label) );
      if ( slicer == NULL ) {
         if ( !PluginManager::load( plugin )) fatal0( "Could not load " + std::string(label) + "slicer" );
         slicer = sys.getSlicer ( std::string(label) );
      }
      return slicer;

   } catch ( ... ) {
      return ( nanos_slicer_t ) NULL;
   }
}

// xteruel: FIXME: to remove ?
#if 0
nanos_instrumentor_t nanos_find_instrumentor ( const char * label )
{
   try
   {
      nanos_instrumentor_t instrumentor;
      std::string plugin = "instrumentor-" + std::string(label);

      instrumentor = sys.getInstrumentor ( std::string(label) );
      if ( instrumentor == NULL ) {
         if ( !PluginManager::load( plugin )) fatal0( "Could not load " + std::string(label) + "instrumentor" );
         instrumentor = sys.getInstrumentor ( std::string(label) );
      }
      return instrumentor;

   } catch ( ... ) {
      return ( nanos_instrumentor_t ) NULL;
   }
}
#endif
