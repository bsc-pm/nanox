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

#ifndef _NANOX_THREADTEAM_DATA
#define _NANOX_THREADTEAM_DATA

#include "omp.h"
#include "threadteam_decl.hpp"

namespace nanos {
   namespace OpenMP {

      class OmpThreadTeamData : public ThreadTeamData
      {
         private:
            int _activeLevel;

            /*! \brief ThreadTeamData copy constructor (private)
             */
            OmpThreadTeamData ( OmpThreadTeamData &ttd );

            /*! \brief OmpThreadTeamData copy assignment operator (private)
             */
            OmpThreadTeamData& operator=  ( OmpThreadTeamData &ttd );

         public:
            /*! \brief OmpThreadTeamData default constructor
             */
            OmpThreadTeamData() : ThreadTeamData(), _activeLevel(0) {}

            /*! \brief OmpThreadTeamData destructor
             */
            virtual ~OmpThreadTeamData() {}

            int getActiveLevel() { return _activeLevel; }

            virtual void init( ThreadTeam * parent)
            {
               if ( parent == NULL ) {
                  _activeLevel = 0;
               } else {
                  _activeLevel = ((OmpThreadTeamData &)parent->getThreadTeamData())._activeLevel + ( parent->size() > 1 ? 1 : 0);
               }
            }
      };
      
   }
} // namespace nanos

#endif

