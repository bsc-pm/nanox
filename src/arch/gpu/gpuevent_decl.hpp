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

#ifndef _GPU_EVENT_DECL
#define _GPU_EVENT_DECL

#include "genericevent_decl.hpp"

#include <cuda_runtime.h>


namespace nanos {

   class GPUEvent : public GenericEvent
   {
      private:
         int            _timesToQuery;
         cudaEvent_t    _cudaEvent;
         cudaStream_t   _cudaStream;

         void updateState();

      public:
        /*! \brief GPUEvent constructor
         */
#ifdef NANOS_GENERICEVENT_DEBUG
         GPUEvent ( WD *wd, cudaStream_t stream = 0, std::string desc = "" );
#else
         GPUEvent ( WD *wd, cudaStream_t stream = 0 );
#endif

         /*! \brief GPUEvent constructor
          */
#ifdef NANOS_GENERICEVENT_DEBUG
         GPUEvent ( WD *wd, ActionList next, cudaStream_t stream = 0, std::string desc = "" );
#else
         GPUEvent ( WD *wd, ActionList next, cudaStream_t stream = 0 );
#endif

        /*! \brief GPUEvent destructor
         */
         ~GPUEvent();

         // set/get methods
         bool isPending();
         void setPending();
         bool isRaised();
         void setRaised();

         // event synchronization related methods
         void waitForEvent();
   };
} // namespace nanos

#endif //_GPU_EVENT_DECL
