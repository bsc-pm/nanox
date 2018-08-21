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

#ifndef _OpenCL_EVENT_DECL
#define _OpenCL_EVENT_DECL

#include "genericevent_decl.hpp"

namespace nanos {

   class OpenCLEvent : public GenericEvent
   {
      private:
         int            _timesToQuery;
         cl_event*    _openclEvents;
         int _numEvents;
         void* _runningKernel;
         bool _usedEvent;

         void updateState();

      public:
        /*! \brief OpenCLEvent constructor
         */
#ifdef NANOS_GENERICEVENT_DEBUG
         OpenCLEvent ( WD *wd, cl_context& context, std::string desc = "" );
#else
         OpenCLEvent ( WD *wd, cl_context& context );
#endif

         /*! \brief OpenCLEvent constructor
          */
#ifdef NANOS_GENERICEVENT_DEBUG
         OpenCLEvent ( WD *wd, ActionList next, cl_context& context, std::string desc = "" );
#else
         OpenCLEvent ( WD *wd, ActionList next, cl_context& context );
#endif

        /*! \brief OpenCLEvent destructor
         */
         ~OpenCLEvent();

         // set/get methods
         bool isPending();
         void setPending();
         bool isRaised();
         void setRaised();
         
         /**
          * *WARNING* If you get this event, you MUST enqueue into an OpenCL queue
          * @return Native OCL event for this event
          */
         cl_event& getCLEvent();
         void* getCLKernel();
         void setCLKernel(void* currKernel);

         // event synchronization related methods
         void waitForEvent();
   };
} // namespace nanos

#endif //_OpenCL_EVENT_DECL
