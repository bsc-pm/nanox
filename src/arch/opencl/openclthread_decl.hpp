
/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_OpenCL_THREAD_DECL
#define _NANOS_OpenCL_THREAD_DECL

#include "smpthread.hpp"
#include "opencldd.hpp"

namespace nanos {
namespace ext {
    
class OpenCLThread : public SMPThread
{
private:
   bool _wdClosingEvents; //! controls whether an instrumentation event should be generated at WD completion
   
   OpenCLThread( const OpenCLThread &thr ); // Do not implement.
   const OpenCLThread &operator=( const OpenCLThread &thr ); // Do not implement.
   
   WD * getNextTask ( WD &wd );
   void prefetchNextTask( WD * next );
   void raiseWDClosingEvents ();
   
public:
   OpenCLThread( WD &wd, PE *pe ) : SMPThread( wd, pe ) {}
   ~OpenCLThread() {}
   
   void initializeDependent();
   void runDependent();
   bool inlineWorkDependent( WD &wd );
   void yield();
   void idle();
   void enableWDClosingEvents ();

private:

   //bool checkForAbort( OpenCLDD::event_iterator i, OpenCLDD::event_iterator e );

};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_THREAD_DECL
