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

#ifndef _NANOS_LIB_DEBUG
#define _NANOS_LIB_DEBUG

#include <stdexcept>
//Having system.hpp here generate too many circular dependences
//but it's not really needed so we can delay it most times until the actual usage
//#include "system.hpp"
#include "xstring.hpp"
#include <iostream>

namespace nanos {

   class FatalError : public  std::runtime_error
   {

      public:
         FatalError ( const std::string &value, int peId=-1 ) :
               runtime_error( std::string( "FATAL ERROR: [" ) + toString<int>( peId ) + "] " + value ) {}

   };

   class FailedAssertion : public  std::runtime_error
   {

      public:
         FailedAssertion ( const char *file, const int line, const std::string &value,
                           const std::string msg, int peId=-1 ) :
               runtime_error(
                  std::string( "ASSERT failed: [" )+ toString<int>( peId ) + "] "
                  + value + ":" + msg
                  + " (" + file + ":" + toString<int>( line )+ ")" ) {}

   };

   void printBt( std::ostream &o );

} // namespace nanos

#define _nanos_ostream ( /* myThread ? *(myThread->_file) : */ std::cerr )

#define fatal(msg) { std::stringstream sts; sts<<msg ; throw nanos::FatalError(sts.str(),getMyThreadSafe()->getId()); }
#define fatal0(msg)  { std::stringstream sts; sts<<msg ; throw nanos::FatalError(sts.str()); }
#define fatal_cond(cond,msg) if ( cond ) fatal(msg);
#define fatal_cond0(cond,msg) if ( cond ) fatal0(msg);

#define warning(msg) { _nanos_ostream << "WARNING: [" << std::dec << getMyThreadSafe()->getId() << "]" << msg << std::endl; }
#define warning0(msg) { _nanos_ostream << "WARNING: [?]" << msg << std::endl; }

#define message(msg) \
   _nanos_ostream << "MSG: [" << std::dec << getMyThreadSafe()->getId() << "] " << msg << std::endl;
#define message0(msg) \
   _nanos_ostream << "MSG: [?] " << msg << std::endl;

#define messageMaster(msg) \
   do { if (sys.getNetwork()->getNodeNum() == 0) { _nanos_ostream << "MSG: m:[" << std::dec << getMyThreadSafe()->getId() << "] " << msg << std::endl; } } while (0)
#define message0Master(msg) \
   do { if (sys.getNetwork()->getNodeNum() == 0) { _nanos_ostream << "MSG: m:[?] " << msg << std::endl; } } while (0)

#ifdef NANOS_DEBUG_ENABLED
#define ensure(cond,msg) if ( !(cond) ) throw nanos::FailedAssertion(__FILE__, __LINE__ , #cond, msg, getMyThreadSafe()->getId());
#define ensure0(cond,msg) if ( !(cond) ) throw nanos::FailedAssertion(__FILE__, __LINE__, #cond, msg );

#define verbose(msg) \
   if (sys.getVerbose()) _nanos_ostream << "[" << std::dec << getMyThreadSafe()->getId() << "]" << msg << std::endl;
#define verbose0(msg) \
   if (sys.getVerbose()) _nanos_ostream << "[?]" << msg << std::endl;

#define debug(msg) \
   if (sys.getVerbose()) _nanos_ostream << "DBG: [" << std::dec << getMyThreadSafe()->getId() << "]" << msg << std::endl;
#define debug0(msg) \
   if (sys.getVerbose()) _nanos_ostream << "DBG: [?]" << msg << std::endl;
#else
#define ensure(cond,msg)
#define ensure0(cond,msg)
#define verbose(msg)
#define verbose0(msg)
#define debug(msg)
#define debug0(msg)
#endif

#endif
