/**************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                    */
/*      Copyright 2009 Barcelona Supercomputing Center                    */
/*                                                                        */
/*      This file is part of the NANOS++ library.                         */
/*                                                                        */
/*      NANOS++ is free software: you can redistribute it and/or modify   */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or  */
/*      (at your option) any later version.                               */
/*                                                                        */
/*      NANOS++ is distributed in the hope that it will be useful,        */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     */
/*      GNU Lesser General Public License for more details.               */
/*                                                                        */
/*      You should have received a copy of the GNU Lesser General Public License  */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.  */
/**************************************************************************/

#ifndef TASKEXECUTIONEXCEPTION_HPP_
#define TASKEXECUTIONEXCEPTION_HPP_

#include "taskexecutionexception_decl.hpp"
#include <stringstream>

using namespace nanos;

inline TaskExecutionException::TaskExecutionException ( siginfo_t& info,
                                                        ucontext_t& context ) :
      info(info), context(context)
{
}

inline const int TaskExecutionException::getSignal ( ) const
{
   return info.si_signo;
}

inline const siginfo_t TaskExecutionException::getSignalInfo ( ) const
{
   return info;
}

inline const ucontext_t TaskExecutionException::getExceptionContext ( ) const
{
   return context;
}

#endif /* TASKEXECUTIONEXCEPTION_HPP_ */
