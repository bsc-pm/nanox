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

#ifndef _NANOS_TASKEXECUTIONEXCEPTION
#define _NANOS_TASKEXECUTIONEXCEPTION

#include "taskexecutionexception_decl.hpp"
#include "xstring.hpp"

#define TEE_ERR_MSG(id) "Signal raised during the execution of task "+toString<int>(id)+".\n"

namespace nanos {

   TaskExecutionException::TaskExecutionException (
         WD const *task_wd, siginfo_t const &info,
         ucontext_t const &context ) throw () :
         runtime_error(TEE_ERR_MSG(task_wd->getId())), task(task_wd), signal_info(
               info), task_context(context)
   {
   }

   TaskExecutionException::TaskExecutionException (
         TaskExecutionException const &tee ) throw () :
         runtime_error(TEE_ERR_MSG(tee.task->getId())), task(tee.task), signal_info(
               tee.signal_info), task_context(tee.task_context)
   {
   }

   TaskExecutionException::~TaskExecutionException ( ) throw ()
   {
      /*
       * Note that this destructor does not delete the WorkDescriptor object pointed by 'task'.
       * This is because that object's life does not finish at this point and,
       * thus, it will be accessed later.
       */
   }

   inline int TaskExecutionException::getSignal ( )
   {
      return signal_info.si_signo;
   }

   inline const siginfo_t TaskExecutionException::getSignalInfo ( ) const
   {
      return signal_info;
   }

   inline const ucontext_t TaskExecutionException::getExceptionContext ( ) const
   {
      return task_context;
   }

} // namespace nanos

#endif /* _NANOS_TASKEXECUTIONEXCEPTION */
