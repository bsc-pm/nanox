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

#ifndef _NANOS_TASKEXECUTIONEXCEPTION_DECL
#define _NANOS_TASKEXECUTIONEXCEPTION_DECL

#include "workdescriptor_decl.hpp"
#include <exception>
#include <signal.h>
#include <ucontext.h>

namespace nanos {

   /*!
    * \class TaskExecutionException
    * \brief Contains usefull information about a runtime error generated in a task execution.
    */
   class TaskExecutionException: public std::runtime_error
   {
      private:
         const WD* task;/*!< Pointer to the affected task */
         const siginfo_t signal_info;/*!< Detailed description after the member */
         const ucontext_t task_context;/*!< Detailed description after the member */

      public:
         /*!
          * Constructor for class TaskExecutionException
          * \param task a pointer to the task where the error appeared
          * \param info information about the signal raised
          * \param context contains the state of execution when the error appeared
          */
         TaskExecutionException ( WD const *task, siginfo_t const &info,
                                  ucontext_t const &context ) throw ();

         /*!
          * Copy constructor for class TaskExecutionException
          */
         TaskExecutionException ( TaskExecutionException const &tee ) throw ();

         /*!
          * Destructor for class TaskExecutionException
          */
         virtual ~TaskExecutionException ( ) throw ();

         /*!
          * Returns some information about the error in text format.
          */
         virtual const char* what ( ) const throw ();

         /*!
          * \return the raised signal number
          */
         int getSignal ( );

         /*!
          * \return the structure containing the signal information.
          * \see siginfo_t
          */
         const siginfo_t getSignalInfo ( ) const;

         /*!
          * \return the structure conteining the execution status when the error appeared
          * \see ucontext_t
          */
         const ucontext_t getExceptionContext ( ) const;
   };

} // namespace nanos

#endif /* _NANOS_TASKEXECUTIONEXCEPTION_DECL */
