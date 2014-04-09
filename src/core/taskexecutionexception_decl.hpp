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

#ifndef TASKEXECUTIONEXCEPTION_DECL_HPP_
#define TASKEXECUTIONEXCEPTION_DECL_HPP_

#include <exception>
#include <siginfo.h>
#include <ucontext.h>
namespace nanos {

   class TaskExecutionException: public std::exception
   {
      private:
         const siginfo_t info;
         const ucontext_t context;

      public:
         TaskExecutionException ( siginfo_t& info, ucontext_t& context);

         virtual const char* what() const;

         const int getSignal() const;

         const siginfo_t getSignalInfo() const;

         const ucontext_t getExceptionContext() const;

   };
}

#endif /* TASKEXECUTIONEXCEPTION_DECL_HPP_ */
