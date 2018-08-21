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

#include "taskexecutionexception_decl.hpp"

using namespace nanos;

const char* TaskExecutionException::what ( ) const throw ()
{
   std::string s(runtime_error::what());

   const char* sig_desc;
   if (signal_info.si_signo >= 0 && signal_info.si_signo < NSIG && (sig_desc =
         _sys_siglist[signal_info.si_signo]) != NULL) {

      s += sig_desc;
      switch (signal_info.si_signo) {
         // Check {glibc_include_path}/bits/{siginfo.h, signum.h}
         case SIGILL:
            switch (signal_info.si_code) {
               case ILL_ILLOPC:
                  s += " Illegal opcode.";
                  break;
               case ILL_ILLOPN:
                  s += " Illegal operand.";
                  break;
               case ILL_ILLADR:
                  s += " Illegal addressing mode.";
                  break;
               case ILL_ILLTRP:
                  s += " Illegal trap.";
                  break;
               case ILL_PRVOPC:
                  s += " Privileged opcode.";
                  break;
               case ILL_PRVREG:
                  s += " Privileged register.";
                  break;
               case ILL_COPROC:
                  s += " Coprocessor error.";
                  break;
               case ILL_BADSTK:
                  s += " Internal stack error.";
                  break;
            }

            break;
         case SIGFPE:
            switch (signal_info.si_code) {

               case FPE_INTDIV:
                  s += " Integer divide by zero.";
                  break;
               case FPE_INTOVF:
                  s += " Integer overflow.";
                  break;
               case FPE_FLTDIV:
                  s += " Floating-point divide by zero.";
                  break;
               case FPE_FLTOVF:
                  s += " Floating-point overflow.";
                  break;
               case FPE_FLTUND:
                  s += " Floating-point underflow.";
                  break;
               case FPE_FLTRES:
                  s += " Floating-poing inexact result.";
                  break;
               case FPE_FLTINV:
                  s += " Invalid floating-point operation.";
                  break;
               case FPE_FLTSUB:
                  s += " Subscript out of range.";
                  break;
            }
            break;
         case SIGSEGV:
            switch (signal_info.si_code) {

               case SEGV_MAPERR:
                  s += " Address not mapped to object.";
                  break;
               case SEGV_ACCERR:
                  s += " Invalid permissions for mapped object.";
                  break;
            }
            break;
         case SIGBUS:
            switch (signal_info.si_code) {

               case BUS_ADRALN:
                  s += " Invalid address alignment.";
                  break;
               case BUS_ADRERR:
                  s += " Nonexisting physical address.";
                  break;
               case BUS_OBJERR:
                  s += " Object-specific hardware error.";
                  break;
#ifdef BUS_MCEERR_AR
                  case BUS_MCEERR_AR: //(since Linux 2.6.32)
                  s += " Hardware memory error consumed on a machine check; action required.";
                  break;
#endif
#ifdef BUS_MCEERR_AO
                  case BUS_MCEERR_AO: //(since Linux 2.6.32)
                  s += " Hardware memory error detected in process but not consumed; action optional.";
                  break;
#endif
            }
            break;
         case SIGTRAP:
            switch (signal_info.si_code) {

               case TRAP_BRKPT:
                  s += " Process breakpoint.";
                  break;
               case TRAP_TRACE:
                  s += " Process trace trap.";
                  break;
            }
            break;

            //default:
            /*
             * note #1: since this exception is going to be thrown by the signal handler
             * only synchronous signals information will be printed, as the remaining
             * are unsupported by -fnon-call-exceptions
             */
      }
   } else {
      /*
       * See note #1
       */
      s += " Unsupported signal (";
      s += signal_info.si_signo;
      s += " )";
   }

   return s.c_str();
}
