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

using namespace nanos;

const char* TaskExecutionException::what ( ) const
{
   std::stringstream ss
   << "Signal raised. ";

   char* sig_desc;
   if (info.si_signo >= 0 && info.si_signo < NSIG && (sig_desc =
         _sys_siglist[info.si_signo]) != NULL) {
      switch (info.si_signo) {
         case SIGILL:
            ss << "SIGILL: ";
            switch (info.si_code) {
               case ILL_ILLOPC:
                  ss << "Illegal opcode.";
                  break;
               case ILL_ILLOPN:
                  ss << "Illegal operand.";
                  break;
               case ILL_ILLADR:
                  ss << "Illegal addressing mode.";
                  break;
               case ILL_ILLTRP:
                  ss << "Illegal trap.";
                  break;
               case ILL_PRVOPC:
                  ss << "Privileged opcode.";
                  break;
               case ILL_PRVREG:
                  ss << "Privileged register.";
                  break;
               case ILL_COPROC:
                  ss << "Coprocessor error.";
                  break;
               case ILL_BADSTK:
                  ss << "Internal stack error.";
                  break;
            }

            break;
         case SIGFPE:
            ss << "SIGFPE: ";
            switch (info.si_code) {

               case FPE_INTDIV:
                  ss << "Integer divide by zero.";
                  break;
               case FPE_INTOVF:
                  ss << "Integer overflow.";
                  break;
               case FPE_FLTDIV:
                  ss << "Floating-point divide by zero.";
                  break;
               case FPE_FLTOVF:
                  ss << "Floating-point overflow.";
                  break;
               case FPE_FLTUND:
                  ss << "Floating-point underflow.";
                  break;
               case FPE_FLTRES:
                  ss << "Floating-poing inexact result.";
                  break;
               case FPE_FLTINV:
                  ss << "Invalid floating-point operation.";
                  break;
               case FPE_FLTSUB:
                  ss << "Subscript out of range.";
                  break;
            }
            break;
         case SIGSEGV:
            ss << "SIGSEGV: ";
            switch (info.si_code) {

               case SEGV_MAPERR:
                  ss << "Address not mapped to object.";
                  break;
               case SEGV_ACCERR:
                  ss << "Invalid permissions for mapped object.";
                  break;
            }
            break;
         case SIGBUS:
            ss << "SIGBUS: ";
            switch (info.si_code) {

               case BUS_ADRALN:
                  ss << "Invalid address alignment.";
                  break;
               case BUS_ADRERR:
                  ss << "Nonexisting physical address.";
                  break;
               case BUS_OBJERR:
                  ss << "Object-specific hardware error.";
                  break;
               case BUS_MCEERR_AR: //(since Linux 2.6.32)
                  code =
                        "Hardware memory error consumed on a machine check; action required.";
                  break;
               case BUS_MCEERR_AO: //(since Linux 2.6.32)
                  code =
                        "Hardware memory error detected in process but not consumed; action optional.";
                  break;
            }
            break;
         case SIGTRAP:
            ss << "SIGTRAP: ";
            switch (info.si_code) {

               case TRAP_BRKPT:
                  ss << "Process breakpoint.";
                  break;
               case TRAP_TRACE:
                  ss << "Process trace trap.";
                  break;
            }
            break;
         default:
            /*
             * note #1: since this exception is going to be thrown by the signal handler
             * only synchronous signals information will be printed, as the remaining
             * are unsupported by -fnon-call-exceptions
             */
            ss << "Unsupported signal (" << info.si_signo << ")";
      }
   } else {
      /*
       * See note #1
       */
      ss << "Unsupported signal (" << info.si_signo << ")";
   }

   return ss.c_str();
}
