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


#ifndef _DEV_INSTR
#define _DEV_INSTR


namespace nanos {

   class InstrCopyDirDevices {

      public:

         typedef enum {
            NANOS_DEVS_CPDIR_NULL_EVENT,          /* 0 */
            NANOS_DEVS_CPDIR_H2D_GPU_EVENT,       /* 1 */
            NANOS_DEVS_CPDIR_D2H_GPU_EVENT,       /* 2 */
            NANOS_DEVS_CPDIR_H2D_HSTR_EVENT,      /* 3 */
            NANOS_DEVS_CPDIR_D2H_HSTR_EVENT,      /* 4 */
         } CopyDirValues;

   };


// Macro's to instrument the code and make it cleaner
#define NANOS_INSTR_OPEN_CP_DIR_DEVS_EVENT(x)     NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseOpenBurstEvent ( \
            sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "copy-dir-devices" ), (x) ); )

#define NANOS_INSTR_CLOSE_CP_DIR_DEVS_EVENT       NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseCloseBurstEvent ( \
            sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "copy-dir-devices" ), 0 ); )

} // namespace nanos

#endif
