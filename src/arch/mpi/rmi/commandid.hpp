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

#ifndef COMMAND_ID_HPP
#define COMMAND_ID_HPP

// MPI Communication tags, we use that many so messages don't collide for different operations
enum {
    TAG_M2S_COMMAND = 1200,
    TAG_M2S_CACHE_COMMAND,    // 1201
    TAG_CACHE_DATA_IN,        // 1202
    TAG_CACHE_DATA_OUT,       // 1203
    TAG_CACHE_ANSWER,         // 1204
    TAG_INI_TASK,             // 1205
    TAG_END_TASK,             // 1206
    TAG_ENV_STRUCT,           // 1207
    TAG_CACHE_ANSWER_REALLOC, // 1208
    TAG_CACHE_ANSWER_ALLOC,   // 1209
    TAG_CACHE_ANSWER_CIN,     // 1210
    TAG_CACHE_ANSWER_COUT,    // 1211
    TAG_CACHE_ANSWER_FREE,    // 1212
    TAG_CACHE_ANSWER_DEV2DEV, // 1213
    TAG_CACHE_ANSWER_CL,      // 1214
    TAG_FP_NAME_SYNC,         // 1215
    TAG_FP_SIZE_SYNC,         // 1216
    TAG_CACHE_DEV2DEV,        // 1217
    TAG_EXEC_CONTROL,         // 1218
    TAG_NUM_PENDING_COMMS,    // 1219
    TAG_UNIFIED_MEM           // 1220
};

//Because of DEV2DEV OPIDs <=0 are RESERVED, and OPIDs > OPID_DEVTODEV too
enum {
    OPID_INVALID=0,
    OPID_FINISH=1,
    OPID_COPYIN = 2,
    OPID_COPYOUT=3,
    OPID_FREE = 4,
    OPID_ALLOCATE =5,
    OPID_COPYLOCAL = 6,
    OPID_REALLOC = 7,
    OPID_CONTROL = 8,
    OPID_CREATEAUXTHREAD=9,
    OPID_UNIFIED_MEM_REQ=10,
    OPID_TASK_INIT=11, /*Keep DEV2DEV value as highest in the OPIDs*/
    OPID_DEVTODEV=999
};
//Assigned rank value for the Daemon Thread, so it doesn't get used by any DD
#define CACHETHREADRANK -1
#define TASK_END_PROCESS -1

//When source or destination comes with this value, it means that the user
//didn't specify any concrete device, runtime launchs in whatever it wants
//so we have to override it's value with the PE value
//WARNING: Keep this defines with the same value than the one existing in the compiler (nanox-mpi.hpp)
#define UNKNOWN_RANK -95
#define MASK_TASK_NUMBER 989

#endif // COMMAND_ID_HPP
