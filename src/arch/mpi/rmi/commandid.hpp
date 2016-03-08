
#ifndef COMMAND_ID_HPP
#define COMMAND_ID_HPP

// This struct is deprecated and no longer used
// However, there are some API declarations with this
// struct type declared (e.g. unifiedMemoryMallocRemote)
typedef struct {
       int opId;
       //In case of dev2dev, hostaddr= srcAddr, devAddr=remoteAddr
       uint64_t hostAddr;
       uint64_t devAddr;
       size_t size;
       //size_t old_size;
       //unsigned char* data;
} cacheOrder;

// MPI Communication tags, we use that many so messages don't collide for different operations
// TODO: maybe we need to split TAG_M2S_ORDER in two: one for Commands and the other for CacheOrders,
//       as they are treated differently. In addition, payload size is different.
enum {
    TAG_M2S_ORDER = 1200, TAG_CACHE_DATA_IN,TAG_CACHE_DATA_OUT,
    TAG_CACHE_ANSWER, TAG_INI_TASK,TAG_END_TASK, TAG_ENV_STRUCT,TAG_CACHE_ANSWER_REALLOC,
    TAG_CACHE_ANSWER_ALLOC, TAG_CACHE_ANSWER_CIN,TAG_CACHE_ANSWER_COUT,TAG_CACHE_ANSWER_FREE,TAG_CACHE_ANSWER_DEV2DEV,TAG_CACHE_ANSWER_CL,
    TAG_FP_NAME_SYNC, TAG_FP_SIZE_SYNC, TAG_CACHE_DEV2DEV, TAG_EXEC_CONTROL, TAG_NUM_PENDING_COMMS, TAG_UNIFIED_MEM
};

//Because of DEV2DEV OPIDs <=0 are RESERVED, and OPIDs > OPID_DEVTODEV too
enum {
    OPID_INVALID=0, OPID_FINISH=1, OPID_COPYIN = 2, OPID_COPYOUT=3, OPID_FREE = 4, OPID_ALLOCATE =5 , OPID_COPYLOCAL = 6, OPID_REALLOC = 7, OPID_CONTROL = 8, 
    OPID_CREATEAUXTHREAD=9, OPID_UNIFIED_MEM_REQ=10, OPID_TASK_INIT=11, /*Keep DEV2DEV value as highest in the OPIDs*/ OPID_DEVTODEV=999
};
//Assigned rank value for the Daemon Thread, so it doesn't get used by any DD
#define CACHETHREADRANK -1
#define TASK_END_PROCESS -1
//When source or destination comes with this value, it means that the user
//didn't specify any concrete device, runtime launchs in whatever it wants
//so we have to override it's value with the PE value
//WARNING: Keep this defines with the same value than the one existing in the compiler (nanox-mpi.hpp)
#define UNKOWN_RANKSRCDST -95
#define MASK_TASK_NUMBER 989

#endif // COMMAND_ID_HPP
