/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "mpiprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include "mpithread.hpp"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace nanos;
using namespace nanos::ext;

extern __attribute__((weak)) int ompss_mpi_masks[1U];
extern __attribute__((weak)) unsigned int ompss_mpi_filenames[1U];
extern __attribute__((weak)) unsigned int ompss_mpi_file_sizes[1U];
extern __attribute__((weak)) unsigned int ompss_mpi_file_ntasks[1U];
extern __attribute__((weak)) void *ompss_mpi_func_pointers_host[1];
extern __attribute__((weak)) void (*ompss_mpi_func_pointers_dev[1])();

MPIProcessor::MPIProcessor(int id, void* communicator, int rank, int uid, bool owner, bool shared) : _pendingReqs(), CachedAccelerator<MPIDevice>(id, &MPI, uid) {
    _communicator = *((MPI_Comm *)communicator);
    _rank = rank;
    _owner=owner;
    _shared=shared;
    _currExecutingWd=NULL;
    _busy=false;
    _currExecutingDD=0;
    _hasWorkerThread=false;
    configureCache(MPIProcessor::getCacheDefaultSize(), MPIProcessor::getCachePolicy());
}

void MPIProcessor::prepareConfig(Config &config) {

    config.registerConfigOption("offl-exec", NEW Config::StringVar(_mpiExecFile), "Defines executable path (in child nodes) used in DEEP_Booster_Alloc");
    config.registerArgOption("offl-exec", "offl-exec");
    config.registerEnvOption("offl-exec", "NX_OFFL_EXEC");
    
    config.registerConfigOption("offl-launcher", NEW Config::StringVar(_mpiLauncherFile), "Defines launcher script path (in child nodes) used in DEEP_Booster_Alloc");
    config.registerArgOption("offl-launcher", "offl-launcher");
    config.registerEnvOption("offl-launcher", "NX_OFFL_LAUNCHER");
    

    config.registerConfigOption("offl-hostfile", NEW Config::StringVar(_mpiHostsFile), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\nThe format of the file is: "
    "One host per line with blank lines and lines beginning with # ignored\n"
    "Multiple processes per host can be specified by specifying the host name as follows: hostA:n\n"
    "Environment variables for the host can be specified separated by comma using hostA:n>env_var1,envar2... or hostA>env_var1,envar2...");
    config.registerArgOption("offl-hostfile", "offl-hostfile");
    config.registerEnvOption("offl-hostfile", "NX_OFFL_HOSTFILE");

    config.registerConfigOption("offl-hosts", NEW Config::StringVar(_mpiHosts), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\n Same format than NX_OFFLHOSTFILE but in a single line and separated with \';\'\nExample: hostZ hostA>env_vars hostB:2>env_vars hostC:3 hostD:4");
    config.registerArgOption("offl-hosts", "offl-hosts");
    config.registerEnvOption("offl-hosts", "NX_OFFL_HOSTS");


    // Set the cache policy for MPI devices
    System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig(_cachePolicy);
    cachePolicyCfg->addOption("wt", System::WRITE_THROUGH);
    cachePolicyCfg->addOption("wb", System::WRITE_BACK);
    cachePolicyCfg->addOption("nocache", System::NONE);
    config.registerConfigOption("offl-cache-policy", cachePolicyCfg, "Defines the cache policy for offload architectures: write-through / write-back (wb by default)");
    config.registerEnvOption("offl-cache-policy", "NX_OFFL_CACHE_POLICY");
    config.registerArgOption("offl-cache-policy", "offl-cache-policy");


//    config.registerConfigOption("offl-cache-size", NEW Config::SizeVar(_cacheDefaultSize), "Defines maximum possible size of the cache for MPI allocated devices (Default: 144115188075855871)");
//    config.registerArgOption("offl-cache-size", "offl-cache-size");
//    config.registerEnvOption("offl-cache-size", "NX_OFFLCACHESIZE");


//    config.registerConfigOption("offl-buffer-size", NEW Config::SizeVar(_bufferDefaultSize), "Defines size of the nanox MPI Buffer (MPI_Buffer_Attach/detach)");
//    config.registerArgOption("offl-buffer-size", "offl-buffer-size");
//    config.registerEnvOption("offl-buffer-size", "NX_OFFLBUFFERSIZE");
    
    config.registerConfigOption("offl-align-threshold", NEW Config::SizeVar(_alignThreshold), "Defines minimum size (bytes) which determines if offloaded variables (copy_in/out) will be aligned (default value: 128), arrays with size bigger or equal than this value will be aligned when offloaded");
    config.registerArgOption("offl-align-threshold", "offl-align-threshold");
    config.registerEnvOption("offl-align-threshold", "NX_OFFL_ALIGNTHRESHOLD");
    
    config.registerConfigOption("offl-alignment", NEW Config::SizeVar(_alignment), "Defines the alignment (bytes) applied to offloaded variables (copy_in/out) (default value: 4096)");
    config.registerArgOption("offl-alignment", "offl-alignment");
    config.registerEnvOption("offl-alignment", "NX_OFFL_ALIGNMENT");
        
    config.registerConfigOption("offl-workers", NEW Config::SizeVar(_maxWorkers), "Defines the maximum number of worker threads created per alloc (Default: 1) ");
    config.registerArgOption("offl-workers", "offl-max-workers");
    config.registerEnvOption("offl-workers", "NX_OFFL_MAX_WORKERS");
    
    config.registerConfigOption("offl-cache-threads", NEW Config::BoolVar(_useMultiThread), "Defines if offload processes will have an extra cache thread,"
        " this is good for applications which need data from other tasks so they don't have to wait until task in owner node finishes. "
        "(Default: False, but if this kind of behaviour is detected, the thread will be created anyways)");
    config.registerArgOption("offl-cache-threads", "offl-cache-threads");
    config.registerEnvOption("offl-cache-threads", "NX_OFFL_CACHE_THREADS");
}

WorkDescriptor & MPIProcessor::getWorkerWD() const {
    MPIDD * dd = NEW MPIDD((MPIDD::work_fct)Scheduler::workerLoop);
    WD *wd = NEW WD(dd);
    return *wd;
}

WorkDescriptor & MPIProcessor::getMasterWD() const {
    WD * wd = NEW WD(NEW MPIDD());
    return *wd;
}

BaseThread &MPIProcessor::createThread(WorkDescriptor &helper) {
    MPIThread &th = *NEW MPIThread(helper, this);

    return th;
}

void MPIProcessor::clearAllRequests() {
    //Wait and clear for all the requests
    if (!_pendingReqs.empty()) {
        std::vector<MPI_Request> nodeVector(_pendingReqs.begin(), _pendingReqs.end());
        MPI_Waitall(_pendingReqs.size(),&nodeVector[0],MPI_STATUSES_IGNORE);
        _pendingReqs.clear();
    }
}


bool MPIProcessor::executeTask(int taskId) {    
    bool ret=false;
    if (taskId==TASK_END_PROCESS){
       nanosMPIFinalize(); 
       ret=true;
    } else {                     
       void (* function_pointer)()=(void (*)()) ompss_mpi_func_pointers_dev[taskId]; 
       //nanos::MPIDevice::taskPreInit();
       function_pointer();       
       //nanos::MPIDevice::taskPostFinish();
    }    
    return ret;
}

int MPIProcessor::getNextPEId() {
    if (_numPrevPEs==0)  {
        return -1;
    }
    if (_numPrevPEs==-1){
        _numPrevPEs=sys.getNumCreatedPEs();
        _numFreeCores=sys.getCpuCount()-_numPrevPEs;
        _currPE=0;
        if (_numFreeCores<=0){
            _numPrevPEs=0;
            _numFreeCores=sys.getCpuCount();
            _currPE=sys.getNumCreatedPEs();
            return -1;
        }
    }
    return (_currPE++%_numFreeCores)+_numPrevPEs;
}


int MPIProcessor::nanosMPIWorker(){
	bool finalize=false;
	//Acquire once
   getTaskLock().acquire();
	while(!finalize){
		//Acquire twice and block until cache thread unlocks
      testTaskQueueSizeAndLock();
      setCurrentTaskParent(getQueueCurrentTaskParent());
		finalize=executeTask(getQueueCurrTaskIdentifier());
      removeTaskFromQueue();
	}     
   //Release the lock so cache thread can finish
	getTaskLock().release();
   return 0;
}

/**
 * Statics (mostly external API adapters provided to user or used by mercurium) begin here
 */


void MPIProcessor::DEEP_Booster_free(MPI_Comm *intercomm, int rank) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_DEEP_BOOSTER_FREE_EVENT);
    cacheOrder order;
    order.opId = OPID_FINISH;
    int nThreads=sys.getNumWorkers();
    int id[2];
    id[0] = -1; 
    //Now sleep the threads which represent the remote processes
    int res=MPI_IDENT;
    bool spawnedWithCommWorld=false;
    
    std::vector<nanos::ext::MPIThread*> threadsToDespawn; 
    //Find threads to de-spawn
    for (int i=0; i< nThreads; ++i){
        BaseThread* bt=sys.getWorker(i);
        nanos::ext::MPIProcessor * myPE = dynamic_cast<nanos::ext::MPIProcessor *>(bt->runningOn());
        if (myPE && !bt->isSleeping() && (myPE->getRank()==rank || rank == -1)){
            MPI_Comm threadcomm=myPE->getCommunicator();
            if (threadcomm!=0 && intercomm!=NULL) MPI_Comm_compare(threadcomm,*intercomm,&res);
            if (res==MPI_IDENT){ 
                spawnedWithCommWorld= spawnedWithCommWorld || myPE->getShared();
                threadsToDespawn.push_back((nanos::ext::MPIThread *)bt);
            }
        }
    }
    //Synchronize before killing shared resources
    if (spawnedWithCommWorld) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    for (std::vector<nanos::ext::MPIThread*>::iterator itThread = threadsToDespawn.begin(); itThread!=threadsToDespawn.end() ; ++itThread) {
        nanos::ext::MPIThread* mpiThread = *itThread;
        std::vector<MPIProcessor*>& myPEs = mpiThread->getRunningPEs();
        for (std::vector<MPIProcessor*>::iterator it = myPEs.begin(); it!=myPEs.end() ; ++it) {
                //Only owner will send kill signal to the worker
                if ( (*it)->getOwner() ) 
                {
                    nanosMPISsend(&order, 1, nanos::MPIDevice::cacheStruct, (*it)->getRank(), TAG_CACHE_ORDER, *intercomm);
                    //nanosMPISsend(id, 2, MPI_INT, (*it)->getRank(), TAG_INI_TASK, *intercomm);
                    //After sending finalization signals, we are not the owners anymore
                    //This way we prevent finalizing them multiple times if more than one thread uses them
                    (*it)->setOwner(false);
                }
        }
        if (rank==-1){                    
            mpiThread->lock();
            mpiThread->sleep();
            mpiThread->unlock();
        }
    }
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

/**
 * All this tasks redefine nanox messages
 */
void MPIProcessor::nanosMPIInit(int *argc, char ***argv, int userRequired, int* userProvided) {    
    if (_inicialized) return;
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_INIT_EVENT);
    verbose0( "loading MPI support" );
    //Unless user specifies otherwise, enable blocking mode in MPI
    if (getenv("I_MPI_WAIT_MODE")==NULL) putenv("I_MPI_WAIT_MODE=1");

    if ( !sys.loadPlugin( "pe-mpi" ) )
      fatal0 ( "Couldn't load MPI support" );
   
    _inicialized=true;   
    int provided;
    //If user provided a null pointer, we'll a value for internal checks
    if (userProvided==NULL) userProvided=&provided;
    //TODO: Try with multiple MPI thread
    int initialized;    
    MPI_Initialized(&initialized);
    //In case it was already initialized (shouldn't happen, since we theorically "rename" the calls with mercurium), we won't try to do so
    //We'll trust user criteria, but show a warning
    if (!initialized) {
        if (userRequired != MPI_THREAD_MULTIPLE) {
            warning0("Initializing MPI with MPI_THREAD_MULTIPLE instead of user required mode, this is a requeriment for OmpSs offload");
        }
        MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, userProvided);
    } else {
        //Do not initialise, but check thread level and return the right provided value to the user
        MPI_Query_thread(userProvided);        
    }
    
    fatal_cond0( (*userProvided) < MPI_THREAD_MULTIPLE,"MPI_Query_Thread returned multithread support less than MPI_THREAD_MULTIPLE, check your MPI "
            "implementation and try to configure it so it can support this multithread level");
    if (_bufferDefaultSize != 0 && _bufferPtr != 0) {
        _bufferPtr = new char[_bufferDefaultSize];
        MPI_Buffer_attach(_bufferPtr, _bufferDefaultSize);
    }
    nanos::MPIDevice::initMPICacheStruct();
    nanos::MPIDevice::setMasterDirectory(sys.getMainDirectory());
        
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

void MPIProcessor::nanosMPIFinalize() {    
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_FINALIZE_EVENT);
    if (_bufferDefaultSize != 0 && _bufferPtr != 0) {
        int size;
        void *ptr;
        MPI_Buffer_detach(&ptr, &size);
        if (ptr != _bufferPtr) {
            warning("Another MPI Buffer was attached instead of the one defined with"
                    " nanox mpi buffer size, not releasing it, user should do it manually");
            MPI_Buffer_attach(ptr, size);
        } else {
            MPI_Buffer_detach(&ptr, &size);
        }
        delete[] _bufferPtr;
    }
    int resul;
    MPI_Finalized(&resul);
    if (!resul){
      //Free every node before finalizing
      DEEP_Booster_free(NULL,-1);
      MPI_Finalize();
    }
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

static inline void trim(std::string& params){
    //Trim params
    size_t pos = params.find_last_not_of(" \t");
    if( std::string::npos != pos ) params = params.substr( 0, pos+1 );
    pos = params.find_first_not_of(" \t");
    if( std::string::npos != pos ) params = params.substr( pos );
}

void MPIProcessor::DEEPBoosterAlloc(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int offset) {  
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_DEEP_BOOSTER_ALLOC_EVENT);
    //IF nanos MPI not initialized, do it
    if (!_inicialized)
        nanosMPIInit(0,0,MPI_THREAD_MULTIPLE,0);
    
    bool ignorePPH=false;
    if (process_per_host<=0){
        process_per_host=1;
        ignorePPH=true;
    }
    int number_of_spawns=number_of_hosts*process_per_host;
    std::list<std::string> tmp_storage;
    std::vector<std::string> tokens_params;
    std::vector<std::string> tokens_host;   
    std::vector<int> host_instances;     
    //In case a host has no parameters, we'll fill our structure with this one
    std::string params="ompssnoparam";
    //Store single-line env value or hostfile into vector, separated by ';' or '\n'
    if ( !_mpiHosts.empty() ){   
        std::stringstream hostInput(_mpiHosts);
        std::string line;
        while( getline( hostInput, line , ';') ){            
            if (offset>0) offset--;
            else tmp_storage.push_back(line);
        }
    } else if ( !_mpiHostsFile.empty() ){
        std::ifstream infile(_mpiHostsFile.c_str());
        fatal_cond0(infile.bad(),"DEEP_Booster alloc error, NX_OFFLHOSTFILE file not found");
        std::string line;
        while( getline( infile, line , '\n') ){            
            if (offset>0) offset--;
            else tmp_storage.push_back(line);
        }
        infile.close();
    }
    
    while( !tmp_storage.empty() )
    {
        std::string line=tmp_storage.front();
        tmp_storage.pop_front();
        //If not commented add it to hosts
        if (!line.empty() && line.find("#")!=0){
            size_t pos_sep=line.find(":");
            size_t pos_end=line.find("<");
            if (pos_end==line.npos) {
                pos_end=line.size();
            } else {
                params=line.substr(pos_end+1,line.size());                
                trim(params);
            }
            if (pos_sep!=line.npos){
                std::string real_host=line.substr(0,pos_sep);
                int number=atoi(line.substr(pos_sep+1,pos_end).c_str());            
                trim(real_host);
                host_instances.push_back(number);
                tokens_host.push_back(real_host); 
                tokens_params.push_back(params);
            } else {
                std::string real_host=line.substr(0,pos_end);           
                trim(real_host);  
                host_instances.push_back(1);                
                tokens_host.push_back(real_host); 
                tokens_params.push_back(params);
            }
        }
    }
    
    //If there are no hosts, that means user "wants" to spawn in localhost
    if (tokens_host.empty()){
        tokens_host.push_back("localhost");
        tokens_params.push_back(params);
        host_instances.push_back(1);              
    }
    
    // Spawn the remote process using previously parsed parameters  
    std::string result_str;
    if ( !_mpiExecFile.empty() ){   
        result_str=_mpiExecFile;
    } else {
        char result[ PATH_MAX ];
        ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );  
        std::string result_tmp(result);
        fatal_cond0(count==0,"Couldn't identify executable filename, please specify it manually using NX_OFFL_EXEC environment variable");  
        result_str=result_tmp.substr(0,count);    
    }
    //Number of spawns = max length (one instance per host)
    char *array_of_commands[number_of_spawns];
    char **array_of_argv[number_of_spawns];
    MPI_Info  array_of_info[number_of_spawns];
    int n_process[number_of_spawns];
    unsigned int host_counter=0;
    
    //This the real length of previously declared arrays, it will be equal to number_of_spawns when 
    //hostfile/line only has one instance per host (aka no host:nInstances)
    int spawn_arrays_length=0;
    int i=0;
    //Build comm_spawn structures, iterate as many times as needed to spawn number_of_spawns instances of processes
    while( i<number_of_spawns ){
        //Fill host
        MPI_Info info;
        MPI_Info_create(&info);
        std::string host;
        do {
            if (host_counter>=tokens_host.size()) host_counter=0;
            host=tokens_host.at(host_counter);
            host_counter++;
        } while (host.empty());    
        //If host is a file, give it to Intel, otherwise put the host in the spawn
        std::ifstream hostfile(host.c_str());
        bool isfile=hostfile;
        int number_of_lines_in_file=0;
        if (isfile){     
            std::string line;
            while (std::getline(hostfile, line)) {
                ++number_of_lines_in_file;
            }
            
            MPI_Info_set(info, const_cast<char*> ("hostfile"), const_cast<char*> (host.c_str()));
        } else {            
            MPI_Info_set(info, const_cast<char*> ("host"), const_cast<char*> (host.c_str()));
        }
        array_of_info[spawn_arrays_length]=info;
        hostfile.close();
        
        //Fill parameter array (including env vars)
        std::stringstream all_param_tmp(tokens_params.at(host_counter-1));
        std::string tmp_param;            
        int params_size=3;
        while (getline(all_param_tmp, tmp_param, ',')) {
            params_size++;
        }
        std::stringstream all_param(tokens_params.at(host_counter-1));
        char **argvv=new char*[params_size];
        //Fill the params
        argvv[0]= const_cast<char*> (result_str.c_str());
        argvv[1]= const_cast<char*> ("empty");  
        int param_counter=2;
        while (getline(all_param, tmp_param, ',')) {            
            //Trim current param
            trim(params);
            char* arg_copy=new char[tmp_param.size()+1];
            strcpy(arg_copy,tmp_param.c_str());
            argvv[param_counter++]=arg_copy;
        }
        argvv[params_size-1]=NULL;              
        array_of_argv[spawn_arrays_length]=argvv;     
        
        array_of_commands[spawn_arrays_length]=const_cast<char*> (_mpiLauncherFile.c_str());      
        
        //Set number of instances this host can handle
        //If user specified PPH <= 0, we'll use number of processes per node in hostfile
        int curr_host_instances;
        if (ignorePPH){
            curr_host_instances=host_instances.at(host_counter-1);
        } else {
            curr_host_instances=process_per_host;
            if (isfile) {
                curr_host_instances=number_of_lines_in_file*process_per_host;
            }
        }
        int remaning_spawns=(number_of_spawns-i);
        n_process[spawn_arrays_length]=(curr_host_instances<remaning_spawns)?curr_host_instances:remaning_spawns;//min(host_instances,remaning_spawns)
        i+=n_process[spawn_arrays_length]; 
        spawn_arrays_length++;
    }   
    int res=MPI_UNEQUAL;
    MPI_Comm_compare(comm,MPI_COMM_WORLD,&res);    
    //Register spawned processes so nanox can use them
    int number_of_spawns_this_process=number_of_spawns;
    int spawn_start=0;
    bool shared=false;
    //If MPI_COMM_WORLD, split total spawns between nodes in order to balance syncs
    int mpi_size;
    if (res!=MPI_UNEQUAL){
        int rank;
        shared=true;
        MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        number_of_spawns_this_process=number_of_spawns/mpi_size;
        spawn_start=rank*number_of_spawns_this_process;
        if (rank==mpi_size-1) //Last process syncs the remaining processes
            number_of_spawns_this_process+=number_of_spawns%mpi_size;
    } else {
        mpi_size=1; //Using MPI_COMM_SELF
    }
    int fd=-1;
    while (!shared && fd==-1) {
       fd=tryGetLock("./lockSpawn");
    }
    MPI_Comm_spawn_multiple(spawn_arrays_length,array_of_commands, array_of_argv, n_process,
            array_of_info, 0, comm, intercomm, MPI_ERRCODES_IGNORE); 
    if (!shared) {
      releaseLock(fd,"./lockSpawn"); 
    }
    //Free all args sent
    for (i=0;i<spawn_arrays_length;i++){  
        //Free all args which were dynamically copied before
        for (int e=2;array_of_argv[i][e]!=NULL;e++){
            delete[] array_of_argv[i][e];
        }
        delete[] array_of_argv[i];
    }
    PE* pes[number_of_spawns];
    int uid=sys.getNumCreatedPEs();
    int arrSize;
    for (arrSize=0;ompss_mpi_masks[arrSize]==MASK_TASK_NUMBER;arrSize++){};
    int rank=spawn_start; //Balance spawn order so each process starts with his owned processes
    int bindingId=getNextPEId(); //All the PEs share the same local bind ID (at the end they'll be executed by the same thread)
    if (bindingId!=-1){ //If no free PE, bind to every core
        bindingId=sys.getBindingId(bindingId);
    }
    //Now they are spawned, send source ordering array so both master and workers have function pointers at the same position
    for ( int rankCounter=0; rankCounter<number_of_spawns; rankCounter++ ){  
        //Each process will have access to every remote node, but only one master will sync each child
        //this way we balance syncs with childs
        if (rank>=spawn_start && rank<spawn_start+number_of_spawns_this_process) {
            pes[rank]=NEW nanos::ext::MPIProcessor( bindingId ,intercomm, rank,uid++, true, shared);
            nanosMPISend(ompss_mpi_filenames, arrSize, MPI_UNSIGNED, rank, TAG_FP_NAME_SYNC, *intercomm);
            nanosMPISend(ompss_mpi_file_sizes, arrSize, MPI_UNSIGNED, rank, TAG_FP_SIZE_SYNC, *intercomm);
            //If user defined multithread cache behaviour, send the creation order
            if (_useMultiThread) {                
                cacheOrder order;
                //if PE is busy, this means an extra cache-thread could be usefull, send creation signal
                order.opId = OPID_CREATEAUXTHREAD;
                nanos::ext::MPIProcessor::nanosMPISend(&order, 1, nanos::MPIDevice::cacheStruct, rank, TAG_CACHE_ORDER, *intercomm);
                ((MPIProcessor*)pes[rank])->setHasWorkerThread(true);
            }
        } else {            
            pes[rank]=NEW nanos::ext::MPIProcessor( bindingId ,intercomm, rank,uid++, false, shared);
        }
        rank=(rank+1)%number_of_spawns;
    }
    //Each node will have nSpawns/nNodes running, with a Maximum of 4
    //We supose that if 8 hosts spawns 16 nodes, each one will usually run 2
    //HINT: This does not mean that some remote nodes wont be accesible
    //using more than 1 thread is a performance tweak
    int number_of_threads=(number_of_spawns/mpi_size);
    if (number_of_threads<1) number_of_threads=1;
    if (number_of_threads>_maxWorkers) number_of_threads=_maxWorkers;
    BaseThread* threads[number_of_threads];
    sys.addOffloadPEsToTeam(pes, number_of_spawns, number_of_threads, threads); 
    //Add all the PEs to the thread
    Lock* gLock=NULL;
    Atomic<int>* gCounter;
    std::vector<MPIThread*>* threadList;
    for ( i=0; i<number_of_threads; i++ ){ 
        MPIThread* mpiThread=(MPIThread*) threads[i];
        //Get the lock of one of the threads
        if (gLock==NULL) {
            gLock=mpiThread->getSelfLock();
            gCounter=mpiThread->getSelfCounter();
            threadList=mpiThread->getSelfThreadList();
        }
        threadList->push_back(mpiThread);
        mpiThread->addRunningPEs((MPIProcessor**)pes,number_of_spawns);
        //Set the group lock so they all share the same lock
        mpiThread->setGroupCounter(gCounter);
        mpiThread->setGroupThreadList(threadList);
        if (number_of_threads>1) {
            //Set the group lock so they all share the same lock
            mpiThread->setGroupLock(gLock);
        }
    }
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

int MPIProcessor::nanosMPISendTaskinit(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    //Send task init order and pendingComms counter
    return nanosMPISend(buf, count, datatype, dest, TAG_INI_TASK, comm);
}

int MPIProcessor::nanosMPIRecvTaskinit(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanosMPIRecv(buf, count, datatype, source, TAG_INI_TASK, comm, status);
}

int MPIProcessor::nanosMPISendTaskend(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    //Ignore destination (as is always parent) and get currentParent
    return nanosMPISend(buf, count, datatype, nanos::ext::MPIProcessor::getCurrentTaskParent(), TAG_END_TASK, comm);
}

int MPIProcessor::nanosMPIRecvTaskend(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanosMPIRecv(buf, count, datatype, source, TAG_END_TASK, comm, status);
}

int MPIProcessor::nanosMPISendDatastruct(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanosMPISend(buf, count, datatype, dest, TAG_ENV_STRUCT, comm);
}

int MPIProcessor::nanosMPIRecvDatastruct(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    //Ignore destination (as is always parent) and get currentParent
     nanosMPIRecv(buf, count, datatype,  nanos::ext::MPIProcessor::getCurrentTaskParent(), TAG_ENV_STRUCT, comm, status);     
     return 0;
}

int MPIProcessor::nanosMPITypeCreateStruct( int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], 
        MPI_Datatype array_of_types[], MPI_Datatype *newtype) {
    int err=MPI_Type_create_struct(count,array_of_blocklengths,array_of_displacements, array_of_types,newtype );
    MPI_Type_commit(newtype);
    return err;
}

int MPIProcessor::nanosMPISend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_SEND_EVENT);
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("Envio con tag %d, a %d\n",tag,dest);
    int err = MPI_Send(buf, count, datatype, dest, tag, comm);
    //printf("Fin Envio con tag %d, a %d\n",tag,dest);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIProcessor::nanosMPIIsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm,MPI_Request *req) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_ISEND_EVENT);
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("Envio con tag %d, a %d\n",tag,dest);
    int err = MPI_Isend(buf, count, datatype, dest, tag, comm,req);
    //printf("Fin Envio con tag %d, a %d\n",tag,dest);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIProcessor::nanosMPISsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_SSEND_EVENT);
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("Enviobloq con tag %d, a %d\n",tag,dest);
    int err = MPI_Ssend(buf, count, datatype, dest, tag, comm);
    //printf("Fin Enviobloq con tag %d, a %d\n",tag,dest);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIProcessor::nanosMPIRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
        MPI_Comm comm, MPI_Status *status) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RECV_EVENT);
    if (source==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        source=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("recv con tag %d, desde %d\n",tag,source);
    int err = MPI_Recv(buf, count, datatype, source, tag, comm, status );
    //printf("Fin recv con tag %d, desde %d\n",tag,source);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIProcessor::nanosMPIIRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
        MPI_Comm comm, MPI_Request *req) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_IRECV_EVENT);
    if (source==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        source=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("recv con tag %d, desde %d\n",tag,source);
    int err = MPI_Irecv(buf, count, datatype, source, tag, comm, req );
    //printf("Fin recv con tag %d, desde %d\n",tag,source);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

/**
 * Synchronizes host and device function pointer arrays to ensure that are in the same order
 * in both files (host and device, which are different architectures, so maybe they were not compiled in the same order)
 */
void MPIProcessor::nanosSyncDevPointers(int* file_mask, unsigned int* file_namehash, unsigned int* file_size,
            unsigned int* task_per_file,void (*ompss_mpi_func_ptrs_dev[])()){
    const int mask = MASK_TASK_NUMBER;
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);   
    //If this process was not spawned, we don't need this reorder (and shouldnt have been called)
    if ( parentcomm != 0 && parentcomm != MPI_COMM_NULL ) {     
        //MPI_Status status;
        int arr_size;
        for ( arr_size=0;file_mask[arr_size]==mask;arr_size++ ){};
        unsigned int total_size=0;
        for ( int k=0;k<arr_size;k++ ) total_size+=task_per_file[k];
        size_t filled_arr_size=0;
        unsigned int* host_file_size=(unsigned int*) malloc(sizeof(unsigned int)*arr_size);
        unsigned int* host_file_namehash=(unsigned int*) malloc(sizeof(unsigned int)*arr_size);
        void (**ompss_mpi_func_pointers_dev_out)()=(void (**)()) malloc(sizeof(void (*)())*total_size);
        //Receive host information
        nanos::ext::MPIProcessor::nanosMPIRecv(host_file_namehash, arr_size, MPI_UNSIGNED, MPI_ANY_SOURCE, TAG_FP_NAME_SYNC, parentcomm, MPI_STATUS_IGNORE);
        nanos::ext::MPIProcessor::nanosMPIRecv(host_file_size, arr_size, MPI_UNSIGNED, MPI_ANY_SOURCE, TAG_FP_SIZE_SYNC, parentcomm, MPI_STATUS_IGNORE );
        int i,e,func_pointers_arr;
        bool found;
        //i loops at host files
        for ( i=0;i<arr_size;i++ ){   
            func_pointers_arr=0;
            found=false;
            //Search the host file in dev file and copy every pointer in the same order
            for ( e=0;!found && e<arr_size;e++ ){
                if( file_namehash[e] == host_file_namehash[i] && file_size[e] == host_file_size[i] ){
                    found=true; 
                    //Copy from _dev_tmp array to _dev array in the same order than the host
                    memcpy(ompss_mpi_func_pointers_dev_out+filled_arr_size,ompss_mpi_func_ptrs_dev+func_pointers_arr,task_per_file[e]*sizeof(void (*)()));
                    filled_arr_size+=task_per_file[e];  
                }
                func_pointers_arr+=task_per_file[e];
            }
            fatal_cond0(!found,"File not found in device, please compile the code using exactly the same sources (same filename and size) for each architecture");
        }
        memcpy(ompss_mpi_func_ptrs_dev,ompss_mpi_func_pointers_dev_out,total_size*sizeof(void (*)()));
        free(ompss_mpi_func_pointers_dev_out);
        free(host_file_size);
        free(host_file_namehash);
    }
}


void MPIProcessor::mpiOffloadSlaveMain(){    
    //If we are slave, turn on slave mode (which keeps working until shutdown) and exit
    if (getenv("OMPSS_OFFLOAD_SLAVE")){
       nanosMPIInit(0,0,MPI_THREAD_MULTIPLE,0);
       nanos::ext::MPIProcessor::nanosSyncDevPointers(ompss_mpi_masks, ompss_mpi_filenames, ompss_mpi_file_sizes,ompss_mpi_file_ntasks,ompss_mpi_func_pointers_dev);
       //Start as worker and cache (same thread)
       nanos::MPIDevice::mpiCacheWorker();
       exit(0);
    }    
}

int MPIProcessor::ompssMpiGetFunctionIndexHost(void* func_pointer){  
    int i;
    //This function WILL find the pointer, if it doesnt, program would crash anyways so I won't limit it
    for (i=0;ompss_mpi_func_pointers_host[i]!=func_pointer;i++);
    return i;
}