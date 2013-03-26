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
#include <iostream>
#include <fstream>
using namespace nanos;
using namespace nanos::ext;

System::CachePolicyType MPIProcessor::_cachePolicy = System::WRITE_THROUGH;
size_t MPIProcessor::_cacheDefaultSize = 10485800;
size_t MPIProcessor::_bufferDefaultSize = 0;
char* MPIProcessor::_bufferPtr = 0;
std::string MPIProcessor::_mpiFilename;
std::string MPIProcessor::_mpiExecFile;
std::string MPIProcessor::_mpiLauncherFile=NANOX_PREFIX"/bin/ompss_mpi_launch.sh";
std::string MPIProcessor::_mpiHosts;
std::string MPIProcessor::_mpiHostsFile;
unsigned int* MPIProcessor::_mpiFileHashname;
unsigned int* MPIProcessor::_mpiFileSize;
int MPIProcessor::_mpiFileArrSize;

MPIProcessor::MPIProcessor(int id, MPI_Comm communicator, int rank) : CachedAccelerator<MPIDevice>(id, &MPI) {
    _communicator = communicator;
    _rank = rank;
}

void MPIProcessor::prepareConfig(Config &config) {

    config.registerConfigOption("mpi-exec", NEW Config::StringVar(_mpiExecFile), "Defines executable path (in child nodes) used in DEEP_Booster_Alloc");
    config.registerArgOption("mpi-exec", "mpi-exec");
    config.registerEnvOption("mpi-exec", "NX_MPIEXEC");
    
    config.registerConfigOption("mpi-launcher", NEW Config::StringVar(_mpiLauncherFile), "Defines launcher script path (in child nodes) used in DEEP_Booster_Alloc");
    config.registerArgOption("mpi-launcher", "mpi-launcher");
    config.registerEnvOption("mpi-launcher", "NX_MPILAUNCHER");
    

    config.registerConfigOption("mpihostfile", NEW Config::StringVar(_mpiHostsFile), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\nThe format of the file is: One host per line with blank lines and lines beginning with # ignored\nMultiple processes per host can be specified by specifying the host name as follows: hostA:n\nEnvironment variables for the host can be specified separated by comma using hostA:n>env_vars or hostA>env_vars");
    config.registerArgOption("mpihostfile", "mpihostfile");
    config.registerEnvOption("mpihostfile", "NX_MPIHOSTFILE");

    config.registerConfigOption("mpihosts", NEW Config::StringVar(_mpiHosts), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\n Same format than NX_MPIHOSTFILE but in a single line and separated with \';\'\nExample: hostZ hostA>env_vars hostB:2>env_vars hostC:3 hostD:4");
    config.registerArgOption("mpihosts", "mpihosts");
    config.registerEnvOption("mpihosts", "NX_MPIHOSTS");


    // Set the cache policy for MPI devices
    System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig(_cachePolicy);
    cachePolicyCfg->addOption("wt", System::WRITE_THROUGH);
    cachePolicyCfg->addOption("wb", System::WRITE_BACK);
    cachePolicyCfg->addOption("nocache", System::NONE);
    config.registerConfigOption("mpi-cache-policy", cachePolicyCfg, "Defines the cache policy for MPI architectures: write-through / write-back (wb by default)");
    config.registerEnvOption("mpi-cache-policy", "NX_MPI_CACHE_POLICY");
    config.registerArgOption("mpi-cache-policy", "mpi-cache-policy");


    config.registerConfigOption("mpi-cache-size", NEW Config::SizeVar(_cacheDefaultSize), "Defines size of the cache for MPI allocated devices");
    config.registerArgOption("mpi-cache-size", "mpi-cache-size");
    config.registerEnvOption("mpi-cache-size", "NX_MPICACHESIZE");


    config.registerConfigOption("mpi-buffer-size", NEW Config::SizeVar(_bufferDefaultSize), "Defines size of the nanox MPI Buffer (MPI_Buffer_Attach/detach)");
    config.registerArgOption("mpi-buffer-size", "mpi-buffer-size");
    config.registerEnvOption("mpi-buffer-size", "NX_MPIBUFFERSIZE");
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

void MPIProcessor::setMpiExename(char* new_name) {
    std::string tmp = std::string(new_name);
    _mpiFilename = tmp;
}

std::string MPIProcessor::getMpiExename() {
    return _mpiFilename;
}

void MPIProcessor::DEEP_Booster_free(MPI_Comm *intercomm, int rank) {
    cacheOrder order;
    order.opId = -1;
    int id = -1; 
    if (rank == -1) {
        int size;
        MPI_Comm_remote_size(*intercomm, &size);
        for (int i = 0; i < size; i++) {
            //Closing cache daemon and user-level daemon
            nanos_MPI_Send(&order, 1, nanos::MPIDevice::cacheStruct, i, TAG_CACHE_ORDER, *intercomm);
            nanos_MPI_Send(&id, 1, MPI_INT, i, TAG_INI_TASK, *intercomm);
        }
    } else {
        nanos_MPI_Send(&order, 1, nanos::MPIDevice::cacheStruct, rank, TAG_CACHE_ORDER, *intercomm);
        nanos_MPI_Send(&id, 1, MPI_INT, rank, TAG_INI_TASK, *intercomm);
    }
}

/**
 * All this tasks redefine nanox messages
 */
void MPIProcessor::nanos_MPI_Init(int *argc, char ***argv) {
    int provided, claimed;
    //TODO: Try with multiple MPI thread
    int initialized;
    MPI_Initialized(&initialized);
    //In case it was already initialized (shouldn't happen, since we theorically "rename" the calls with mercurium), we won't try to do so
    //We'll trust user criteria, but show a warning
    if (!initialized)
        MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    else
        warning0("MPI was already initialized, please, call to nanos_MPI_Init routine instead of default MPI_routines");
    
    //If not initiliazed with the paralelism level we need/implementation doesn't provide it, error.
    MPI_Query_thread(&claimed);
    fatal_cond0(claimed < MPI_THREAD_MULTIPLE,"MPI_Query_Thread returned multithread support less than MPI_THREAD_MULTIPLE, check your MPI "
            "implementation and try to configure it so it can support this multithread level");
    if (_bufferDefaultSize != 0 && _bufferPtr != 0) {
        _bufferPtr = new char[_bufferDefaultSize];
        MPI_Buffer_attach(_bufferPtr, _bufferDefaultSize);
    }
    nanos::MPIDevice::initMPICacheStruct();
        
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);
    //If this process was not spawned, we don't need this daemon-thread
    if (parentcomm != NULL && parentcomm != MPI_COMM_NULL) {
         //In this case we are child, when nanox spawns us, it fills both args
        if (argc!=0)
           setMpiExename((*argv)[(*argc)-2]); //This should not be needed
        
        //Initialice MPI PE with a communicator and special rank for the cache thread
        PE *mpi = NEW nanos::ext::MPIProcessor(999, MPI_COMM_WORLD, CACHETHREADRANK);
        MPIDD * dd = NEW MPIDD((MPIDD::work_fct) nanos::MPIDevice::mpiCacheWorker);
        WD *wd = NEW WD(dd);
        mpi->startThread(*wd);
    }
}

void MPIProcessor::nanos_MPI_Finalize() {    
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
      MPI_Finalize();
    }
}

static inline void trim(std::string& params){
    //Trim params
    size_t pos = params.find_last_not_of(" \t");
    if( std::string::npos != pos ) params = params.substr( 0, pos+1 );
    pos = params.find_first_not_of(" \t");
    if( std::string::npos != pos ) params = params.substr( pos );
}

void MPIProcessor::DEEP_Booster_alloc(MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm, int offset) {  
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
        fatal_cond0(infile.bad(),"DEEP_Booster alloc error, NX_MPIHOSTFILE file not found");
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
        //If we have _mpiFilename, we are a child, so we use master's executable name
        if (!_mpiFilename.empty()){
            result_tmp=_mpiFilename;
            count=_mpiFilename.size();
        }
        fatal_cond0(count==0,"Couldn't identify executable filename, please specify it manually using NX_MPIEXEC environment variable");  
        result_str=result_tmp.substr(0,count);    
    }
    //Number of spawns = max length (one instance per host)
    char *array_of_commands[number_of_spawns];
    char **array_of_argv[number_of_spawns];
    MPI_Info  array_of_info[number_of_spawns];
    int n_process[number_of_spawns];
    int host_counter=0;
    
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
        MPI_Info_set(info, const_cast<char*> ("host"), const_cast<char*> (host.c_str()));
        array_of_info[spawn_arrays_length]=info;
        
        
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
        argvv[1]= TAG_MAIN_OMPSS;    
        int param_counter=2;
        while (getline(all_param, tmp_param, ',')) {            
            //Trim current param
            trim(params);
            char* arg_copy=new char[tmp_param.size()+1];
            strcpy(arg_copy,tmp_param.c_str());
            argvv[param_counter]=arg_copy;
            param_counter++;
        }
        argvv[params_size-1]=NULL;              
        array_of_argv[spawn_arrays_length]=argvv;     
        
        array_of_commands[spawn_arrays_length]=const_cast<char*> (_mpiLauncherFile.c_str());      
        
        //Set number of instances this host can handle
        int curr_host_instances=host_instances.at(host_counter-1);
        int remaning_spawns=(number_of_spawns-i);
        n_process[spawn_arrays_length]=(curr_host_instances<remaning_spawns)?curr_host_instances:remaning_spawns;//min(host_instances,remaning_spawns)
        i+=n_process[spawn_arrays_length]; 
        spawn_arrays_length++;
    }   

    MPI_Comm_spawn_multiple(spawn_arrays_length,array_of_commands, array_of_argv, n_process,
            array_of_info, 0, comm, intercomm,
            MPI_ERRCODES_IGNORE);
    //Free all args sent
    for (i=0;i<spawn_arrays_length;i++){  
        //Free all args which were dynamically copied before
        for (int e=2;array_of_argv[i][e]!=NULL;e++){
            delete[] array_of_argv[i][e];
        }
        delete[] array_of_argv[i];
    }
    //Register spawned processes so nanox can use them
    sys.DEEP_Booster_register_spawns(number_of_spawns,intercomm);
    //Now they are spawned, send source ordering array so both master and workers have function pointers at the same position
    for ( int rank=0; rank<number_of_spawns; rank++ ){
        nanos_MPI_Send(_mpiFileHashname, _mpiFileArrSize, MPI_UNSIGNED, rank, TAG_FP_NAME_SYNC, *intercomm);
        nanos_MPI_Send(_mpiFileSize, _mpiFileArrSize, MPI_UNSIGNED, rank, TAG_FP_SIZE_SYNC, *intercomm);
    }
}

int MPIProcessor::nanos_MPI_Send_taskinit(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanos_MPI_Send(buf, count, datatype, dest, TAG_INI_TASK, comm);
}

int MPIProcessor::nanos_MPI_Recv_taskinit(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanos_MPI_Recv(buf, count, datatype, source, TAG_INI_TASK, comm, status);
}

int MPIProcessor::nanos_MPI_Send_taskend(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanos_MPI_Send(buf, count, datatype, dest, TAG_END_TASK, comm);
}

int MPIProcessor::nanos_MPI_Recv_taskend(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanos_MPI_Recv(buf, count, datatype, source, TAG_END_TASK, comm, status);
}

int MPIProcessor::nanos_MPI_Send_datastruct(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanos_MPI_Send(buf, count, datatype, dest, TAG_ENV_STRUCT, comm);
}

int MPIProcessor::nanos_MPI_Recv_datastruct(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanos_MPI_Recv(buf, count, datatype, source, TAG_ENV_STRUCT, comm, status);
}

int MPIProcessor::nanos_MPI_Type_create_struct( int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], 
        MPI_Datatype array_of_types[], MPI_Datatype *newtype) {
    int err=MPI_Type_create_struct(count,array_of_blocklengths,array_of_displacements, array_of_types,newtype );
    MPI_Type_commit(newtype);
    return err;
}

int MPIProcessor::nanos_MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("Envio con tag %d, a %d\n",tag,dest);
    int err = MPI_Send(buf, count, datatype, dest, tag, comm);
    //printf("Fin Envio con tag %d, a %d\n",tag,dest);
    return err;
}

int MPIProcessor::nanos_MPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("Enviobloq con tag %d, a %d\n",tag,dest);
    int err = MPI_Ssend(buf, count, datatype, dest, tag, comm);
    //printf("Fin Enviobloq con tag %d, a %d\n",tag,dest);
    return err;
}

int MPIProcessor::nanos_MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
        MPI_Comm comm, MPI_Status *status) {
    if (source==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        source=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("recv con tag %d, desde %d\n",tag,source);
    int err = MPI_Recv(buf, count, datatype, source, tag, comm, status);
    //printf("Fin recv con tag %d, desde %d\n",tag,source);
    return err;
}