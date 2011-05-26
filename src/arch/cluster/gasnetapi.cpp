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

#include "gasnetapi.hpp"
#include "smpdd.hpp"

#ifdef GPU_DEV
//FIXME: GPU Support
#include "gpudd.hpp"
#endif

#include "system.hpp"
#include "os.hpp"
#include "clusterinfo.hpp"
#include "instrumentation.hpp"
#include "remoteworkgroup_decl.hpp"
#include "atomic_decl.hpp"
#include <list>

RemoteWorkGroup *rwgGPU;
RemoteWorkGroup *rwgSMP;

Directory *_masterDir;

typedef struct {
WD *wd;
unsigned int count;
} wdDeps;

Lock depsLock;
Lock sentDataLock;
std::vector<std::set<uint64_t> *> _sentData;
std::multimap<uint64_t, wdDeps *> _depsMap; //needed data
std::set<uint64_t> _recvdDeps; //already got the data


//#define HALF_PRESEND

#ifdef HALF_PRESEND
Atomic<int> wdindc = 0;
WD* buffWD = NULL;
#endif

extern "C" {
#include <gasnet.h>
}
typedef struct {
   void (*outline) (void *);
} nanos_smp_args_t;

#ifdef GPU_DEV
//FIXME: GPU Support
void * local_nanos_gpu_factory( void *prealloc, void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   if ( prealloc != NULL )
   {
      return ( void * )new (prealloc) ext::GPUDD( smp->outline );
   }
   else
   {
      return ( void * ) new ext::GPUDD( smp->outline );
   }
}
#endif
void * local_nanos_smp_factory( void *prealloc, void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;

   if ( prealloc != NULL )
   {
      return ( void * )new (prealloc) ext::SMPDD( smp->outline );
   }
   else 
   {
      return ( void * )new ext::SMPDD( smp->outline );
   }
}

#ifndef __SIZEOF_POINTER__
#error This compiler does not define __SIZEOF_POINTER__ :( 
#else

#if __SIZEOF_POINTER__ == 8

#define MERGE_ARG( _Hi, _Lo) (  ( uint32_t ) _Lo + ( ( ( uintptr_t ) ( ( uint32_t ) _Hi ) ) << 32 ) )
#define ARG_HI( _Arg ) ( ( uint32_t ) ( ( ( uintptr_t ) ( _Arg ) ) >> 32 ) )
#define ARG_LO( _Arg ) ( ( uint32_t ) ( ( uintptr_t ) _Arg ) )

#else

#define MERGE_ARG( _Hi, _Lo) ( ( uintptr_t ) ( _Lo ) )
#define ARG_HI( _Arg ) ( ( uint32_t ) 0 )
#define ARG_LO( _Arg ) ( ( uint32_t ) _Arg )

#endif

#endif

using namespace nanos;
using namespace ext;

extern char **environ;

#if 0
static void inspect_environ(void)
{
	int i = 0;

	fprintf(stderr, "+------------- Environ Start = %p --------------\n", environ);
	while (environ[i] != NULL)
		fprintf(stderr, "| %s\n", environ[i++]);
	fprintf(stderr, "+-------------- Environ End = %p ---------------\n", &environ[i]);
}
#endif

struct put_req_desc {
   unsigned int dest;
   void *origAddr;
   void *destAddr;
   size_t len;
};

static std::list<struct put_req_desc * > put_req_vector;
Lock put_req_vector_lock;

void enqueue_put_request( unsigned int dest, void *origAddr, void *destAddr, size_t len)
{
   struct put_req_desc *prd = NEW struct put_req_desc();
   //fprintf(stderr, "enqueue req to node %d\n", dest);
   prd->dest = dest;
   prd->origAddr = origAddr;
   prd->destAddr = destAddr;
   prd->len = len;

   put_req_vector.push_back( prd );
}

//static char *global_work_data = NULL;
//static size_t global_work_data_len;

static void am_exit(gasnet_token_t token)
{
    gasnet_node_t src_node;
    if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
    {
        fprintf(stderr, "gasnet: Error obtaining node information.\n");
    }
    //fprintf(stderr, "EXIT msg from node %d.\n", src_node);
    //gasnet_AMReplyShort0(token, 204);
    //finish_gasnet = true;
    sys.stopFirstThread();
}

static void am_exit_reply(gasnet_token_t token)
{
    gasnet_node_t src_node;
    if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
    {
        fprintf(stderr, "gasnet: Error obtaining node information.\n");
    }
    //fprintf(stderr, "EXIT message to node %d completed.\n", src_node);
}

static void am_work(gasnet_token_t token, void *arg, size_t argSize,
      gasnet_handlerarg_t workLo,
      gasnet_handlerarg_t workHi,
      gasnet_handlerarg_t xlateLo,
      gasnet_handlerarg_t xlateHi,
      gasnet_handlerarg_t rmwdLo,
      gasnet_handlerarg_t rmwdHi,
      //gasnet_handlerarg_t rmthdLo,
      //gasnet_handlerarg_t rmthdHi,
      unsigned int dataSize, unsigned int wdId, unsigned int numPe, int arch )
{
	void (*work)( void *) = (void (*)(void *)) MERGE_ARG( workHi, workLo );
	void (*xlate)( void *, void *) = (void (*)(void *, void *)) MERGE_ARG( xlateHi, xlateLo );
	void *rmwd = (void *) MERGE_ARG( rmwdHi, rmwdLo );
	//void *rmthd = (void *) MERGE_ARG( rmthdHi, rmthdLo );
	gasnet_node_t src_node;
	unsigned int i;
	//size_t realSize;
	WG *rwg;

	if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
	{
		fprintf(stderr, "gasnet: Error obtaining node information.\n");
	}
	{
		NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
			NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
			NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( wdId ) ; )
			NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_AM_WORK, id, 0, 0, src_node ); )
	}
        char *work_data = NULL;
	size_t work_data_len;

	if ( work_data == NULL )
	{
		work_data = NEW char[ argSize ];
		memcpy( work_data, arg, argSize );
		//realSize = argSize;
	}
	else
	{
		memcpy( &work_data[ work_data_len ], arg, argSize );
		//realSize = work_data_len + argSize;
	}
	//fprintf(stderr, "n:%d-%d am_work: WORK message from node %d: fct %p, argSize %d.\n", gasnet_mynode(), myThread->getId(), src_node, work, realSize);

	nanos_smp_args_t smp_args;
	smp_args.outline = (void (*)(void *)) work; //wk_wrapper; 

	WD *localWD = NULL;
	char *data = NULL;
	unsigned int numCopies = *((int *) &work_data[ dataSize ]);
	CopyData *newCopies = NULL;
	CopyData **newCopiesPtr = ( numCopies > 0 ) ? &newCopies : NULL ;
	nanos_device_t newDeviceSMP = { local_nanos_smp_factory, sizeof(SMPDD), (void *) &smp_args } ;
#ifdef GPU_DEV
	nanos_device_t newDeviceGPU = { local_nanos_gpu_factory, sizeof(GPUDD), (void *) &smp_args } ;
#endif
	nanos_device_t *devPtr = NULL;

	if (arch == 0)
	{
		//Cluster
		devPtr = &newDeviceSMP;
		rwg = rwgSMP;
		//newCopiesPtr = NULL;
	}
#ifdef GPU_DEV
	else if (arch == 1)
	{
		//FIXME: GPU support
		devPtr = &newDeviceGPU;
		//    newCopiesPtr = &newCopies;
		rwg = rwgGPU;
	}
#endif
	else
	{
		rwg = NULL;
		fprintf(stderr, "Unsupported architecture\n");
	}
	//std::cerr <<"n:" << gasnet_mynode() << " am_work, wd is " << wdId << " arch is " << arch << std::endl;

	sys.createWD( &localWD, (size_t) 1, devPtr, (size_t) dataSize, (int) ( sizeof(void *) ), (void **) &data, (WG *)rwg, (nanos_wd_props_t *) NULL, (size_t) numCopies, newCopiesPtr, xlate );

	//warg->arg = data;
	::memcpy(data, work_data, dataSize);

	//fprintf(stderr, "NUM COPIES %d addr %llx, in? %s, out? %s\n",
	//      1,
	//      recvcd->getAddress(),
	//      recvcd->isInput() ? "yes" : "no",
	//      recvcd->isOutput() ? "yes" : "no" );



	//fprintf(stderr, "NUM COPIES %d %ld %d %ld %ld\n", numCopies, realSize, dataSize, sizeof(CopyData), sizeof(size_t));
	//CopyData *newCopies = new CopyData[ numCopies ]; 

	//Directory *thisNodeMasterDirectory = myThread->getThreadWD().getDirectory( true );

	//std::cerr <<"n:" << gasnet_mynode() << " directory is " << thisNodeMasterDirectory << " this thd wd is " <<  myThread->getThreadWD().getId() << std::endl;


	unsigned int numDeps = *((int *) &work_data[ dataSize + sizeof(int) + numCopies * sizeof(CopyData) ]);
	uint64_t *depTags =  ((uint64_t *) &work_data[ dataSize + sizeof(int) + numCopies * sizeof(CopyData) +sizeof(int) ]);
	//if ( arch == 1 )
	//{
	for (i = 0; i < numCopies; i += 1) {
		new ( &newCopies[i] ) CopyData( *( (CopyData *) &work_data[ dataSize + sizeof(int) + i * sizeof( CopyData ) ] ) );

		//if (!newCopies[i].isOutput() ) 
		//{
		//DirectoryEntry *ent = _masterDir->findEntry( newCopies[i].getAddress() );
		//if (ent != NULL) 
		//{
		//   if (ent->getOwner() != NULL )
		//      if ( !ent->isInvalidated() )
		//      {
		//         char *tmpBuffer = NEW char[newCopies[i].getSize()];
		//         ::memcpy((void *) tmpBuffer, (char *) newCopies[i].getAddress(), newCopies[i].getSize()); 
		//         std::list<uint64_t> tagsToInvalidate;
		//         tagsToInvalidate.push_back(  newCopies[i].getAddress() );
		//         //std::cerr  <<"n:" << gasnet_mynode() << " sync host (tag) inv copy in" << std::endl;
		//         _masterDir->synchronizeHost( tagsToInvalidate );
		//         //std::cerr <<"n:" << gasnet_mynode() << " go on with get req " << *(( float *) newCopies[i].getAddress() )<< std::endl;
		//         ::memcpy((void *) newCopies[i].getAddress(),tmpBuffer,  newCopies[i].getSize()); 
		//         delete tmpBuffer;
		//      }
		//}
		//}


		//std::cerr <<"n:" << gasnet_mynode() << " wd id " << wdId<< " copies: " << i << " addr " << (void *) newCopies[i].getAddress() << " is only out? " << !newCopies[i].isInput() << std::endl;
		//char *tmp = new char[ newCopies[i].getSize() ];
		//fprintf(stderr, "copiant new copy %p to %p, size %ld bytes\n", (void *) newCopies[i].getAddress(), tmp, newCopies[i].getSize());
		//::memcpy(tmp, (void *) newCopies[i].getAddress(), newCopies[i].getSize() );
		//newCopies[i].setAddress( (uint64_t) tmp );
	}
	//}
	localWD->setId( wdId );
	localWD->setRemoteAddr( rmwd );
	//localWD->setRemoteThdread( rmthd );

	if ( numDeps > 0 )
	{
		wdDeps *thisWdDeps = NEW wdDeps;
		thisWdDeps->count = 0;
		thisWdDeps->wd = localWD;
		depsLock.acquire();
		//std::cerr <<"n:" << gasnet_mynode() << " I have " << numDeps << " dependence(s) for tag(s): " << std::endl;
		for (i = 0; i < numDeps; i++)
		{
			std::set<uint64_t>::iterator recvDepsIt = _recvdDeps.find( depTags[i] );
			if ( recvDepsIt == _recvdDeps.end() ) 
			{
				thisWdDeps->count += 1;
				_depsMap.insert( std::pair<uint64_t, wdDeps*> ( depTags[i], thisWdDeps ) ); 
				//std::cerr<<"n:" << gasnet_mynode() << " dep: "<< i << " tag: " << (void *)depTags[i] <<std::endl;
			}
			else
			{
				_recvdDeps.erase( recvDepsIt );
				//std::cerr<<"n:" << gasnet_mynode() << " dep: "<< i << " tag: " << (void *)depTags[i] << " FOUND!"<<std::endl;
			}
		}
		depsLock.release();
		if ( thisWdDeps->count == 0) 
		{
			{
				NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
					NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) numPe) << 32 ) + gasnet_mynode() ; )
					NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *localWD, NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
			}
			//std::cerr<<"n:" <<gasnet_mynode()<< " submitting wd with deps " << localWD->getId() <<std::endl;
#ifdef HALF_PRESEND
			if ( wdindc++ == 0 ) { sys.submit( *localWD ); /* std::cerr<<"n:" <<gasnet_mynode()<< " submitted wd " << localWD->getId() <<std::endl; */}
			else { buffWD = localWD; /* std::cerr<<"n:" <<gasnet_mynode()<< " saved wd " << buffWD->getId() <<std::endl; */}
#else
			sys.submit( *localWD );
#endif
			delete thisWdDeps;

		}
	}
	else 
	{
		wdDeps *thisWdDeps = NULL;
		depsLock.acquire();
		for (i = 0; i < numCopies; i += 1) {
			if ( _depsMap.find( newCopies[i].getAddress() ) != _depsMap.end() ) 
			{
				if ( thisWdDeps == NULL ) {
					thisWdDeps = NEW wdDeps;
					thisWdDeps->count = 1;
					thisWdDeps->wd = localWD;
				}
				else thisWdDeps->count += 1;

				//std::cerr << "WE MAY HAVE A BUG HERE!!! wd is "<<localWD->getId() << std::endl;
				_depsMap.insert( std::pair<uint64_t, wdDeps*> ( newCopies[i].getAddress(), thisWdDeps ) ); 
			}
		}
		depsLock.release();
		if ( thisWdDeps == NULL)
		{
			{
				NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
					NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) numPe) << 32 ) + gasnet_mynode() ; )
					NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *localWD, NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
			}
			//std::cerr<<"n:" <<gasnet_mynode()<< " submitting wd no deps " << localWD->getId() <<std::endl;
#ifdef HALF_PRESEND
			if ( wdindc++ == 0 ) { sys.submit( *localWD ); /* std::cerr<<"n:" <<gasnet_mynode()<< " submitted+ wd " << localWD->getId() <<std::endl;*/ }
			else { buffWD = localWD; /*std::cerr<<"n:" <<gasnet_mynode()<< " saved+ wd " << buffWD->getId() <<std::endl;*/ }
#else
			sys.submit( *localWD );
#endif
		}
	}

	delete work_data;
	work_data = NULL;
	work_data_len = 0;


	//fprintf(stderr, "n:%d out of am_work (from %d) wd %d.\n", gasnet_mynode(), src_node, wdId);
}

static void am_work_data(gasnet_token_t token, void *buff, size_t len,
      gasnet_handlerarg_t msgNum,
      gasnet_handlerarg_t totalLenLo,
      gasnet_handlerarg_t totalLenHi)
{
   gasnet_node_t src_node;
   //size_t totalLen = (size_t) MERGE_ARG( totalLenHi, totalLenLo );
   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
       fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }

   std::cerr<<"UNSUPPORTED FOR NOW"<<std::endl;
   //if ( msgNum == 0 )
   //{
   //   if (work_data != NULL)
   //      delete work_data;
   //   work_data = NEW char[ totalLen ];
   //   work_data_len = 0;
   //}
   //memcpy( &work_data[ work_data_len ], buff, len );
   //work_data_len += len;
}

static void am_work_done( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi, gasnet_handlerarg_t peId )
{
    gasnet_node_t src_node;
    void * addr = (void *) MERGE_ARG( addrHi, addrLo );
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    //fprintf(stderr, "WORK DONE msg from node %d, numPe %d.\n", src_node, numPe);
{
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_AM_WORK_DONE, id, 0, 0, src_node ); )
}
    sys.getNetwork()->notifyWorkDone( src_node, addr, peId );
}

static void am_malloc( gasnet_token_t token, gasnet_handlerarg_t size, unsigned int id )
{
    gasnet_node_t src_node;
    void *addr = NULL;
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    addr = malloc( ( size_t ) size );
    if ( gasnet_AMReplyShort3( token, 208, ( gasnet_handlerarg_t ) ARG_LO( addr ), ( gasnet_handlerarg_t ) ARG_HI( addr ), (gasnet_handlerarg_t ) id ) != GASNET_OK )
    {
       fprintf( stderr, "gasnet: Error sending a message to node %d.\n", src_node );
    }
}

/* GASNet medium active message handler */
static void am_malloc_reply( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi, unsigned int id )
{
   void * addr = (void *) MERGE_ARG( addrHi, addrLo );
    gasnet_node_t src_node;
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    sys.getNetwork()->notifyMalloc( src_node, ( void * ) addr, id );
}

static void am_memfree( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi )
{
   //void * addr = (void *) MERGE_ARG( addrHi, addrLo );
   gasnet_node_t src_node;

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   
   //std::cerr << "n:" << gasnet_mynode() << " I have to free addr " << addr << " in this node" << std::endl;
   //Directory *thisNodeMasterDirectory = myThread->getThreadWD().getDirectory( false );
   //DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) addr );
//if ( ent != NULL ) std::cerr << "n:" << gasnet_mynode() << " Got an entry for add " << addr << " owner is " << ent->getOwner() << std::endl; 
   
   //sys.memInvalidate( addr );
}
static void am_memrealloc( gasnet_token_t token, gasnet_handlerarg_t oldAddrLo, gasnet_handlerarg_t oldAddrHi, gasnet_handlerarg_t oldSizeLo, gasnet_handlerarg_t oldSizeHi, gasnet_handlerarg_t newAddrLo, gasnet_handlerarg_t newAddrHi, gasnet_handlerarg_t newSizeLo, gasnet_handlerarg_t newSizeHi)
{
   void * oldAddr = (void *) MERGE_ARG( oldAddrHi, oldAddrLo );
   void * newAddr = (void *) MERGE_ARG( newAddrHi, newAddrLo );
   size_t oldSize = (size_t) MERGE_ARG( oldSizeHi, oldSizeLo );
   //size_t newSize = (size_t) MERGE_ARG( newSizeHi, newSizeLo );
   gasnet_node_t src_node;

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   
   DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) newAddr );
   if (ent != NULL) 
   { 
	if (ent->getOwner() != NULL) 
        {
           ent->getOwner()->discard( *_masterDir, (uint64_t) newAddr, ent);
           std::cerr << "REALLOC WARNING, newAddr had an entry n:" << gasnet_mynode() << " discarding tag " << (void *) newAddr << std::endl;
        }
   }
   ::memcpy( newAddr, oldAddr, oldSize );
   
   //DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) newAddr );
   //std::cerr << "n:" << gasnet_mynode() << " I have to free addr " << addr << " in this node" << std::endl;
   //Directory *thisNodeMasterDirectory = myThread->getThreadWD().getDirectory( false );
   //DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) addr );
//if ( ent != NULL ) std::cerr << "n:" << gasnet_mynode() << " Got an entry for add " << addr << " owner is " << ent->getOwner() << std::endl; 
   
   //sys.memInvalidate( addr );
}

/* GASNet medium active message handler */
static void am_my_hostname( gasnet_token_t token, void *buff, size_t nbytes )
{
    gasnet_node_t src_node;
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    /* for now we only allow this at node 0 */
    if ( src_node == 0 )
    {
       sys.getNetwork()->setMasterHostname( ( char  *) buff );
    }
}

static void am_transfer_put( gasnet_token_t token,
      void *buf,
      size_t len)
{
   gasnet_node_t src_node;
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( buf ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )

   //sys.addIncVerData( buf ); 

          DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) buf );
   //fprintf(stderr, "n:%d put copy>  buff %p, %u res=%f, entry is %p\n",  gasnet_mynode(), buf, len, *((float*)buf), ent);
//{
#if 0 //version before deps workarounds
          if (ent != NULL) {ent->increaseVersion();
           /*std::cerr << "n:" << gasnet_mynode() << " upgrading version for tag " << (void *) buf << " to " << ent->getVersion() << std::endl; */}
#else
          if (ent != NULL) 
          { if (ent->getOwner() != NULL) { ent->getOwner()->discard( *_masterDir, (uint64_t) buf, ent); /*std::cerr << "n:" << gasnet_mynode() << " discarding tag " << (void *) buf << std::endl; */ }
            else { ent->increaseVersion() ; /* std::cerr << "n:" << gasnet_mynode() << " upgrading version for tag " << (void *) buf << " to " << ent->getVersion() << std::endl; */ };
          }
#endif
    if ( src_node > 0 )
    {
       depsLock.acquire();
       std::multimap<uint64_t, wdDeps *>::iterator depIt;
       std::pair <std::multimap<uint64_t, wdDeps *>::iterator, std::multimap<uint64_t, wdDeps *>::iterator> allWdsWithDeps = _depsMap.equal_range( (uint64_t) buf );
       if ( allWdsWithDeps.first != allWdsWithDeps.second )
       {
	       for ( depIt = allWdsWithDeps.first; depIt != allWdsWithDeps.second; ++depIt )
	       {
		       //std::cerr << "n:" << gasnet_mynode() << " found a wd that had a dependence for tag " << buf << std::endl;
		       (depIt->second)->count -= 1;
		       if ( (depIt->second)->count == 0 ) 
		       {
			       //std::cerr << "n:" << gasnet_mynode() << " all deps satisfied, submiting wd " << (depIt->second)->wd->getId() << std::endl;
#ifdef HALF_PRESEND
			       if ( wdindc++ == 0 ) { sys.submit( *(depIt->second)->wd ); /*std::cerr<<"n:" <<gasnet_mynode()<< " submitted(2) wd " << ((depIt->second)->wd)->getId() <<std::endl; */}
			       else {  buffWD = (depIt->second)->wd ; /*std::cerr<<"n:" <<gasnet_mynode()<< " saved wd(2) " << buffWD->getId() <<std::endl; */}
#else
			       sys.submit( *(depIt->second)->wd );
#endif
			       delete (depIt->second);

		       }
	       }
	       _depsMap.erase( (uint64_t) buf );
       }
       else
       {
          //std::cerr << "n:" << gasnet_mynode() << " WARNING MSG AM_WORK NOT YET RECEIVED! handle it!" <<std::endl;
          _recvdDeps.insert( (uint64_t) buf); 
       }
       depsLock.release();
    }
//             if (ent->getOwner() != NULL )
//{
//                if ( !ent->isInvalidated() )
//                {
//                   char *tmpBuffer = NEW char[len];
//                   ::memcpy((void *) tmpBuffer, buf, len); 
//                   std::list<uint64_t> tagsToInvalidate;
//                   tagsToInvalidate.push_back( (uint64_t) buf );
//                   //std::cerr  <<"n:" << gasnet_mynode() << " sync host (tag) inv copy in" << std::endl;
//                   _masterDir->synchronizeHost( tagsToInvalidate );
//                   //std::cerr <<"n:" << gasnet_mynode() << " go on with get req " << *(( float *) newCopies[i].getAddress() )<< std::endl;
//                   ::memcpy((void *) buf, tmpBuffer, len); 
//                   delete tmpBuffer;
//                }
//else
//   fprintf(stderr, "n:%d put copy>  buff %p, %u res=%f, entry is %p owner is %p, ALREADY INVALID!\n",  gasnet_mynode(), buf, len, *((float*)buf), ent), ent->getOwner();
//}
//else
//   fprintf(stderr, "n:%d put copy>  buff %p, %u res=%f, entry is %p owner is NULL!\n",  gasnet_mynode(), buf, len, *((float*)buf), ent);
//          }
//}
   //fprintf(stderr, "n:%d put copy>  buff %p, %u res=%f, entry is %p\n",  gasnet_mynode(), buf, len, *((float*)buf), ent);
}

static void am_transfer_put_after_get( gasnet_token_t token,
      void *buf,
      size_t len,
      gasnet_handlerarg_t lastLo,
      gasnet_handlerarg_t lastHi)
{
   gasnet_node_t src_node;
   uint64_t last = ( uint64_t ) MERGE_ARG( lastHi, lastLo );

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) last; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

   if ( last != 0 )
   {
      sys.getNetwork()->getNotify( src_node, last );
   }
   //fprintf(stderr, "get copy>  buff %p, %u, %llu\n",  buf, len, last);
}

static void am_transfer_get( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t origAddrLo,
      gasnet_handlerarg_t origAddrHi,
      gasnet_handlerarg_t tagAddrLo,
      gasnet_handlerarg_t tagAddrHi,
      gasnet_handlerarg_t len,
      gasnet_handlerarg_t lastLo,
      gasnet_handlerarg_t lastHi )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   void *tagAddr = ( void * ) MERGE_ARG( tagAddrHi, tagAddrLo );
   NANOS_INSTRUMENT ( uint64_t last = ( uint64_t ) MERGE_ARG( lastHi, lastLo ); )

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   //fprintf(stderr, "n:%d thd %d am_xfer_get: srcAddr=%p, srcHi=%p, srcLo=%p, dstAddr=%p, dstHi=%p, dstLo=%p res=%f\n", gasnet_mynode(), myThread->getId(), origAddr, (void *)origAddrHi, (void *)origAddrLo, destAddr, (void*)destAddrHi, (void*)destAddrLo, *((float*)origAddr) );

   //sys.addInvData( tagAddr );
   //Directory *thisNodeMasterDirectory = myThread->getThreadWD().getDirectory( false );
   DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) tagAddr );
if (ent != NULL) 
{
 if (ent->getOwner() != NULL )
   if ( !ent->isInvalidated() )
   {
      std::list<uint64_t> tagsToInvalidate;
      tagsToInvalidate.push_back( ( uint64_t ) tagAddr );
      //std::cerr  <<"n:" << gasnet_mynode() << " sync host (tag)" << std::endl;
      _masterDir->synchronizeHost( tagsToInvalidate );
      //std::cerr <<"n:" << gasnet_mynode() << " go on with get req " << *(( float *) tagAddr )<< std::endl;
   }
}
   //fprintf(stderr, "n:%d thd %d am_xfer_get: srcAddr=%p, srcHi=%p, srcLo=%p, dstAddr=%p, dstHi=%p, dstLo=%p res=%f\n", gasnet_mynode(), myThread->getId(), origAddr, (void *)origAddrHi, (void *)origAddrLo, destAddr, (void*)destAddrHi, (void*)destAddrLo, *((float*)origAddr) );
   if ( ( unsigned int ) len <= gasnet_AMMaxLongRequest() )
   {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) last; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )
      //NANOS_INSTRUMENT ( instr->raiseClosePtPEventNkvs ( NANOS_XFER_GET, id, 0, NULL, NULL, src_node ); )

      NANOS_INSTRUMENT ( xferSize = len; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

      if ( gasnet_AMReplyLong2( token, 212, origAddr, len, destAddr, lastLo, lastHi ) != GASNET_OK )
      {
         fprintf( stderr, "gasnet: Error sending reply msg.\n" );
      }

   }
   else
   {
       fprintf( stderr, "gasnet: Error, requested a GET of size > gasnet_AMMaxLong() bytes.\n" );
   }
}

static void am_flash_put( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t len,
      gasnet_handlerarg_t wordSize,
      gasnet_handlerarg_t valueLo,
      gasnet_handlerarg_t valueHi )
{
   gasnet_node_t src_node;
   int i;
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   uint64_t value = ( uint64_t ) MERGE_ARG( valueHi, valueLo );

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   uint64_t *ptr64 = ( uint64_t * ) destAddr;
   uint32_t *ptr32 = ( uint32_t * ) destAddr;
   uint16_t *ptr16 = ( uint16_t * ) destAddr;
   uint8_t *ptr8  = ( uint8_t * ) destAddr;

   uint64_t val64 = ( uint64_t ) value;
   uint32_t val32 = ( uint32_t ) value;
   uint16_t val16 = ( uint16_t ) value;
   uint8_t val8  = ( uint8_t ) value;

   switch ( wordSize )
   {
      case 8:
         for ( i = 0; i < (len/8) ; i++ )
         {
            ptr64[ i ] = val64;
         }
         break;
      case 4:
         for ( i = 0; i < (len/4) ; i++ )
         {
            ptr32[ i ] = val32;
         }
         break;
      case 2:
         for ( i = 0; i < (len/2) ; i++ )
         {
            ptr16[ i ] = val16;
         }
         break;
      case 1:
         for ( i = 0; i < len ; i++ )
         {
            ptr8[ i ] = val8;
         }
         break;
      default:
         break;
   }
   
      
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( destAddr ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )
}

static void am_request_put( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t origAddrLo,
      gasnet_handlerarg_t origAddrHi,
      gasnet_handlerarg_t len,
      gasnet_handlerarg_t dst )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   //fprintf(stderr, "req put from %d to send stuff to %d, addr %p to %p\n", src_node, dst, origAddr, destAddr);

   enqueue_put_request( dst, origAddr, destAddr, len );
}

void GASNetAPI::initialize ( Network *net )
{
   int my_argc = OS::getArgc();
   char **my_argv = OS::getArgv();
   uintptr_t segSize;

   rwgSMP = NEW RemoteWorkGroup( 0 );
   rwgGPU = NEW RemoteWorkGroup( 1 );
 
   
   _net = net;

   gasnet_handlerentry_t htable[] = {
      { 203, (void (*)()) am_exit },
      { 204, (void (*)()) am_exit_reply },
      { 205, (void (*)()) am_work },
      { 206, (void (*)()) am_work_done },
      { 207, (void (*)()) am_malloc },
      { 208, (void (*)()) am_malloc_reply },
      { 209, (void (*)()) am_my_hostname },
      { 210, (void (*)()) am_transfer_put },
      { 211, (void (*)()) am_transfer_get },
      { 212, (void (*)()) am_transfer_put_after_get },
      { 213, (void (*)()) am_flash_put },
      { 214, (void (*)()) am_request_put },
      { 215, (void (*)()) am_work_data },
      { 216, (void (*)()) am_memfree },
      { 217, (void (*)()) am_memrealloc }
   };

   fprintf(stderr, "argc is %d\n", my_argc);
   for (int i = 0; i < my_argc; i++)
      fprintf(stderr, "\t[%d]: %s\n", i, my_argv[i]);

   gasnet_init( &my_argc, &my_argv );

   segSize = gasnet_getMaxLocalSegmentSize();

   gasnet_attach( htable, sizeof( htable ) / sizeof( gasnet_handlerentry_t ), segSize, 0);

   fprintf(stderr, "gasnet: segment size was %p bytes\n", ( void * ) segSize);

   _net->setNumNodes( gasnet_nodes() );
   _net->setNodeNum( gasnet_mynode() );

   _getRequests.reserve( _net->getNumNodes() );
   _sentData.reserve( _net->getNumNodes() );

   for (unsigned int i = 0; i < _net->getNumNodes(); i++ )
   {
      _getRequests.push_back( NEW GetRequestCtl );
      _sentData.push_back( NEW std::set<uint64_t> );
   }

   nodeBarrier();

   if ( _net->getNodeNum() == 0)
   {
      unsigned int i;
      char myHostname[256];
      if ( gethostname( myHostname, 256 ) != 0 )
      {
         fprintf(stderr, "os: Error getting the hostname.\n");
      }

      sys.getNetwork()->setMasterHostname( (char *) myHostname );

      for ( i = 1; i < _net->getNumNodes() ; i++ )
      {
         sendMyHostName( i );
      }
   }

   nodeBarrier();

   if ( _net->getNodeNum() == 0)
   {
      unsigned int idx;
      
      gasnet_seginfo_t seginfoTable[ gasnet_nodes() ];
      gasnet_getSegmentInfo( seginfoTable, gasnet_nodes() );

      void *segmentAddr[ gasnet_nodes() ];
      size_t segmentLen[ gasnet_nodes() ];

      fprintf(stderr, "GasNet segment information:\n");
      for ( idx = 0; idx < gasnet_nodes(); idx += 1)
      {
         segmentAddr[ idx ] = seginfoTable[ idx ].addr;
         segmentLen[ idx ] = seginfoTable[ idx ].size;
         fprintf(stderr, "\tnode %d: @=%p, len=%p\n", idx, seginfoTable[ idx ].addr, (void *) seginfoTable[ idx ].size);
         ClusterInfo::addSegments( gasnet_nodes(), segmentAddr, segmentLen );
      }
      _thisNodeSegment = NEW SimpleAllocator( ( uintptr_t ) ClusterInfo::getSegmentAddr( 0 ), ClusterInfo::getSegmentLen( 0 ) );
   }
}

void GASNetAPI::finalize ()
{
    gasnet_barrier_notify( 0, GASNET_BARRIERFLAG_ANONYMOUS );
    gasnet_barrier_wait( 0, GASNET_BARRIERFLAG_ANONYMOUS );
    //gasnet_AMPoll();
    fprintf(stderr, "Node %d closing the network...\n", _net->getNodeNum());
    //gasnet_exit(0);
    exit(0);
}



void GASNetAPI::poll ()
{
   if (myThread != NULL)
{
   //if (masterDir == NULL) { masterDir = myThread->getThreadWD().getDirectory( true ); std::cerr << "n:" << gasnet_mynode() << " getting masterDir from WD " << myThread->getThreadWD().getId() << std::endl;}
   if (put_req_vector_lock.tryAcquire())
   {
      while (put_req_vector.size() > 0)
      {
         struct put_req_desc *prd = put_req_vector.front();//.pop_front();
         //fprintf(stderr, "process req to node %d / queue size %d\n", prd->dest, put_req_vector.size());

         //void GASNetAPI::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size )
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_IN, key, (nanos_event_value_t) prd->len) );
         put(prd->dest, (uint64_t) prd->destAddr, prd->origAddr, prd->len);
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );

         put_req_vector.pop_front();
         //  fprintf(stderr, "del prd %p size %d\n", prd, put_req_vector.size());
         delete prd;
      }
      put_req_vector_lock.release();
   }
   gasnet_AMPoll();
}
}

void GASNetAPI::sendExitMsg ( unsigned int dest )
{
   if (gasnet_AMRequestShort0( dest, 203 ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe, size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch, void *remoteWdAddr/*, void *remoteThd*/ )
{
   //fprintf(stderr, "sending msg WORK %p, arg size %d to node %d, numPe %d %d max med req\n", work, argSize, dest, numPe, gasnet_AMMaxMedium());
   size_t sent = 0;
   unsigned int msgCount = 0;

   int numCopies = *((int *) &arg[ dataSize ]);
   CopyData *copiesForThisWork = (CopyData *) &arg[ dataSize + sizeof(int)];
   int *depCount = ((int *) &arg[ dataSize + sizeof(int) + sizeof(CopyData) * numCopies ]);
   uint64_t *depAddrs = ((uint64_t *) &arg[ dataSize + sizeof(int) + sizeof(CopyData) * numCopies + sizeof(int) ]);
   
   *depCount = 0;
   
   for (int i = 0; i < numCopies; i++)
   {
      uint64_t tag = copiesForThisWork[i].getAddress();
      sentDataLock.acquire();
      std::set<uint64_t >::iterator addrIt = _sentData[ dest ]->find( tag );
      if( addrIt != _sentData[ dest ]->end() )
      {
          //found an element, I had previously sent a PUT REQUEST to reach this node, set up a dependence.
          depAddrs[ *depCount ] = tag;
          *depCount += 1;
          _sentData[ dest ]->erase( addrIt );
          //std::cerr << "n:" <<gasnet_mynode()<<" detected a dependence need for tag " << (void *) tag << " dest node "<< dest <<std::endl;
      }
      sentDataLock.release();
   }

   while ( (argSize - sent) > gasnet_AMMaxMedium() )
   {
      if ( gasnet_AMRequestMedium3( dest, 215, &arg[ sent ], gasnet_AMMaxMedium(),
               msgCount, 
               ARG_LO( argSize ),
               ARG_HI( argSize ) ) != GASNET_OK )
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
      }
      msgCount++;
      sent += gasnet_AMMaxMedium();
         //fprintf(stderr, "gasnet: sending msg_data %d.\n", sent);
   }

//std::cerr <<"n:" << gasnet_mynode() << " SEND work to node "<< dest << " , numPe is " << numPe << " " << std::endl;
{
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( wdId ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_AM_WORK, id, 0, 0, dest ); )
}

   if (gasnet_AMRequestMedium10( dest, 205, &arg[ sent ], argSize - sent,
            ( gasnet_handlerarg_t ) ARG_LO( work ),
            ( gasnet_handlerarg_t ) ARG_HI( work ),
            ( gasnet_handlerarg_t ) ARG_LO( xlate ),
            ( gasnet_handlerarg_t ) ARG_HI( xlate ),
            ( gasnet_handlerarg_t ) ARG_LO( remoteWdAddr ),
            ( gasnet_handlerarg_t ) ARG_HI( remoteWdAddr ),
   //         ( gasnet_handlerarg_t ) ARG_LO( remoteThd ),
   //         ( gasnet_handlerarg_t ) ARG_HI( remoteThd ),
            dataSize, wdId, numPe, arch ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendWorkDoneMsg ( unsigned int dest, void *remoteWdAddr, int peId )
{
   //fprintf(stderr, "sending msg WORK DONE to node %d, numPe %d\n", dest, numPe);
//std::cerr <<"n:" << gasnet_mynode() << " work DONE, numPe is " << numPe << " " << std::endl;
{
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteWdAddr ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_AM_WORK_DONE, id, 0, 0, dest ); )
}
#ifdef HALF_PRESEND
   if ( wdindc-- == 2 ) { sys.submit( *buffWD ); /*std::cerr<<"n:" <<gasnet_mynode()<< " submitted wd " << buffWD->getId() <<std::endl;*/} 
#endif
   if (gasnet_AMRequestShort3( dest, 206, 
            ( gasnet_handlerarg_t ) ARG_LO( remoteWdAddr ),
            ( gasnet_handlerarg_t ) ARG_HI( remoteWdAddr ),
            ( gasnet_handlerarg_t ) peId
            /*( gasnet_handlerarg_t ) ARG_HI( remoteThd )*/ ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size )
{
#if 0
   gasnet_put_bulk( ( gasnet_node_t ) remoteNode, ( void * ) remoteAddr, localAddr, size );
#endif

   size_t sent = 0, thisReqSize;

#if 0
   unsigned int i = 1;
   unsigned int totalWords;
   unsigned int selectedSize;
   uint64_t value;
   //test the buffer:
   if ( size % 8 == 0 )
   {
      uint64_t *ptr = ( uint64_t * ) localAddr;
      totalWords = size / 8;
      selectedSize = 8;
      value = ptr[ 0 ];
      while ( ptr[ 0 ] == ptr[ i ] )
         i++;
   }
   else if ( size % 4 )
   {
      uint32_t *ptr = (uint32_t *) localAddr;
      totalWords = size / 4;
      selectedSize = 4;
      value = ( uint64_t ) ptr[ 0 ];
      while (ptr[0] == ptr[i])
         i++;
   }
   else if ( size % 2 )
   {
      uint16_t *ptr = (uint16_t *) localAddr;
      totalWords = size / 2;
      selectedSize = 2;
      value = ( uint64_t ) ptr[ 0 ];
      while (ptr[0] == ptr[i])
         i++;
   }
   else
   {
      uint8_t *ptr = (uint8_t *) localAddr;
      totalWords = size;
      selectedSize = 1;
      value = ( uint64_t ) ptr[ 0 ];
      while (ptr[0] == ptr[i])
         i++;
   }
   if ( i == totalWords)
   {
      //fprintf(stderr, "I can do a flash put here!, selected size %d\n", selectedSize);

      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = size; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteAddr ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, remoteNode ); )

      if ( gasnet_AMRequestShort6( remoteNode, 213,
               ( gasnet_handlerarg_t ) ARG_LO( remoteAddr ),
               ( gasnet_handlerarg_t ) ARG_HI( remoteAddr ),
               ( gasnet_handlerarg_t ) size,
               ( gasnet_handlerarg_t ) selectedSize,
               ( gasnet_handlerarg_t ) ARG_LO( value ),
               ( gasnet_handlerarg_t ) ARG_HI( value )) != GASNET_OK )
         {
            fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
         }
   }
   else
#endif
   {

   //sys.addInvData( localAddr );
   DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) localAddr );
if (ent != NULL) 
{
 if (ent->getOwner() != NULL )
{
   //if ( !ent->isInvalidated() )
   //{
      std::list<uint64_t> tagsToInvalidate;
      tagsToInvalidate.push_back( ( uint64_t ) localAddr );
      //std::cerr  <<"n:" << gasnet_mynode() << " sync host (tag)" << std::endl;
      _masterDir->synchronizeHost( tagsToInvalidate );
      //std::cerr <<"n:" << gasnet_mynode() << " go on with PUT req addr is " << localAddr << " data is " << *(( float *) localAddr )<< std::endl;
   //}
}
//else
//
//      std::cerr <<"n:" << gasnet_mynode() << " NO CACHE for addr " << localAddr << " data is " << *(( float *) localAddr )<< std::endl;
}
//else
//      std::cerr <<"n:" << gasnet_mynode() << " NO entry for addr " << localAddr << " data is " << *(( float *) localAddr )<< std::endl;
   
   //fprintf(stderr, "n:%d put ( dest=%d, remote=%p, locla=%p, size=%d, val=%f)\n", gasnet_mynode(), remoteNode, (void *) remoteAddr, localAddr, size, *((float *)localAddr));
   while ( sent < size )
   {
      thisReqSize = ( ( size - sent ) <= gasnet_AMMaxLongRequest() ) ? size - sent : gasnet_AMMaxLongRequest();
      
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = thisReqSize; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteAddr + sent ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, remoteNode ); )

      if ( gasnet_AMRequestLong0( remoteNode, 210,
               &( ( char *) localAddr )[ sent ],
               thisReqSize,
               ( char *) ( remoteAddr + sent ) ) != GASNET_OK)
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
      }
      sent += thisReqSize;
   }
   //fprintf(stderr, "end put ( dest=%d, remote=%p, locla=%p, size=%d)\n", remoteNode, (void *) remoteAddr, localAddr, size);
   }
}
Lock getLock;
Lock getLockGlobal;

void GASNetAPI::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size )
{
#if 0
   gasnet_get_bulk ( localAddr, ( gasnet_node_t ) remoteNode, ( void * ) remoteAddr, size );
#endif
   size_t sent = 0, thisReqSize;

   getLockGlobal.acquire();
   void *addr = _thisNodeSegment->allocate( size );
   getLockGlobal.release();
   //fprintf(stderr, "get ( dest=%d, remote=%p, locla=%p, size=%ld, localtmp=%p, maxreq=%ld)\n", remoteNode, (void *) remoteAddr, localAddr, size, addr, gasnet_AMMaxLongRequest());

//std::cerr << "_getReq[ " << remoteNode << " ][" << (void *) remoteAddr << " GET WAITING !!!! " << std::endl; 
//std::cerr << "_getReq[ " << remoteNode << " ][" << (void *) remoteAddr << " GET WAITING !!!! " << std::endl; 
//std::cerr << "_getReq[ " << remoteNode << " ][" << (void *) remoteAddr << " GET WAITING !!!! " << std::endl; 
//std::cerr << "_getReq[ " << remoteNode << " ][" << (void *) remoteAddr << " GET WAITING !!!! " << std::endl; 
   getLock.acquire();
   (*_getRequests[ remoteNode ])[ remoteAddr ] = GET_WAITING;
   getLock.release();
   
   while ( sent < size )
   {
      thisReqSize = ( ( size - sent ) <= gasnet_AMMaxLongRequest() ) ? size - sent : gasnet_AMMaxLongRequest();

      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ( ( sent + thisReqSize ) == size ) * remoteAddr ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, remoteNode ); )

      //fprintf(stderr, "n:%d send get req to node %d(src=%p, srcHi=%p, srcLo=%p, dst=%p dstHi=%p, dstLo=%p localtag=%p)\n", gasnet_mynode(), remoteNode, (void *) remoteAddr, (void *) ARG_HI( remoteAddr + sent ), (void *) ARG_LO( remoteAddr + sent ), (void *) ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent ), (void *) ARG_HI( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ), (void *) ARG_LO( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ), localAddr  );
      if ( gasnet_AMRequestShort9( remoteNode, 211,
               ( gasnet_handlerarg_t ) ARG_LO( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ),
               ( gasnet_handlerarg_t ) ARG_HI( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ),
               ( gasnet_handlerarg_t ) ARG_LO( remoteAddr + sent ),
               ( gasnet_handlerarg_t ) ARG_HI( remoteAddr + sent ),
               ( gasnet_handlerarg_t ) ARG_LO( remoteAddr ),
               ( gasnet_handlerarg_t ) ARG_HI( remoteAddr ),
               ( gasnet_handlerarg_t ) thisReqSize,
               ( gasnet_handlerarg_t ) ARG_LO( ( ( sent + thisReqSize ) == size ) * remoteAddr ),
               ( gasnet_handlerarg_t ) ARG_HI( ( ( sent + thisReqSize ) == size ) * remoteAddr ) ) != GASNET_OK)
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
      }
      sent += thisReqSize;
   }

   getLock.acquire();
   //   std::cerr  <<"n:" << gasnet_mynode() << " pre get wait at gasnet " << std::endl;
   while ( (*_getRequests[ remoteNode ])[ remoteAddr ] == GET_WAITING )
      /*sys.getNetwork()->*/poll( /*myThread->getId()*/ );
   _getRequests[ remoteNode ]->erase( remoteAddr );
  //    std::cerr  <<"n:" << gasnet_mynode() << " post get wait at gasnet " << std::endl;
   getLock.release();

   //sys.addInvData( localAddr );
   DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) localAddr );
   //if (ent != NULL) 
   //{ ent->increaseVersion();}
if (ent != NULL) 
{
 if (ent->getOwner() != NULL )
{
   if ( !ent->isInvalidated() )
   {
      std::list<uint64_t> tagsToInvalidate;
      tagsToInvalidate.push_back( ( uint64_t ) localAddr );
      //std::cerr  <<"n:" << gasnet_mynode() << " sync host (tag)" << std::endl;
      _masterDir->synchronizeHost( tagsToInvalidate );
      //std::cerr <<"n:" << gasnet_mynode() << " go on with get req " << *(( float *) localAddr )<< std::endl;
   }
}
//else
//      std::cerr <<"n:" << gasnet_mynode() << " OWNER IS NULL data" << *(( float *) localAddr )<< std::endl;
}
   //std::cerr << "n:" << gasnet_mynode() << " completed a get from node " << remoteNode << ", entry for addr " << (void *) localAddr << " is " << ent << " remote tag is " << (void *) remoteAddr <<" data is " << *(( float *) addr )<< std::endl;

   //fprintf(stderr, "end get ( dest=%d, remote=%p, locla=%p, size=%d)\n", remoteNode, (void *) remoteAddr, localAddr, size);

   // copy the data to the correct addr;
   ::memcpy( localAddr, addr, size );
   getLockGlobal.acquire();
   _thisNodeSegment->free( addr );
   getLockGlobal.release();
   //fprintf(stderr, "!!!!!!!! end get ( dest=%d, remote=%p, locla=%p, size=%d, res=%f)\n", remoteNode, (void *) remoteAddr, localAddr, size, *((float*)addr));
}

void GASNetAPI::malloc ( unsigned int remoteNode, size_t size, unsigned int id )
{
   if (gasnet_AMRequestShort2( remoteNode, 207, size, id ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }
} 
void GASNetAPI::memRealloc ( unsigned int remoteNode, void *oldAddr, size_t oldSize, void *newAddr, size_t newSize )
{
   if (gasnet_AMRequestShort8( remoteNode, 217,
	ARG_LO( oldAddr ),
	ARG_HI( oldAddr ),
	ARG_LO( oldSize ),
	ARG_HI( oldSize ),
	ARG_LO( newAddr ),
	ARG_HI( newAddr ),
	ARG_LO( newSize ),
	ARG_HI( newSize ) 
	) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }
}

void GASNetAPI::memFree ( unsigned int remoteNode, void *addr )
{
   if (gasnet_AMRequestShort2( remoteNode, 216, ARG_LO( addr ), ARG_HI( addr ) ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }
}

void GASNetAPI::nodeBarrier()
{
   gasnet_barrier_notify( 0, GASNET_BARRIERFLAG_ANONYMOUS );
   gasnet_barrier_wait( 0, GASNET_BARRIERFLAG_ANONYMOUS );
}

void GASNetAPI::getNotify( unsigned int node, uint64_t remoteAddr )
{
   (*_getRequests[ node ])[ remoteAddr ] = GET_COMPLETE;
}

void GASNetAPI::sendMyHostName( unsigned int dest )
{
   const char *masterHostname = sys.getNetwork()->getMasterHostname();

   if ( masterHostname == NULL )
      fprintf(stderr, "Error, master hostname not set!\n" );
   
   if ( gasnet_AMRequestMedium0( dest, 209, ( void * ) masterHostname, strlen( masterHostname ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest );
   }
}

void GASNetAPI::sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, size_t len )
{
   //fprintf(stderr, "tag %p req put to %d to send stuff to %d : %p\n", origAddr, dest, dataDest, dstAddr);
      sentDataLock.acquire();
   _sentData[ dataDest ]->insert( dstAddr );
      sentDataLock.release();
   if ( gasnet_AMRequestShort6( dest, 214,
            ( gasnet_handlerarg_t ) ARG_LO( dstAddr ),
            ( gasnet_handlerarg_t ) ARG_HI( dstAddr ),
            ( gasnet_handlerarg_t ) ARG_LO( ( ( uintptr_t ) origAddr ) ),
            ( gasnet_handlerarg_t ) ARG_HI( ( ( uintptr_t ) origAddr ) ),
            ( gasnet_handlerarg_t ) len,
            ( gasnet_handlerarg_t ) dataDest ) != GASNET_OK )

   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::setMasterDirectory(Directory *dir)
{
   //std::cerr << "n:" << gasnet_mynode() << " set Master dir to " << dir << std::endl;
   _masterDir = dir;
}
