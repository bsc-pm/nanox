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
#include "system.hpp"
#include "os.hpp"
#include "clusterinfo.hpp"
#include "instrumentation.hpp"
#include <list>

#include <gasnet.h>

#ifdef __powerpc64__

#define MERGE_ARG( _Hi, _Lo) ( ( ( uintptr_t ) ( _Lo ) ) + ( ( ( uintptr_t ) ( _Hi ) ) << 32 ) )
#define ARG_HI( _Arg ) ( ( uint32_t ) ( ( ( uintptr_t ) ( _Arg ) ) >> 32 ) )
#define ARG_LO( _Arg ) ( ( uint32_t ) ( ( uintptr_t ) _Arg ) )

#else

#define MERGE_ARG( _Hi, _Lo) ( ( uintptr_t ) ( _Lo ) )
#define ARG_HI( _Arg ) ( ( uint32_t ) 0 )
#define ARG_LO( _Arg ) ( ( uint32_t ) _Arg )

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
   struct put_req_desc *prd = new struct put_req_desc();
   //fprintf(stderr, "enqueue req to node %d\n", dest);
   prd->dest = dest;
   prd->origAddr = origAddr;
   prd->destAddr = destAddr;
   prd->len = len;

   put_req_vector.push_back( prd );
}

struct work_wrapper_args
{
   void ( * work ) ( void * );
   char *arg;
   unsigned int id;
   unsigned int numPe;
   size_t argSize;
};

static void wk_wrapper(struct work_wrapper_args *arg)
{
   //fprintf(stderr, "node %d starting work> outline %p, arg struct %p rId %d, numPe %d\n", sys.getNetwork()->getNodeNum(), arg->work, arg->arg, arg->id, arg->numPe);
   arg->work(arg->arg);

   //fprintf(stderr, "node %d finishing work> outline %p, arg struct %p rId %d, numPe %d\n", sys.getNetwork()->getNodeNum(), arg->work, arg->arg, arg->id, arg->numPe);
   sys.getNetwork()->sendWorkDoneMsg( Network::MASTER_NODE_NUM, arg->numPe );

   delete[] arg->arg;
   delete arg;
}

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
    fprintf(stderr, "EXIT message to node %d completed.\n", src_node);
}

static void am_work(gasnet_token_t token, void *arg, size_t argSize, void *workLo, void * workHi, unsigned int dataSize, unsigned int wdId, unsigned int numPe )
{
   void (*work)( void *) = (void (*)(void *)) MERGE_ARG( workHi, workLo );
    gasnet_node_t src_node;
    unsigned int i;
    if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
    {
        fprintf(stderr, "gasnet: Error obtaining node information.\n");
    }
    //fprintf(stderr, "am_work: WORK message from node %d: fct %p, argSize %d.\n", src_node, work, argSize);

    struct work_wrapper_args * warg = new struct work_wrapper_args;
    bzero(warg, sizeof(struct work_wrapper_args));
    warg->work = work;
    warg->arg = new char [ dataSize ];
    memcpy(warg->arg, arg, dataSize);
    warg->id = wdId;
    warg->numPe = numPe;
    warg->argSize = dataSize;

    //CopyData *recvcd = (CopyData *) &((char *) arg)[arg0];
    //fprintf(stderr, "NUM COPIES %d addr %llx, in? %s, out? %s\n",
    //      1,
    //      recvcd->getAddress(),
    //      recvcd->isInput() ? "yes" : "no",
    //      recvcd->isOutput() ? "yes" : "no" );
    
    unsigned int numCopies = ( argSize - dataSize ) / sizeof( CopyData );
    CopyData *newCopies = new CopyData[ numCopies ]; 

    for (i = 0; i < numCopies; i += 1) {
       new ( &newCopies[i] ) CopyData( *( (CopyData *) &( ( char * )arg)[ dataSize + i * sizeof( CopyData ) ] ) );
    }

    SMPDD * dd = new SMPDD ( ( SMPDD::work_fct ) wk_wrapper );
    WD *wd = new WD( dd, sizeof(struct work_wrapper_args), 1, warg, numCopies, newCopies );
    wd->setId( wdId );

    //fprintf(stderr, "WD %p , args->arg %p size %d args->id %d\n", wd, warg->arg, arg0, warg->id );

    wd->setPe( NULL );
    //WD *wd = new WD( dd/*, sizeof(struct work_wrapper_args), warg*/ );

    //SMPDD * dd = new SMPDD ( ( SMPDD::work_fct ) work);
    //WD *wd = new WD( dd, argSize, arg );

    //fprintf(stderr, "WD is %p\n", wd);
    //for (int i = 0; i < wd->getNumCopies(); i++)
    //fprintf(stderr, "NUM COPIES %d addr %llx, in? %s, out? %s, sharing %d\n",
    //      wd->getNumCopies(),
    //      wd->getCopies()[i].getAddress(),
    //      wd->getCopies()[i].isInput() ? "yes" : "no",
    //      wd->getCopies()[i].isOutput() ? "yes" : "no",
    //      wd->getCopies()[i].getSharing());

    {
       NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
       NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) numPe) << 32 ) + gasnet_mynode() ; )
       NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *wd, NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
    }

    sys.submit( *wd );

    //fprintf(stderr, "out of am_work.\n", src_node, work, argSize);
}

static void am_work_done( gasnet_token_t token, unsigned int numPe )
{
    gasnet_node_t src_node;
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    //fprintf(stderr, "WORK DONE msg from node %d, numPe %d.\n", src_node, numPe);
    sys.getNetwork()->notifyWorkDone( src_node, numPe );
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

   //fprintf(stderr, "put copy>  buff %p, %u\n",  buf, len);
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
      gasnet_handlerarg_t len,
      gasnet_handlerarg_t lastLo,
      gasnet_handlerarg_t lastHi )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   NANOS_INSTRUMENT ( uint64_t last = ( uint64_t ) MERGE_ARG( lastHi, lastLo ); )

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
       fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

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
      { 214, (void (*)()) am_request_put }
   };

   fprintf(stderr, "argc is %d\n", my_argc);
   for (int i = 0; i < my_argc; i++)
      fprintf(stderr, "\t[%d]: %s\n", i, my_argv[i]);

   //inspect_environ();
   gasnet_init( &my_argc, &my_argv );

   segSize = gasnet_getMaxLocalSegmentSize();

   gasnet_attach( htable, sizeof( htable ) / sizeof( gasnet_handlerentry_t ), segSize, 0);

   fprintf(stderr, "gasnet: segment size was %p bytes\n", ( void * ) segSize);

   _net->setNumNodes( gasnet_nodes() );
   _net->setNodeNum( gasnet_mynode() );

   _getRequests.reserve( _net->getNumNodes() );

   for (unsigned int i = 0; i < _net->getNumNodes(); i++ )
   {
      _getRequests.push_back( new GetRequestCtl );
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
      _thisNodeSegment = new SimpleAllocator( ( uintptr_t ) ClusterInfo::getSegmentAddr( 0 ), ClusterInfo::getSegmentLen( 0 ) );
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
   if (put_req_vector_lock.tryAcquire())
   {
      while (put_req_vector.size() > 0)
      {
         struct put_req_desc *prd = put_req_vector.front();//.pop_front();
         //fprintf(stderr, "process req to node %d / queue size %d\n", prd->dest, put_req_vector.size());

         //void GASNetAPI::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size )
         put(prd->dest, (uint64_t) prd->destAddr, prd->origAddr, prd->len);

         put_req_vector.pop_front();
         //  fprintf(stderr, "del prd %p size %d\n", prd, put_req_vector.size());
         delete prd;
      }
      put_req_vector_lock.release();
   }
   gasnet_AMPoll();
}

void GASNetAPI::sendExitMsg ( unsigned int dest )
{
   if (gasnet_AMRequestShort0( dest, 203 ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe, size_t argSize, void * arg )
{
   //fprintf(stderr, "sending msg WORK %p, arg size %d to node %d, numPe %d\n", work, argSize, dest, numPe);
   if (gasnet_AMRequestMedium5( dest, 205, arg, argSize, (gasnet_handlerarg_t ) ARG_LO( work ), (gasnet_handlerarg_t) ARG_HI( work ), dataSize, wdId, numPe ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendWorkDoneMsg ( unsigned int dest, unsigned int numPe )
{
   //fprintf(stderr, "sending msg WORK DONE to node %d, numPe %d\n", dest, numPe);
   if (gasnet_AMRequestShort1( dest, 206, numPe ) != GASNET_OK)
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
   //fprintf(stderr, "put ( dest=%d, remote=%p, locla=%p, size=%d)\n", remoteNode, (void *) remoteAddr, localAddr, size);
   
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

void GASNetAPI::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size )
{
#if 0
   gasnet_get_bulk ( localAddr, ( gasnet_node_t ) remoteNode, ( void * ) remoteAddr, size );
#endif
   size_t sent = 0, thisReqSize;

   void *addr = _thisNodeSegment->allocate( size );
   //fprintf(stderr, "get ( dest=%d, remote=%p, locla=%p, size=%d)\n", remoteNode, (void *) remoteAddr, localAddr, size);

   (*_getRequests[ remoteNode ])[ remoteAddr ] = GET_WAITING;
   
   while ( sent < size )
   {
      thisReqSize = ( ( size - sent ) <= gasnet_AMMaxLongRequest() ) ? size - sent : gasnet_AMMaxLongRequest();

      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ( ( sent + thisReqSize ) == size ) * remoteAddr ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, remoteNode ); )

      if ( gasnet_AMRequestShort7( remoteNode, 211,
               ( gasnet_handlerarg_t ) ARG_LO( ( ( uintptr_t ) addr ) + sent ),
               ( gasnet_handlerarg_t ) ARG_HI( ( ( uintptr_t ) addr ) + sent ),
               ( gasnet_handlerarg_t ) ARG_LO( remoteAddr + sent ),
               ( gasnet_handlerarg_t ) ARG_HI( remoteAddr + sent ),
               ( gasnet_handlerarg_t ) thisReqSize,
               ( gasnet_handlerarg_t ) ARG_LO( ( ( sent + thisReqSize ) == size ) * remoteAddr ),
               ( gasnet_handlerarg_t ) ARG_HI( ( ( sent + thisReqSize ) == size ) * remoteAddr ) ) != GASNET_OK)
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
      }
      sent += thisReqSize;
   }

   while ( (*_getRequests[ remoteNode ])[ remoteAddr ] == GET_WAITING )
      poll();
   _getRequests[ remoteNode ]->erase( remoteAddr );

   //fprintf(stderr, "end get ( dest=%d, remote=%p, locla=%p, size=%d)\n", remoteNode, (void *) remoteAddr, localAddr, size);

   // copy the data to the correct addr;
   ::memcpy( localAddr, addr, size );
   _thisNodeSegment->free( addr );
   //fprintf(stderr, "!!!!!!!! end get ( dest=%d, remote=%p, locla=%p, size=%d)\n", remoteNode, (void *) remoteAddr, localAddr, size);
}

void GASNetAPI::malloc ( unsigned int remoteNode, size_t size, unsigned int id )
{
   if (gasnet_AMRequestShort2( remoteNode, 207, size, id ) != GASNET_OK)
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
   //fprintf(stderr, "req put to %d to send stuff to %d\n", dest, dataDest);
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
