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

#include <gasnet.h>

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

static void am_work(gasnet_token_t token, void *arg, size_t argSize, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe )
{
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
    WD *wd = new WD( dd, sizeof(struct work_wrapper_args), warg, numCopies, newCopies );
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
    if ( gasnet_AMReplyShort2( token, 208, ( gasnet_handlerarg_t ) addr, (gasnet_handlerarg_t ) id ) != GASNET_OK )
    {
       fprintf( stderr, "gasnet: Error sending a message to node %d.\n", src_node );
    }
}

/* GASNet medium active message handler */
static void am_malloc_reply( gasnet_token_t token, gasnet_handlerarg_t addr, unsigned int id )
{
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
      { 209, (void (*)()) am_my_hostname }
   };

   fprintf(stderr, "argc is %d\n", my_argc);
   for (int i = 0; i < my_argc; i++)
      fprintf(stderr, "\t[%d]: %s\n", i, my_argv[i]);

   //inspect_environ();
   gasnet_init( &my_argc, &my_argv );

   segSize = gasnet_getMaxLocalSegmentSize();

   gasnet_attach( htable, sizeof( htable ) / sizeof( gasnet_handlerentry_t ), segSize, 0);

   fprintf(stderr, "gasnet: segment size was %d bytes\n", segSize);

   _net->setNumNodes( gasnet_nodes() );
   _net->setNodeNum( gasnet_mynode() );

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
         fprintf(stderr, "\tnode %d: @=%p, len=%d\n", idx, seginfoTable[ idx ].addr, seginfoTable[ idx ].size);
         ClusterInfo::addSegments( gasnet_nodes(), segmentAddr, segmentLen );
      }


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
   if (gasnet_AMRequestMedium4( dest, 205, arg, argSize, work, dataSize, wdId, numPe ) != GASNET_OK)
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
   gasnet_put_bulk( ( gasnet_node_t ) remoteNode, ( void * ) remoteAddr, localAddr, size );
}

void GASNetAPI::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size )
{
   gasnet_get_bulk ( localAddr, ( gasnet_node_t ) remoteNode, ( void * ) remoteAddr, size );
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
