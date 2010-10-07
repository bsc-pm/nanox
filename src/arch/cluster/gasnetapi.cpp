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
#include "clusterdevice.hpp"

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
   void *arg;
   size_t argSize;
};

static void wk_wrapper(struct work_wrapper_args *arg)
{
   //fprintf(stderr, "node %d starting work> outline %p, arg struct %p\n", sys.getNetwork()->getNodeNum(), arg->work, arg->arg);
   arg->work(arg->arg);

   delete arg;

   //fprintf(stderr, "node %d finishing work> outline %p, arg struct %p\n", sys.getNetwork()->getNodeNum(), arg->work, arg->arg);
   sys.getNetwork()->sendWorkDoneMsg( Network::MASTER_NODE_NUM );
}

static void am_exit(gasnet_token_t token)
{
    gasnet_node_t src_node;
    if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
    {
        fprintf(stderr, "gasnet: Error obtaining node information.\n");
    }
    fprintf(stderr, "EXIT msg from node %d.\n", src_node);
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

static void am_work(gasnet_token_t token, void *arg, size_t argSize, void ( *work ) ( void * ), unsigned int arg0 )
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
    warg->arg = arg;
    warg->argSize = arg0;

    //CopyData *recvcd = (CopyData *) &((char *) arg)[arg0];
    //fprintf(stderr, "NUM COPIES %d addr %llx, in? %s, out? %s\n",
    //      1,
    //      recvcd->getAddress(),
    //      recvcd->isInput() ? "yes" : "no",
    //      recvcd->isOutput() ? "yes" : "no" );
    
    unsigned int numCopies = ( argSize - arg0 ) / sizeof( CopyData );
    CopyData *newCopies = new CopyData[ numCopies ]; 

    for (i = 0; i < numCopies; i += 1) {
       new ( &newCopies[i] ) CopyData( *( (CopyData *) &( ( char * )arg)[ arg0 + i * sizeof( CopyData ) ] ) );
    }

    SMPDD * dd = new SMPDD ( ( SMPDD::work_fct ) wk_wrapper );
    WD *wd = new WD( dd, sizeof(struct work_wrapper_args), warg, numCopies, newCopies );

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

    sys.submit( *wd );

    //fprintf(stderr, "out of am_work.\n", src_node, work, argSize);
}

static void am_work_done( gasnet_token_t token )
{
    gasnet_node_t src_node;
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    //fprintf(stderr, "WORK DONE msg from node %d.\n", src_node);
    sys.getNetwork()->notifyWorkDone( src_node );
}

static void am_malloc( gasnet_token_t token, gasnet_handlerarg_t size )
{
    gasnet_node_t src_node;
    void *addr = NULL;
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    //fprintf(stderr, "WORK DONE msg from node %d.\n", src_node);
    addr = malloc( ( size_t ) size );
    if ( gasnet_AMReplyShort1( token, 208, ( gasnet_handlerarg_t ) addr ) != GASNET_OK )
    {
       fprintf( stderr, "gasnet: Error sending a message to node %d.\n", src_node );
    }
}

static void am_malloc_reply( gasnet_token_t token, gasnet_handlerarg_t addr )
{
    gasnet_node_t src_node;
    if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
    {
        fprintf( stderr, "gasnet: Error obtaining node information.\n" );
    }
    //fprintf(stderr, "WORK DONE msg from node %d.\n", src_node);
    sys.getNetwork()->notifyMalloc( src_node, ( void * ) addr );
}

void GasnetAPI::initialize ( Network *net )
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
      { 208, (void (*)()) am_malloc_reply }
   };

   //fprintf(stderr, "argc is %d\n", my_argc);
   //for (int i = 0; i < my_argc; i++)
   //   fprintf(stderr, "\t[%d]: %s\n", i, my_argv[i]);

   //inspect_environ();
   gasnet_init( &my_argc, &my_argv );

   segSize = gasnet_getMaxLocalSegmentSize();

   gasnet_attach( htable, sizeof( htable ) / sizeof( gasnet_handlerentry_t ), segSize, 0);

   fprintf(stderr, "gasnet: segment size was %d bytes\n", segSize);

   _net->setNumNodes( gasnet_nodes() );
   _net->setNodeNum( gasnet_mynode() );

   gasnet_barrier_notify( 0, GASNET_BARRIERFLAG_ANONYMOUS );
   gasnet_barrier_wait( 0, GASNET_BARRIERFLAG_ANONYMOUS );

   if ( gasnet_mynode() == 0)
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
         ClusterDevice::addSegments( gasnet_nodes(), segmentAddr, segmentLen );
      }


   }
}

void GasnetAPI::finalize ()
{
    gasnet_barrier_notify( 0, GASNET_BARRIERFLAG_ANONYMOUS );
    gasnet_barrier_wait( 0, GASNET_BARRIERFLAG_ANONYMOUS );
    //gasnet_AMPoll();
    fprintf(stderr, "Node %d closing the network...\n", _net->getNodeNum());
    gasnet_exit(0);
}

void GasnetAPI::poll ()
{
   gasnet_AMPoll();
}

void GasnetAPI::sendExitMsg ( unsigned int dest )
{
   if (gasnet_AMRequestShort0( dest, 203 ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GasnetAPI::sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, size_t argSize, void * arg )
{
   //fprintf(stderr, "sending msg WORK %p, arg size %d to node %d\n", work, argSize, dest);
   if (gasnet_AMRequestMedium2( dest, 205, arg, argSize, work, arg0 ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GasnetAPI::sendWorkDoneMsg ( unsigned int dest )
{
   if (gasnet_AMRequestShort0( dest, 206 ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GasnetAPI::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size )
{
   gasnet_put_bulk( ( gasnet_node_t ) remoteNode, ( void * ) remoteAddr, localAddr, size );
}

void GasnetAPI::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size )
{
   gasnet_get_bulk ( localAddr, ( gasnet_node_t ) remoteNode, ( void * ) remoteAddr, size );
}

void GasnetAPI::malloc ( unsigned int remoteNode, size_t size )
{
   if (gasnet_AMRequestShort1( remoteNode, 207, size ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }
}
