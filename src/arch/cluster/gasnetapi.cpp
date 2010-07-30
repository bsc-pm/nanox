#include "gasnetapi.hpp"
#include "smpdd.hpp"
#include "system.hpp"
#include "os.hpp"

#include <gasnet.h>

using namespace nanos;
using namespace ext;

struct work_wrapper_args
{
   void ( * work ) ( void * );
   void *arg;
   size_t argSize;
};

static void wk_wrapper(struct work_wrapper_args *arg)
{
   arg->work(arg->arg);

   delete arg;

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
    if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
    {
        fprintf(stderr, "gasnet: Error obtaining node information.\n");
    }
    //fprintf(stderr, "am_work: WORK message from node %d: fct %p, argSize %d.\n", src_node, work, argSize);

    struct work_wrapper_args * warg = new struct work_wrapper_args;
    warg->work = work;
    warg->arg = arg;
    warg->argSize = arg0;

    CopyData *recvcd = (CopyData *) &((char *) arg)[arg0];

    //fprintf(stderr, "NUM COPIES %d addr %llx, in? %s, out? %s\n",
    //      1,
    //      recvcd->getAddress(),
    //      recvcd->isInput() ? "yes" : "no",
    //      recvcd->isOutput() ? "yes" : "no" );
    //
    SMPDD * dd = new SMPDD ( ( SMPDD::work_fct ) wk_wrapper );
    WD *wd = new WD( dd, sizeof(struct work_wrapper_args), warg, ( size_t ) ( argSize - arg0 ) / sizeof(CopyData), ( CopyData * ) &( ( char * )arg )[ arg0 ] );
    //WD *wd = new WD( dd/*, sizeof(struct work_wrapper_args), warg*/ );

    //SMPDD * dd = new SMPDD ( ( SMPDD::work_fct ) work);
    //WD *wd = new WD( dd, argSize, arg );

    //fprintf(stderr, "NUM COPIES %d addr %llx, in? %s, out? %s\n",
    //      wd->getNumCopies(),
    //      wd->getCopies()[0].getAddress(),
    //      wd->getCopies()[0].isInput() ? "yes" : "no",
    //      wd->getCopies()[0].isOutput() ? "yes" : "no" );

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

   gasnet_init(&my_argc, &my_argv);

   segSize = gasnet_getMaxLocalSegmentSize();

   gasnet_attach(htable, sizeof(htable)/sizeof(gasnet_handlerentry_t), segSize, 0);

   fprintf(stderr, "gasnet: segment size was %d bytes\n", segSize);

   _net->setNumNodes(gasnet_nodes());
   _net->setNodeNum(gasnet_mynode());

   gasnet_barrier_notify(0,GASNET_BARRIERFLAG_ANONYMOUS);
   gasnet_barrier_wait(0,GASNET_BARRIERFLAG_ANONYMOUS);
}

void GasnetAPI::finalize ()
{
    gasnet_barrier_notify(0,GASNET_BARRIERFLAG_ANONYMOUS);
    gasnet_barrier_wait(0,GASNET_BARRIERFLAG_ANONYMOUS);
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
   fprintf(stderr, "sending msg WORK %p, arg size %d to node %d\n", work, argSize, dest);
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
#if 0
   unsigned int sent = 0, thisSendSize;

   while (sent < size)
   {
      thisSendSize = (size - sent) > 1000 ? 1000 : size - sent;

      fprintf(stderr, "gasnet_put_bulk size %d\n", thisSendSize);
      
      gasnet_put_bulk( ( gasnet_node_t ) remoteNode, &( ( ( char * ) remoteAddr )[ sent ] ), &( ( ( char * )localAddr )[ sent ] ), thisSendSize );

      sent += thisSendSize;
   }
#else
   gasnet_put_bulk( ( gasnet_node_t ) remoteNode, &( ( ( char * ) remoteAddr )[ 0 ] ), &( ( ( char * )localAddr )[ 0 ] ), size );
#endif
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
