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

#include "plugin.hpp"
#include "clusterdd.hpp"
#include "clusternode.hpp"
#include "clusternodeinfo.hpp"
#include "clustermsg.hpp"
#include "system.hpp"
#include "os.hpp"

#include <gasnet.h>

namespace nanos {
namespace ext {

volatile bool finish_gasnet = false;

class GASNetMessaging
{
    public:
        static void sendMessage(ClusterRemoteNode *dest)
        {
            if (gasnet_AMRequestShort0(dest->getClusterID(), 203) != GASNET_OK)
            {
                fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest->getClusterID());
            }
            fprintf(stderr, "Message sent to node %d\n", dest->getClusterID());
        }
};

void am_handler_exit(gasnet_token_t token)
{
    gasnet_node_t src_node;
    if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
    {
        fprintf(stderr, "gasnet: Error obtaining node information.\n");
    }
    fprintf(stderr, "EXIT msg from node %d.\n", src_node);
    //gasnet_AMReplyShort0(token, 204);
    finish_gasnet = true;
}

void am_handler_exit_reply(gasnet_token_t token)
{
    gasnet_node_t src_node;
    if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
    {
        fprintf(stderr, "gasnet: Error obtaining node information.\n");
    }
    fprintf(stderr, "EXIT message to node %d completed.\n", src_node);
}

void GASNet_finalize()
{
    gasnet_barrier_notify(0,GASNET_BARRIERFLAG_ANONYMOUS);
    gasnet_barrier_wait(0,GASNET_BARRIERFLAG_ANONYMOUS);
    gasnet_AMPoll();
    fprintf(stderr, "Node %d closing the network...\n", ClusterNodeInfo::getNodeNum());
    gasnet_exit(0);
}

void GASNet_idleLoop()
{
    while (true)
    {
        if (finish_gasnet == false)
        {
            gasnet_AMPoll();
        }
        else
        {
            GASNet_finalize();
            exit(0);
        }
    }
}

class ClusterPlugin : public Plugin
{
   public:
      ClusterPlugin() : Plugin( "Cluster PE Plugin", 1 ) {}

      virtual void config( Config& config )
      {
         config.setOptionsSection( "Cluster Arch", "Cluster specific options" );
         //config.registerConfigOption ( "num-gpus", new Config::IntegerVar( _numGPUs ),
         //                              "Defines the maximum number of GPUs to use" );
         //config.registerArgOption ( "num-gpus", "gpus" );
         //config.registerEnvOption ( "num-gpus", "NX_GPUS" );
      }

      virtual void init()
      {
          int my_argc = OS::getArgc();
          char **my_argv = OS::getArgv();

          gasnet_handlerentry_t htable[] = { { 203, (void (*)()) am_handler_exit }, { 204, (void (*)()) am_handler_exit_reply } };

          gasnet_init(&my_argc, &my_argv);
          gasnet_attach(htable, sizeof(htable)/sizeof(gasnet_handlerentry_t), 0, 0);

          ClusterMsg::setSendFinishMessageFunc(GASNetMessaging::sendMessage);

          ClusterNodeInfo::setIdleLoopFunc(GASNet_idleLoop);
          ClusterNodeInfo::setNetworkFinalizeFunc(GASNet_finalize);

          ClusterNodeInfo::setNumNodes(gasnet_nodes());
          ClusterNodeInfo::setNodeNum(gasnet_mynode());
          fprintf(stderr, "Im node %d, total nodes %d\n", ClusterNodeInfo::getNodeNum(), ClusterNodeInfo::getNumNodes());
          //_nodeInfo = new ClusterNodeInfo(gasnet_nodes(), gasnet_mynode());

          //ClusterDD::_numNodes = gasnet_nodes();
          //ClusterDD::_nodeNum = gasnet_mynode();

          //thisNode = new ClusterLocalNode(gasnet_mynode());
          
          gasnet_barrier_notify(0,GASNET_BARRIERFLAG_ANONYMOUS);
          gasnet_barrier_wait(0,GASNET_BARRIERFLAG_ANONYMOUS);

          //if (gasnet_mynode() == 0)
          //    GASNetMessaging::sendMessage(1);
      }
};

}
}

nanos::ext::ClusterPlugin NanosXPlugin;

