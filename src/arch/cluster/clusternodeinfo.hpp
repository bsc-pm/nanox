#ifndef _CLUSTER_NODE_INFO
#define _CLUSTER_NODE_INFO

#include "clusternode.hpp"

namespace nanos {
namespace ext {

typedef void ( *idleLoopFunc ) ( void );
typedef void ( *networkFinalizeFunc ) ( void );

class ClusterNodeInfo
{
    private:
        static int _numNodes; 
        static int _nodeNum;
        static ClusterLocalNode *_thisNode;
        static idleLoopFunc _idleLoopFunc;
        static networkFinalizeFunc _networkFinalizeFunc;
        
    public:
        static const int MASTER_NODE_NUM = 0;

        //ClusterNodeInfo(int numNodes, int nodeNum);
        static void setNumNodes(int numNodes);
        static void setNodeNum(int nodeNum);

        static int getNumNodes();
        static int getNodeNum();
        static ClusterLocalNode *getThisNodePE();

        static void setIdleLoopFunc(idleLoopFunc _func);
        static void callIdleLoopFunc();
        
        static void setNetworkFinalizeFunc(networkFinalizeFunc _func);
        static void callNetworkFinalizeFunc();
};

}
}

#endif
