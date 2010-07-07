#include "clusternodeinfo.hpp"

namespace nanos {
namespace ext {


int ClusterNodeInfo::_numNodes; 
int ClusterNodeInfo::_nodeNum;
ClusterLocalNode *ClusterNodeInfo::_thisNode;
idleLoopFunc ClusterNodeInfo::_idleLoopFunc;
networkFinalizeFunc ClusterNodeInfo::_networkFinalizeFunc;

void ClusterNodeInfo::setNumNodes(int numNodes) { _numNodes = numNodes; }
void ClusterNodeInfo::setNodeNum(int nodeNum)
{
    _nodeNum = nodeNum;
    _thisNode = new ClusterLocalNode(nodeNum);
}

int ClusterNodeInfo::getNumNodes() { return _numNodes; }
int ClusterNodeInfo::getNodeNum() { return _nodeNum; }
ClusterLocalNode *ClusterNodeInfo::getThisNodePE() { return _thisNode; }

void ClusterNodeInfo::setNetworkFinalizeFunc(networkFinalizeFunc _func)
{
    _networkFinalizeFunc = _func;
}
void ClusterNodeInfo::callNetworkFinalizeFunc()
{
    if (_networkFinalizeFunc)
    {
        _networkFinalizeFunc();
    }
}

void ClusterNodeInfo::setIdleLoopFunc(idleLoopFunc _func)
{
    _idleLoopFunc = _func;
}
void ClusterNodeInfo::callIdleLoopFunc()
{
    if (_idleLoopFunc)
    {
        _idleLoopFunc();
    }
}

}
}

