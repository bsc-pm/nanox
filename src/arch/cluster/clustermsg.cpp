
#include "clustermsg.hpp"

#if 0
ClusterMessaging::sendMsgFinish(ClusterRemoteNode dest)
{
    if (gasnet_AMRequestShort0(dest->getClusterID, 203) != GASNET_OK)
    {
        fprintf(stderr, "Error sending a message to node %d\n", dest);
    }
}
#endif
using namespace nanos;
using namespace ext;

sendFinishMessageFunc ClusterMsg::_f = NULL;

void ClusterMsg::setSendFinishMessageFunc(sendFinishMessageFunc func)
{
    _f = func;
}

void ClusterMsg::sendFinishMessage(ClusterRemoteNode *dest)
{
    if (_f == NULL)
    {
        fprintf(stderr, "can not call to send msg finish!\n");
    }
    else
    _f(dest);
}

ClusterMsg msg;
