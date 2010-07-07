#ifndef _CLUSTER_MESSAGING
#define _CLUSTER_MESSAGING

#include "clusternode.hpp"

namespace nanos {
namespace ext {

typedef void ( *sendFinishMessageFunc ) ( ClusterRemoteNode *dest );

class ClusterMsg
{
    private:
        static sendFinishMessageFunc _f;
    public:
        static void sendFinishMessage(ClusterRemoteNode *dest);
        static void setSendFinishMessageFunc(sendFinishMessageFunc func);
};

}
}

extern nanos::ext::ClusterMsg clusterMsg;

#endif
