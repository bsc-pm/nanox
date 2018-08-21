/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/


#ifndef _NANOX_NETWORK
#define _NANOX_NETWORK

#include <string>
#include <set>
#include <list>
#include <vector>
#include "functor_decl.hpp"
#include "globalregt_decl.hpp"
#include "requestqueue_decl.hpp"
#include "networkapi.hpp"
#include "packer_decl.hpp"
#include "deviceops_decl.hpp"
#include "debug.hpp"

namespace nanos {

   class SendDataRequest {
      protected:
         NetworkAPI *_api;
         unsigned int _issueNode;
         unsigned int _seqNumber;
         void *_origAddr;
         void *_destAddr;
         std::size_t _len;
         std::size_t _count;
         std::size_t _ld;
         unsigned int _destination;
         unsigned int _wdId;
         void *_hostObject;
         reg_t _hostRegId;
         unsigned int _metaSeq;
      public:
         SendDataRequest( NetworkAPI *api, unsigned int issueNode, unsigned int seqNumber, void *origAddr, void *destAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int dst, unsigned int wdId, void *hostObject, reg_t hostRegId, unsigned int metaSeq );
         virtual ~SendDataRequest();
         void doSend();
         void *getOrigAddr() const;
         unsigned int getDestination() const;
         unsigned int getIssueNode() const;
         unsigned int getWdId() const;
         unsigned int getSeqNumber() const;
         virtual void doSingleChunk() = 0;
         virtual void doStrided( void *localAddr ) = 0;
   };

   struct GetRequest {
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
      int _complete;
#else
      volatile int _complete;
#endif
      char* _hostAddr;
      std::size_t _size;
      char* _recvAddr;
      DeviceOps *_ops;

      GetRequest( char* hostAddr, std::size_t size, char *recvAddr, DeviceOps *ops );
      virtual ~GetRequest();

      void complete();
      bool isCompleted() const;
      virtual void clear();
   };

   struct GetRequestStrided : public GetRequest {
      std::size_t _count;
      std::size_t _ld;
      Packer *_packer;

      GetRequestStrided( char* hostAddr, std::size_t size, std::size_t count, std::size_t ld, char *recvAddr, DeviceOps *ops, Packer *packer );
      virtual ~GetRequestStrided();

      virtual void clear();
   };

   class RegionsForwarded {
      std::map< const_reg_key_t, std::set< reg_t > > _regions;

      public:

      bool isRegionForwarded( global_reg_t const &reg ) const {
         bool result = false;
         std::map< const_reg_key_t, std::set< reg_t > >::const_iterator it = _regions.find( reg.key );
         if ( it != _regions.end() ) {
            result = ( it->second.end() != it->second.find( reg.id ) );
         }
         return result;
      }

      void addForwardedRegion( global_reg_t const &reg ) {
         std::map< const_reg_key_t, std::set< reg_t > >::iterator it = _regions.lower_bound( reg.key );
         if ( it == _regions.end() || _regions.key_comp()( reg.key, it->first ) ) {
            it = _regions.insert( it, std::map< const_reg_key_t, std::set< reg_t > >::value_type( reg.key, std::set< reg_t >() ) );
         }
         std::set< reg_t >::iterator sit = it->second.lower_bound( reg.id );
         if ( sit == it->second.end() || it->second.key_comp()( reg.id, *sit ) ) {
            sit = it->second.insert( sit, reg.id );
         }
      }

      void removeForwardedRegion( global_reg_t const &reg ) {
         std::map< const_reg_key_t, std::set< reg_t > >::iterator it = _regions.lower_bound( reg.key );
         if ( it == _regions.end() || _regions.key_comp()( reg.key, it->first ) ) {
            //it can happen: fatal("Region not found @ _regions (by reg.key).");
         } else {
            _regions.erase( it );
         }
      }
   };


   class Network
   {
      private:
         unsigned int _numNodes;
         NetworkAPI *_api; 
         unsigned int _nodeNum;

         //std::string _masterHostname;
         char * _masterHostname;
         bool _checkForDataInOtherAddressSpaces;
         Atomic<unsigned int> *_putRequestSequence;

         class ReceivedWDData {
            private:
               struct recvDataInfo {
                  recvDataInfo() : _wd( NULL ), _count( 0 ), _expected( 0 ) { }
                  WorkDescriptor *_wd;
                  std::size_t _count;
                  std::size_t _expected;
               };
               std::map< unsigned int, struct recvDataInfo > _recvWdDataMap;
               Lock _lock;
               Atomic<unsigned int> _receivedWDs;
            public:
               ReceivedWDData();
               ~ReceivedWDData();
               void addData( unsigned int wdId, std::size_t size, WD *parent );
               void addWD( unsigned int wdId, WorkDescriptor *wd, std::size_t expectedData, WD *parent );
               unsigned int getReceivedWDsCount() const;
         };

         class SentWDData {
            private:
               std::map< unsigned int, std::size_t > _sentWdDataMap;
               Lock _lock;
            public:
               SentWDData();
               ~SentWDData();
               void addSentData( unsigned int wdId, std::size_t sentData );
               std::size_t getSentData( unsigned int wdId );
         };

         ReceivedWDData _recvWdData;
         SentWDData _sentWdData;

         std::list< std::pair< unsigned int, std::pair<WD *, std::size_t> > > _deferredWorkReqs;
         Lock _deferredWorkReqsLock;
         Atomic<unsigned int> _recvSeqN;

         void checkDeferredWorkReqs();

         Lock _waitingPutRequestsLock;
         std::set< void * > _waitingPutRequests;
         std::set< void * > _receivedUnmatchedPutRequests;

         struct UnorderedRequest {
            SendDataRequest *_req;
            void            *_addr;
            unsigned int     _seqNumber;
            UnorderedRequest( SendDataRequest *req ) : _req( req ), _addr( (void *) 0 ), _seqNumber( 0 ) {}
            UnorderedRequest( void *addr, unsigned int seqNum ) : _req( (SendDataRequest *) 0 ), _addr( addr ), _seqNumber( seqNum ) {}
            UnorderedRequest( UnorderedRequest const &r ) : _req( r._req ), _addr( r._addr ), _seqNumber( r._seqNumber ) {}
            UnorderedRequest &operator=( UnorderedRequest const &r ) { _req = r._req; _addr = r._addr; _seqNumber = r._seqNumber; return *this; }
         };
         std::list< UnorderedRequest > _delayedBySeqNumberPutReqs;
         Lock _delayedBySeqNumberPutReqsLock;


         RegionsForwarded *_forwardedRegions;
         int _gpuPresend;
         int _smpPresend;
         Atomic<unsigned int> *_metadataSequenceNumbers;
         Atomic<unsigned int> _recvMetadataSeq;

         class SyncWDs {
            unsigned int _numWDs;
            WD **_wds;
            void *_addr;
            public:
            SyncWDs( int num, WD **wds, void *addr ) : _numWDs( num ), _addr( addr ) {
               _wds = NEW WD*[num];
               for ( unsigned int idx = 0; idx < _numWDs; idx += 1 ) {
                  _wds[idx] = wds[idx];
               }
            }
            SyncWDs( SyncWDs const &s ) : _numWDs( s._numWDs ), _addr( s._addr ) {
               _wds = NEW WD*[_numWDs];
               for ( unsigned int idx = 0; idx < _numWDs; idx += 1 ) {
                  _wds[idx] = s._wds[idx];
               }
            }
            SyncWDs &operator=( SyncWDs const &s ) {
               this->_numWDs = s._numWDs;
               this->_addr = s._addr;
               this->_wds = NEW WD*[this->_numWDs];
               for ( unsigned int idx = 0; idx < this->_numWDs; idx += 1 ) {
                  this->_wds[idx] = s._wds[idx];
               }
               return *this;
            }
            ~SyncWDs() { delete[] _wds; }
            unsigned int getNumWDs() const { return _numWDs; }
            WD **getWDs() const { return _wds; }
            void *getAddr() const { return _addr; }
         };
         std::list<SyncWDs> _syncReqs;
         RecursiveLock _syncReqsLock;

      public:
         static const unsigned int MASTER_NODE_NUM = 0;
         typedef struct {
            int complete;
            void * resultAddr;
         } mallocWaitObj;


         std::list< SendDataRequest * > _delayedPutReqs;
         Lock _delayedPutReqsLock;


         RequestQueue< SendDataRequest > _dataSendRequests;
         int _nodeBarrierCounter;
         WD *_parentWD;

         // constructor

         Network ();
         ~Network ();

         void setAPI ( NetworkAPI *api );
         NetworkAPI *getAPI ();
         void setNumNodes ( unsigned int numNodes );
         unsigned int getNumNodes () const;
         void setNodeNum ( unsigned int nodeNum );
         unsigned int getNodeNum () const;
         void notifyWorkDone ( unsigned int nodeNum, void *remoteWdAddr, int peId );
         void notifyMalloc ( unsigned int nodeNum, void * result, mallocWaitObj *waitObjAddr );

         void initialize ( NetworkAPI *api );
         void finalize ( void );
         void finalizeNoBarrier ( void );
         void poll ( unsigned int id );
         void sendExitMsg( unsigned int nodeNum );
         void sendWorkMsg( unsigned int dest, WorkDescriptor const &wd );
         bool isWorking( unsigned int dest, unsigned int numPe ) const;
         void sendWorkDoneMsg( unsigned int nodeNum, void const *remoteWdaddr );
         void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size, unsigned int wdId, WD const *wd, void *hostObject, reg_t hostRegId );
         void putStrided1D ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, void *localPack, std::size_t size, std::size_t count, std::size_t ld, unsigned int wdId, WD const *wd, void *hostObject, reg_t hostRegId );
         void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, std::size_t size, GetRequest *req, void *hostObject, reg_t hostRegId );
         void getStrided1D ( void *packedAddr, unsigned int remoteNode, uint64_t remoteTag, uint64_t remoteAddr, std::size_t size, std::size_t count, std::size_t ld, GetRequestStrided *req, void *hostObject, reg_t hostRegId );
         void *malloc ( unsigned int remoteNode, std::size_t size );
         void memFree ( unsigned int remoteNode, void *addr );
         void memRealloc ( unsigned int remoteNode, void *oldAddr, std::size_t oldSize, void *newAddr, std::size_t newSize );
         void nodeBarrier( void );
         //void sendRegionMetadata( unsigned int dest, CopyData *cd );

         void setMasterHostname( char *name );
         //const std::string & getMasterHostname( void ) const;
         const char * getMasterHostname( void ) const;
         void sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, unsigned int wdId, WD const *wd, void *hostObject, reg_t hostRegId );
         void sendRequestPutStrided1D( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int wdId, WD const *wd, void *hostObject, reg_t hostRegId );
         std::size_t getTotalBytes();
         void mallocSlaves ( void **addresses, std::size_t size );

         static Lock _nodeLock;
         static Atomic<uint64_t> _nodeRCAaddr;
         static Atomic<uint64_t> _nodeRCAaddrOther;

         void enableCheckingForDataInOtherAddressSpaces();
         bool doIHaveToCheckForDataInOtherAddressSpaces() const;

         SimpleAllocator *getPackerAllocator() const;
         std::size_t getMaxGetStridedLen() const;

         void *allocateReceiveMemory( std::size_t len );
         void freeReceiveMemory( void * addr );

         void notifyWork( std::size_t expectedData, WD *delayedWD, unsigned int delayedSeq);
         void notifyPut( unsigned int from, unsigned int wdId, std::size_t len, std::size_t count, std::size_t ld, uint64_t realTag, void *hostObject, reg_t hostRegId, unsigned int metaSeq );
         void notifyWaitRequestPut( void *addr, unsigned int wdId, unsigned int seqNumber );
         void notifyRequestPut( SendDataRequest *req );
         void notifyGet( SendDataRequest *req );
         void notifyRegionMetaData( CopyData *cd, unsigned int seq );
         void notifySynchronizeDirectory( unsigned int numWDs, WorkDescriptor **wds, void *addr );
         void notifyIdle( unsigned int node );
         void invalidateDataFromDevice( uint64_t addr, std::size_t len, std::size_t count, std::size_t ld, void *hostObject, reg_t hostRegId );
         void getDataFromDevice( uint64_t addr, std::size_t len, std::size_t count, std::size_t ld, void *hostObject, reg_t hostRegId );
         unsigned int getPutRequestSequenceNumber( unsigned int dest );
         unsigned int checkPutRequestSequenceNumber( unsigned int dest ) const;
         bool updatePutRequestSequenceNumber( unsigned int dest, unsigned int value );
         void processWaitRequestPut( void *addr, unsigned int seqNumber );
         void processRequestsDelayedBySeqNumber();
         void processSendDataRequest( SendDataRequest *req ) ;
         void setGpuPresend(int p);
         void setSmpPresend(int p);
         int getGpuPresend() const;
         int getSmpPresend() const;
         void deleteDirectoryObject( GlobalRegionDictionary const *obj );
         unsigned int getMetadataSequenceNumber( unsigned int dest );
         unsigned int checkMetadataSequenceNumber( unsigned int dest );
         unsigned int updateMetadataSequenceNumber( unsigned int value );
         void synchronizeDirectory( void *addr );
         void broadcastIdle();
         void processSyncRequests();
         void setParentWD(WD *wd);
   };

} // namespace nanos

#endif
