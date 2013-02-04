
#include "openclcache.hpp"
#include "openclconfig.hpp"
#include "openclprocessor.hpp"

using namespace nanos;
using namespace nanos::ext;

Lock OpenCLMemDump::_dumpLock;


//
// OpenCLMemDump implementation.
//

void OpenCLMemDump::dumpCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        size_t size) {
    LockBlock lock(OpenCLMemDump::_dumpLock);

    std::cerr << "### COPY-IN ###" << std::endl
            << "  Dst = " << localDst
            << std::endl
            << "  Src = " << reinterpret_cast<void *> (remoteSrc.getTag())
            << std::endl;
}

void OpenCLMemDump::dumpCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        size_t size) {
    LockBlock lock(OpenCLMemDump::_dumpLock);

    std::cerr << "### COPY-OUT ###" << std::endl
            << "  Dst = " << reinterpret_cast<void *> (remoteDst.getTag())
            << std::endl
            << "  Src = " << localSrc
            << std::endl;
}

//
// OpenCLCache implementation.
//

void OpenCLCache::initialize() {
    size_t pageSize = getPageSize();

    // Compute the amount of cache for the device.
    _devCacheSize = OpenCLConfig::getDevCacheSize();
    if (_devCacheSize == 0)
        _devCacheSize = _openclAdapter.getGlobalSize();
    _devCacheSize = roundUpToPageSize(_devCacheSize);

    // Initialize the device allocator. It manages virtual addresses above the
    // addresses managed by host allocator.
    _devAllocator.init(pageSize , _devCacheSize);
}

void *OpenCLCache::deviceAllocate(size_t size) {
    cl_mem buf;

    if (_openclAdapter.allocBuffer(size, buf) != CL_SUCCESS)
        fatal("Device allocation failed");

    void *addr = _devAllocator.allocate(size);

    if (!addr)
        fatal("Device cache allocation failed");

    _bufAddrMappings[addr] = buf;

    return addr;
}

void *OpenCLCache::deviceReallocate(void * addr, size_t size, size_t ceSize) {
    deviceFree(addr);

    return deviceAllocate(size);
}

void OpenCLCache::deviceFree(void * addr) {
    void *allocAddr = addr;
    _devAllocator.free(allocAddr);
    cl_mem buf = _bufAddrMappings[allocAddr];    
    cl_int errCode;
    errCode = _openclAdapter.freeBuffer(buf);
    if ( errCode != CL_SUCCESS)
        fatal("Cannot free device buffer.");
    
    _bufAddrMappings.erase(_bufAddrMappings.find(allocAddr)); 
}

bool OpenCLCache::deviceCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        size_t size) {
    cl_int errCode;    
    cl_mem buf = _bufAddrMappings[localDst];  
    
    errCode = _openclAdapter.writeBuffer(buf,
              (void*) remoteSrc.getTag(),
              0,
              size);
    if (errCode != CL_SUCCESS){
        fatal("Buffer writing failed.");
    }
    _bytesIn += ( unsigned int ) size;

    return true;
}

bool OpenCLCache::deviceCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        size_t size) {
    cl_int errCode;
    cl_mem buffer=_bufAddrMappings[localSrc];
    
    errCode = _openclAdapter.readBuffer(buffer,
                ((void*)remoteDst.getTag()),
                0,
                size);
    
    if (errCode != CL_SUCCESS) {
        fatal("Buffer reading failed.");
    }    
    
    _bytesOut += ( unsigned int ) size;

    return true;
}

//
// DMATransfer implementation.
//

namespace nanos {
    namespace ext {

        template <>
        void DMATransfer<void *, CopyDescriptor>::dump() const {
            std::cerr << *this;
        }

        template <>
        const uint64_t DMATransfer<void *, CopyDescriptor>::MaxId =
                std::numeric_limits<uint64_t>::max();

        template <>
        Atomic<uint64_t> DMATransfer<void *, CopyDescriptor>::_freeId = 0;

        template <>
        void DMATransfer<CopyDescriptor, void *>::dump() const {
            std::cerr << *this;
        }

        template <>
        const uint64_t DMATransfer<CopyDescriptor, void *>::MaxId =
                std::numeric_limits<uint64_t>::max();

        template <>
        Atomic<uint64_t> DMATransfer<CopyDescriptor, void *>::_freeId = 0;

    } // End namespace ext.
} // End namespace nanos.

//
// OpenCLDMA implementation.
//

bool OpenCLDMA::copyIn(void *localDst, CopyDescriptor &remoteSrc, size_t size) {
    LockBlock lock(_lock);
    
    _ins.push(DMAInTransfer(localDst, remoteSrc, size));

    return false;
}

bool OpenCLDMA::copyOut(CopyDescriptor &remoteDst, void *localSrc, size_t size) {
    LockBlock lock(_lock);

    _outs.push(DMAOutTransfer(remoteDst, localSrc, size));

    return false;
}

void OpenCLDMA::syncTransfer(uint64_t hostAddress) {
    LockBlock lock(_lock);

    _ins.prioritize(hostAddress);
    _outs.prioritize(hostAddress);
}

void OpenCLDMA::execTransfers() {
    LockBlock lock(_lock);

    // TODO: select a smarter policy.
    while (!_ins.empty()) {
        const DMAInTransfer &trans = _ins.top();
        CopyDescriptor &src = trans.getSource();

        _proc.copyIn(trans.getDestination(), src, trans.getSize());
        _proc.synchronize(src);

        _ins.pop();
    }

    // TODO: select a smarted policy.
    while (!_outs.empty()) {
        const DMAOutTransfer &trans = _outs.top();
        CopyDescriptor &dst = trans.getDestination();

        _proc.copyOut(dst, trans.getSource(), trans.getSize());
        _proc.synchronize(dst);

        _outs.pop();
    }
}

std::ostream &nanos::ext::operator<<(std::ostream &os, size_t size) {

    os << "### Size ###"
            << std::endl
            << "  RawSize: " 
            << std::endl
            << "  Operation: " << "DEVICE"
            << std::endl;

    return os;
}

std::ostream &
nanos::ext::operator<<(std::ostream &os,
        const DMATransfer<void *, CopyDescriptor> &trans) {
    const CopyDescriptor &src = trans.getSource();

    void *srcAddr = reinterpret_cast<void *> (src.getTag());
    unsigned version = src.getDirectoryVersion();

    os << "### DMA Input Transfer ###" << std::endl
            << "  To: " << trans.getDestination() << std::endl
            << "  From: [" << version << "] " << srcAddr << std::endl;

    return os;
}

std::ostream &
nanos::ext::operator<<(std::ostream &os,
        const DMATransfer<CopyDescriptor, void *> &trans) {
    const CopyDescriptor &dst = trans.getDestination();

    void *dstAddr = reinterpret_cast<void *> (dst.getTag());
    unsigned version = dst.getDirectoryVersion();

    os << "### DMA Output Transfer ###" << std::endl
            << "  To: [" << version << "] " << dstAddr << std::endl
            << "  From: " << trans.getSource() << std::endl;

    return os;
}
