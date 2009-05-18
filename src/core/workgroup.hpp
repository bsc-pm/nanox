#ifndef _NANOS_WORK_GROUP
#define _NANOS_WORK_GROUP

#include <vector>
#include "atomic.hpp"

namespace nanos {

//TODO: Memory management and structures of this class need serious thinking...
class WorkGroup {
private:
      static Atomic<int> atomicSeed;

      typedef std::vector<WorkGroup *> ListOfWGs;
      ListOfWGs partOf;
      int id;

      Atomic<int>  components;
      Atomic<int>  phase_counter;
      
      void addToGroup (WorkGroup &parent);
      void exitWork (WorkGroup &work);
public:
      // constructors
      WorkGroup() : id(atomicSeed++),components(0), phase_counter(0) {  }
      // to do these two properly we would need to keep also the information of the components
      // TODO:copy constructor
      WorkGroup(const WorkGroup &wg);
      // TODO:assignment operator
      const WorkGroup & operator= (const WorkGroup &wg);
      
      // destructor
      virtual ~WorkGroup();
      
      void addWork(WorkGroup &wg);
      void sync();
      void waitCompletation();
      void done();
      int getId() const { return id; }
};

typedef WorkGroup WG;

};

#endif

