#ifndef _NANOS_WORK_GROUP
#define _NANOS_WORK_GROUP

#include <vector>
#include "atomic.hpp"

namespace nanos {

//TODO: Memory management and structures of this class need serious thinking...
class WorkGroup {
private:
      typedef std::vector<WorkGroup *> ListOfWGs;
      ListOfWGs partOf;

      Atomic<int>  components;
      Atomic<int>  phase_counter;
      
      void addToGroup (WorkGroup &parent);
      void exitWork (WorkGroup &work);
public:
      // constructors
      WorkGroup() : components(0), phase_counter(0) {  }
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
};

typedef WorkGroup WG;

};

#endif