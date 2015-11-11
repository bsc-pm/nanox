/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_OpenCL_PROFILER
#define _NANOS_OpenCL_PROFILER

#include "dbmanager.hpp"
#include <map>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
namespace nanos {



enum OpenCLProfilerExceptions {
   CLP_WRONG_NUMBER_OF_DIMENSIONS = -1000,
   CLP_OPENCL_STANDARD_ERROR
};

/**
 * @brief This class throw the OpenCL Profiler exception following the Nanos rules
 */
class OpenCLProfilerException {
public:
   OpenCLProfilerException(nanos::OpenCLProfilerExceptions exception, cl_int clError = 0, char* errorString = NULL);
};

/**
 * @brief This class keep the useful data to obtain the best configuration for a given kernel
 */
class Execution {
private:
   const unsigned char _ndims;
   const unsigned int _localX;
   const unsigned int _localY;
   const unsigned int _localZ;
   const cl_ulong _time;

public:
   Execution(unsigned int ndims, unsigned int localX, unsigned int localY, unsigned int localZ, long long int time) :
      _ndims(ndims), _localX(localX), _localY(localY), _localZ(localZ), _time(time) {}
   unsigned char getNdims() const
   {
      return _ndims;
   }

   cl_ulong getTime() const
   {
      return _time;
   }

   bool operator<(const Execution& execution)
   {
      return _time<execution.getTime();
   }

   unsigned int getLocalX() const
   {
      return _localX;
   }

   unsigned int getLocalY() const
   {
      return _localY;
   }

   unsigned int getLocalZ() const
   {
      return _localZ;
   }
};

/**
 * @brief This class storage the global dimensions of the range
 */
class Dims {
private:
   const unsigned char _ndims;
   const unsigned long long int _globalX;
   const unsigned long long int _globalY;
   const unsigned long long int _globalZ;
   const double _cost;
public:
   Dims(unsigned long long int ndims, unsigned long long int globalX, unsigned long long int globalY, unsigned long long int globalZ, double cost) :
      _ndims(ndims), _globalX(globalX), _globalY(globalY), _globalZ(globalZ), _cost(cost) {}

   unsigned int getGlobalX() const
   {
      return _globalX;
   }

   unsigned long long int getGlobalY() const
   {
      return _globalY;
   }

   unsigned long long int getGlobalZ() const
   {
      return _globalZ;
   }

   unsigned char getNdims() const
   {
      return _ndims;
   }

   bool operator<(const Dims& dims) const
   {
      if ( dims.getNdims()!=getNdims() ) {
         throw;
         // TODO throw nanos fatal error
      }
      switch (dims.getNdims()) {
         case 1:
            return (getGlobalX()<dims.getGlobalX());
            break;
         case 2:
            if ( getGlobalX()<dims.getGlobalX() )
               return true;
            else {
               if ( getGlobalX()==dims.getGlobalX()
                        &&getGlobalY()<dims.getGlobalY() )
                  return true;
            }
            return false;
            break;
         case 3:
            if ( getGlobalX()<dims.getGlobalX() )
               return true;
            else {
               if ( getGlobalX()==dims.getGlobalX()
                        &&getGlobalY()<dims.getGlobalY() )
                  return true;
               else {
                  if ( getGlobalX()==dims.getGlobalX()
                           &&getGlobalY()==dims.getGlobalY()
                           &&getGlobalZ()<dims.getGlobalZ() )
                     return true;
               }
            }
            return false;
            break;
         default:
            return true;
      }
   }

   double getCost() const
   {
      return _cost;
   }
};

class OpenCLProfilerDbManager {
   DbManager _dbManager;
   Execution *_execution;
   bool _created;
   bool _isExecutionSet;
   unsigned int _selectStmtNumber;
   unsigned int _insertStmtNumber;
   unsigned int _tableStmtNumber;
   const std::string CL_PROFILING_DEFAULT_TABLE;
public:
   OpenCLProfilerDbManager() : _execution(NULL), _created(false), _isExecutionSet(false), CL_PROFILING_DEFAULT_TABLE("opencl_kernels") {
      _dbManager.openConnection("nanos_opencl_kernels.db");
      initialize();
   }

   ~OpenCLProfilerDbManager();

   void setKernelConfig(Dims &dims, Execution &execution, std::string kernelName);
   Execution* getKernelConfig(Dims &dims, std::string kernelName);

   void setExecution(Execution* execution) {
      if ( _isExecutionSet )
         delete _execution;
      _execution = execution;
      _isExecutionSet = true;
   }

private:
   void initialize();
   uint32_t getKernelHash(const Dims &dims, const std::string kernelName);
   void destroyExecution();

   const Execution* getExecution() const {
      return _execution;
   }

};

typedef std::map<Dims, Execution*> DimsBest;
typedef std::map<Dims, ulong> DimsExecutions;

}

#endif /* _NANOS_OpenCL_PROFILER */
