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

#ifndef _NANOS_OpenCL_PROFILER
#define _NANOS_OpenCL_PROFILER

#include "dbmanager_sqlite3.hpp"
#include <map>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
namespace nanos {

enum OCLP_Exception {
   CLP_WRONG_NUMBER_OF_DIMENSIONS = -1000,
   CLP_OPENCL_STANDARD_ERROR
};

enum OCLP_DBMError {
   CLP_DBM_SUCCESS    = 0,
   CLP_DBM_NOT_FOUND,
   CLP_DBM_ERROR
};

/**
 * @brief This class throw the OpenCL Profiler exception following the Nanos rules
 */
class OpenCLProfilerException {
public:
   OpenCLProfilerException(nanos::OCLP_Exception exception, cl_int clError = 0, char* errorString = NULL);
};

/**
 * @brief This class keep the useful data to obtain the best configuration for a given kernel
 */
class Execution {
private:
   unsigned char _ndims;
   // Represents the best one found so far
   unsigned int _localX;
   unsigned int _localY;
   unsigned int _localZ;
   cl_ulong _time;
   // Represents the profiling status
   int _nextX;
   int _nextY;
   int _nextZ;
   bool _finished;
   bool _loadedFromDB;
public:
   explicit Execution() :
      _ndims(0), _localX(0), _localY(0), _localZ(0), _time(0),
      _nextX(0), _nextY(0), _nextZ(0), _finished(false), _loadedFromDB(false) {}

   explicit Execution(unsigned int ndims) :
      _ndims(ndims), _localX(0), _localY(0), _localZ(0), _time(0),
      _nextX(0), _nextY(0), _nextZ(0), _finished(false), _loadedFromDB(false) {}

   explicit Execution(unsigned int ndims, unsigned int localX, unsigned int localY, unsigned int localZ, long long int time,
            unsigned int nextX, unsigned int nextY, unsigned int nextZ, bool finished, bool loadedFromDB) :
      _ndims(ndims), _localX(localX), _localY(localY), _localZ(localZ), _time(time), _nextX(nextX), _nextY(nextY), _nextZ(nextZ), _finished(finished), _loadedFromDB(loadedFromDB) {}

   unsigned char getNdims() const
   {
      return _ndims;
   }

   cl_ulong getTime() const
   {
      return _time;
   }

   bool operator<(const Execution& execution) const
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

   void setTime(cl_ulong time)
   {
      _time=time;
   }

   void setLocalX(unsigned int localX)
   {
      _localX=localX;
   }

   void setLocalY(unsigned int localY)
   {
      _localY=localY;
   }

   void setLocalZ(unsigned int localZ)
   {
      _localZ=localZ;
   }

   int getNextX() const
   {
      return _nextX;
   }

   void setNextX(int nextX)
   {
      _nextX=nextX;
   }

   int getNextY() const
   {
      return _nextY;
   }

   void setNextY(int nextY)
   {
      _nextY=nextY;
   }

   int getNextZ() const
   {
      return _nextZ;
   }

   void setNextZ(int nextZ)
   {
      _nextZ=nextZ;
   }

   void setNdims(unsigned char ndims)
   {
      _ndims=ndims;
   }

   bool isFinished() const
   {
      return _finished;
   }

   void setFinished(bool finished)
   {
      _finished=finished;
   }

   bool isLoadedFromDb() const
   {
      return _loadedFromDB;
   }

   void setLoadedFromDb(bool loadedFromDb)
   {
      _loadedFromDB = loadedFromDb;
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

/**
 * @brief This class manages database interaction for the OpenCL Profiler extension
 */
class OpenCLProfilerDbManager {
   SQLite3DbManager _dbManager;
   bool _created;
   bool _isExecutionSet;
   unsigned int _selectStmtNumber;
   unsigned int _insertStmtNumber;
   unsigned int _updateStmtNumber;
   unsigned int _tableStmtNumber;
   const std::string CL_PROFILING_DEFAULT_TABLE;
public:
   OpenCLProfilerDbManager() : _created(false), _isExecutionSet(false), CL_PROFILING_DEFAULT_TABLE("opencl_kernels") {
      _dbManager.openConnection("nanos_opencl_kernels.db");
      initialize();
   }

   OCLP_DBMError setKernelConfig(Dims &dims, Execution &execution, std::string &kernelName);

   OCLP_DBMError getKernelConfig(Dims &dims, Execution &execution, std::string &kernelName);

private:
   void initialize();
   uint32_t getKernelHash(const Dims &dims, const std::string kernelName);
};

/**
 * @brief Keeps the device performance information
 */
class DevPerfInfo {
   std::string _devName;
   size_t _multiplePreferred;
   size_t _maxWorkGroup;
   bool _initialized;
public:
   DevPerfInfo() : _initialized(false) { }

   const std::string& getDevName() const
   {
      return _devName;
   }

   void setDevName(const std::string& devName)
   {
      _devName = devName;
      _initialized = true;
   }

   size_t getMaxWorkGroup() const
   {
      return _maxWorkGroup;
   }

   void setMaxWorkGroup(size_t maxWorkGroup)
   {
      _maxWorkGroup = maxWorkGroup;
   }

   size_t getMultiplePreferred() const
   {
      return _multiplePreferred;
   }

   void setMultiplePreferred(size_t multiplePreferred)
   {
      _multiplePreferred = multiplePreferred;
   }

   bool isInitialized() const
   {
      return _initialized;
   }
};

typedef std::map<Dims, Execution> DimsBest;
typedef std::map<Dims, unsigned long> DimsExecutions;

} // namespace nanos

#endif /* _NANOS_OpenCL_PROFILER */
