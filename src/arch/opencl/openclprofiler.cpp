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

#include "openclprofiler.hpp"
#include <string>
#include <sstream>
#include "debug.hpp"
#include "openclutils.hpp"

using namespace nanos;

OpenCLProfilerException::OpenCLProfilerException(nanos::OpenCLProfilerExceptions exception, cl_int clError, char* errorString)
{
   std::string baseMsg = "[OpenCL Profiling Error] : ";
   std::stringstream errorBuffer;
   switch ( exception ) {
      case CLP_WRONG_NUMBER_OF_DIMENSIONS:
         fatal0( baseMsg + "Wrong number of dimensions");
         break;
      case CLP_OPENCL_STANDARD_ERROR:
         errorBuffer << clError;
         fatal0( baseMsg + errorBuffer.str() + ", " + errorString);
         break;
      default:
         fatal0( baseMsg + "Generic" );
         break;
   }
}

uint32_t OpenCLProfilerDbManager::getKernelHash(const Dims &dims, const std::string kernelName)
{
   // Hash generation based on kernel name and dimensions
   std::stringstream fullString;
   fullString << kernelName;
   switch ( dims.getNdims() ) {
      case 1:
         fullString << dims.getGlobalX();
         break;
      case 2:
         fullString << dims.getGlobalX()+dims.getGlobalY();
         break;
      case 3:
         fullString << dims.getGlobalX()+dims.getGlobalY()+dims.getGlobalZ();
         break;
      default:
         break;
   }
   return nanos::ext::gnuHash(fullString.str().c_str());
}

void OpenCLProfilerDbManager::setKernelConfig(Dims &dims, Execution &execution, std::string kernelName)
{
   _dbManager.bindIntParameter(_insertStmtNumber, 1, getKernelHash(dims, kernelName));   // Hash
   _dbManager.bindIntParameter(_insertStmtNumber, 2, static_cast<int>(dims.getNdims())); // Dims
   _dbManager.bindIntParameter(_insertStmtNumber, 3, execution.getLocalX());             // X
   _dbManager.bindIntParameter(_insertStmtNumber, 4, execution.getLocalY());             // Y
   _dbManager.bindIntParameter(_insertStmtNumber, 5, execution.getLocalZ());             // Z

   _dbManager.doStep(_insertStmtNumber);

   _created = true;
}

Execution* OpenCLProfilerDbManager::getKernelConfig(Dims &dims, std::string kernelName)
{
   destroyExecution();

   _dbManager.bindIntParameter(_selectStmtNumber, 1, getKernelHash(dims, kernelName));

   if ( _dbManager.doStep(_selectStmtNumber) ) {
      setExecution(new Execution(
               dims.getNdims(),
               _dbManager.getIntColumnValue(_selectStmtNumber, 0),
               _dbManager.getIntColumnValue(_selectStmtNumber, 1),
               _dbManager.getIntColumnValue(_selectStmtNumber, 2),
               0));
   } else {
      // No configurations found
      setExecution(new Execution(9,0,0,0,0));
   }

   return _execution;
}

OpenCLProfilerDbManager::~OpenCLProfilerDbManager()
{
   destroyExecution();
}

void OpenCLProfilerDbManager::destroyExecution()
{
   if ( _isExecutionSet ) {
      delete _execution;
      _isExecutionSet = false;
      _execution = NULL;
   }
}

void OpenCLProfilerDbManager::initialize()
{
   const std::string tableSql  = "CREATE TABLE IF NOT EXISTS opencl_kernels(hash INT, dims INT, x INT, y INT, z INT, PRIMARY KEY(hash));";
   const std::string selectSql = "SELECT x,y,z FROM " + CL_PROFILING_DEFAULT_TABLE + " WHERE hash = @hash";
   const std::string insertSql = "INSERT INTO " + CL_PROFILING_DEFAULT_TABLE + " VALUES(@hash, @dims, @x, @y, @z)";

   // Create table if not exists
   _tableStmtNumber = _dbManager.prepareStmt(tableSql);
   _dbManager.doStep(_tableStmtNumber);

   _selectStmtNumber = _dbManager.prepareStmt(selectSql);
   _insertStmtNumber = _dbManager.prepareStmt(insertSql);
}
