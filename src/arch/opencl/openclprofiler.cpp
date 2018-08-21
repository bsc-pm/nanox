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

#include "openclprofiler.hpp"
#include <string>
#include <sstream>
#include "debug.hpp"
#include "openclutils.hpp"

using namespace nanos;

OpenCLProfilerException::OpenCLProfilerException(OCLP_Exception exception, cl_int clError, char* errorString)
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

OCLP_DBMError OpenCLProfilerDbManager::setKernelConfig(Dims &dims, Execution &execution, std::string &kernelName)
{
   try {
      if ( !execution.isLoadedFromDb() ) {
         debug(" [OpenCL][Profiling] Inserting in DB >>>");

         _dbManager.bindIntParameter(_insertStmtNumber, 1, getKernelHash(dims, kernelName));   // Hash
         _dbManager.bindIntParameter(_insertStmtNumber, 2, static_cast<int>(dims.getNdims())); // Dims
         _dbManager.bindIntParameter(_insertStmtNumber, 3, execution.getLocalX());             // X
         _dbManager.bindIntParameter(_insertStmtNumber, 4, execution.getLocalY());             // Y
         _dbManager.bindIntParameter(_insertStmtNumber, 5, execution.getLocalZ());             // Z
         _dbManager.bindInt64Parameter(_insertStmtNumber, 6, execution.getTime());             // Time
         if ( !execution.isFinished() ) {
            _dbManager.bindIntParameter(_insertStmtNumber, 7, execution.getNextX());           // next X
            _dbManager.bindIntParameter(_insertStmtNumber, 8, execution.getNextY());           // next Y
            _dbManager.bindIntParameter(_insertStmtNumber, 9, execution.getNextZ());           // next Z
         } else {
            _dbManager.bindIntParameter(_insertStmtNumber, 7, -1);                             // Finished
            _dbManager.bindIntParameter(_insertStmtNumber, 8, -1);                             // Finished
            _dbManager.bindIntParameter(_insertStmtNumber, 9, -1);                             // Finished
         }

         _dbManager.doStep(_insertStmtNumber);

         debug(" [OpenCL][Profiling] Inserting in DB <<<");
      } else {
         debug(" [OpenCL][Profiling] Updating in DB >>>");

         _dbManager.bindIntParameter(_updateStmtNumber, 1, execution.getLocalX());             // X
         _dbManager.bindIntParameter(_updateStmtNumber, 2, execution.getLocalY());             // Y
         _dbManager.bindIntParameter(_updateStmtNumber, 3, execution.getLocalZ());             // Z
         _dbManager.bindInt64Parameter(_updateStmtNumber, 4, execution.getTime());             // Time
         if ( !execution.isFinished() ) {
            _dbManager.bindIntParameter(_updateStmtNumber, 5, execution.getNextX());           // next X
            _dbManager.bindIntParameter(_updateStmtNumber, 6, execution.getNextY());           // next Y
            _dbManager.bindIntParameter(_updateStmtNumber, 7, execution.getNextZ());           // next Z
         } else {
            _dbManager.bindIntParameter(_updateStmtNumber, 5, -1);                             // next X
            _dbManager.bindIntParameter(_updateStmtNumber, 6, -1);                             // next Y
            _dbManager.bindIntParameter(_updateStmtNumber, 7, -1);                             // next Z
         }
         _dbManager.bindIntParameter(_updateStmtNumber, 8, getKernelHash(dims, kernelName));

         _dbManager.doStep(_updateStmtNumber);

         debug(" [OpenCL][Profiling] Updating in DB <<<");
      }
      _created = true;
      return CLP_DBM_SUCCESS;
   } catch (...) {
      return CLP_DBM_ERROR;
   }
}

OCLP_DBMError OpenCLProfilerDbManager::getKernelConfig(Dims &dims, Execution &execution, std::string &kernelName)
{
   _dbManager.bindIntParameter(_selectStmtNumber, 1, getKernelHash(dims, kernelName));

   if ( _dbManager.doStep(_selectStmtNumber) ) {
      try {
         execution.setLocalX(_dbManager.getIntColumnValue(_selectStmtNumber, 0));
         execution.setLocalY(_dbManager.getIntColumnValue(_selectStmtNumber, 1));
         execution.setLocalZ(_dbManager.getIntColumnValue(_selectStmtNumber, 2));
         execution.setTime(_dbManager.getIntColumnValue(_selectStmtNumber, 3));
         execution.setNextX(_dbManager.getIntColumnValue(_selectStmtNumber, 4));
         execution.setNextY(_dbManager.getIntColumnValue(_selectStmtNumber, 5));
         execution.setNextZ(_dbManager.getIntColumnValue(_selectStmtNumber, 6));
         execution.setLoadedFromDb(true);
         if ( execution.getNextX() == -1 && execution.getNextY() == -1 && execution.getNextZ() == -1 )
            execution.setFinished(true);
         return CLP_DBM_SUCCESS;
      } catch (...) {
         return CLP_DBM_ERROR;
      }
   } else {
      return CLP_DBM_NOT_FOUND;
   }
}

void OpenCLProfilerDbManager::initialize()
{
   const std::string tableSql  = "CREATE TABLE IF NOT EXISTS opencl_kernels(hash INT, dims INT, x INT, y INT, z INT, time INT64, "
            "nextX INT, nextY INT, nextZ INT, PRIMARY KEY(hash));";
   const std::string selectSql = "SELECT x,y,z,time,nextX,nextY,nextZ FROM " + CL_PROFILING_DEFAULT_TABLE + " WHERE hash = @hash";
   const std::string insertSql = "INSERT INTO " + CL_PROFILING_DEFAULT_TABLE + " VALUES(@hash, @dims, @x, @y, @z, @time, @nextX, @nextY, @nextZ)";
   const std::string updateSql = "UPDATE " + CL_PROFILING_DEFAULT_TABLE + " SET x=@x, y=@y, z=@z, time=@time, nextX=@nextX, "
            "nextY=@nextY, nextZ=@nextZ WHERE hash = @hash";

   // Create table if not exists
   _tableStmtNumber = _dbManager.prepareStmt(tableSql);
   _dbManager.doStep(_tableStmtNumber);

   _selectStmtNumber = _dbManager.prepareStmt(selectSql);
   _insertStmtNumber = _dbManager.prepareStmt(insertSql);
   _updateStmtNumber = _dbManager.prepareStmt(updateSql);
}
