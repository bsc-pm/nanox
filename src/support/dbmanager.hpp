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

#ifndef DBMANAGER_HPP_
#define DBMANAGER_HPP_

#include <string>
#include <vector>
#include "debug.hpp"

namespace nanos {

const std::string defaultDbName = "nanos.db";

class DbManager {
public:
   DbManager() {};

   virtual ~DbManager() {};

   /**
    * @brief Open a new connection to databaseName database. If any connection
    * is already opened, it will be closed.
    */
   virtual bool openConnection(const std::string &databaseName) = 0;

   /**
    * @brief Open a new connection to defaultDbName (nanos.db) database. If any connection
    * is already opened, it will be closed.
    */
   virtual bool openConnection() = 0;

   /**
    * @brief Closes the database connection
    */
   virtual bool closeConnection() = 0;

   /**
    * @brief This function pre-compile the statement to be used later
    * @param stmt statement to be pre-compile
    * @return The number of the statement prepared
    */
   virtual unsigned int prepareStmt(const std::string &stmt) = 0;

   /**
    * @brief This function bind an integer value to a prepared statement
    * @param stmtNumber Parameter to reference the prepared statement
    * @param parameterIndex Parameter to choose the parameter to reference
    * @param value value to be set on the parameter
    */
   virtual void bindIntParameter(const unsigned int stmtNumber, const unsigned int parameterIndex, int value) { fatal0("DbManager: bindIntParameter not implemented") };

   /**
    * @brief This function bind an integer64 value to a prepared statement
    * @param stmtNumber Parameter to reference the prepared statement
    * @param parameterIndex Parameter to choose the parameter to reference
    * @param value value to be set on the parameter
    */
   virtual void bindInt64Parameter(const unsigned int stmtNumber, const unsigned int parameterIndex, long long int value) { fatal0("DbManager: bindInt64Parameter not implemented") };

   /**
    * @brief This function return the value of a given column
    * @param stmtNumber stmtNumber Parameter to reference the prepared statement
    * @param columnIndex Parameter to choose the column to reference
    * @return The value of a given column
    */
   virtual int getIntColumnValue(const unsigned int stmtNumber, const unsigned int columnIndex) { fatal0("DbManager: getIntColumnValue not implemented") };

   /**
    * @brief This function makes to ask for a row with the according statement
    * @param stmtNumber stmtNumber Parameter to reference the prepared statement
    */
   virtual bool doStep(const unsigned int stmtNumber) { fatal0("DbManager: doStep not implemented") };

private:
   /**
    * @brief Error check function
    */
   virtual void sqlCheck(const int err, const std::string &msg) = 0;

   /**
    * @brief Clean the prepared statement vector
    */
   virtual void cleanPreparedStmts() = 0;
};

} // namespace nanos

#endif /* DBMANAGER_HPP_ */
