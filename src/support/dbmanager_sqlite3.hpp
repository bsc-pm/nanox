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

#ifndef SQLITE3DBMANAGER_HPP_
#define SQLITE3DBMANAGER_HPP_

#include <sqlite3.h>

#include <string>
#include <vector>
#include "dbmanager.hpp"

namespace nanos {

class SQLite3DbManager : public DbManager {
   bool _isOpen;
   sqlite3 *_db;
   std::vector <sqlite3_stmt*> _stmtVector;
public:
   SQLite3DbManager() : _isOpen(false), _db(NULL)  {}

   ~SQLite3DbManager();

   /**
    * @brief Open a new connection to databaseName database. If any connection
    * is already opened, it will be closed.
    */
   bool openConnection(const std::string &databaseName);

   /**
    * @brief Open a new connection to defaultDbName (nanos.db) database. If any connection
    * is already opened, it will be closed.
    */
   bool openConnection();

   /**
    * @brief Closes the database connection
    */
   bool closeConnection();

   /**
    * @brief This function pre-compile and statement to be used later
    * @param stmt statement to be pre-compile
    * @return The number of the statement prepared
    */
   unsigned int prepareStmt(const std::string &stmt);

   /**
    * @brief This function bind an integer value to a prepared statement
    * @param stmtNumber Parameter to reference the prepared statement
    * @param parameterIndex Parameter to choose the parameter to reference
    * @param value value to be set on the parameter
    */
   void bindIntParameter(const unsigned int stmtNumber, const unsigned int parameterIndex, int value);

   /**
    * @brief This function bind an integer64 value to a prepared statement
    * @param stmtNumber Parameter to reference the prepared statement
    * @param parameterIndex Parameter to choose the parameter to reference
    * @param value value to be set on the parameter
    */
   void bindInt64Parameter(const unsigned int stmtNumber, const unsigned int parameterIndex, long long int value);

   /**
    * @brief This function return the value of a given column
    * @param stmtNumber stmtNumber Parameter to reference the prepared statement
    * @param columnIndex Parameter to choose the column to reference
    * @return The value of a given column
    */
   int getIntColumnValue(const unsigned int stmtNumber, const unsigned int columnIndex);

   /**
    * @brief This function makes to ask for a row with the according statement
    * @param stmtNumber stmtNumber Parameter to reference the prepared statement
    */
   bool doStep(const unsigned int stmtNumber);

private:
   /**
    * @brief SQLite3 error check function
    */
   void sqlCheck(const int err, const std::string &msg);

   /**
    * @brief Clean the prepared statement vector
    */
   void cleanPreparedStmts();
};

} // namespace nanos

#endif /* SQLITE3DBMANAGER_HPP_ */
