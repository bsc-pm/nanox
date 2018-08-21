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

#include "dbmanager.hpp"

namespace nanos {

class SQLite3DbManager : public DbManager {
public:
   SQLite3DbManager();
   bool openConnection(const std::string &databaseName);
   bool openConnection();
   bool closeConnection();
   unsigned int prepareStmt(const std::string &stmt);
   void bindIntParameter(const unsigned int stmtNumber, const unsigned int parameterIndex, int value);
   void bindInt64Parameter(const unsigned int stmtNumber, const unsigned int parameterIndex, long long int value);
   int getIntColumnValue(const unsigned int stmtNumber, const unsigned int columnIndex);
   bool doStep(const unsigned int stmtNumber);
private:
   void sqlCheck(const int err, const std::string &msg);
   void cleanPreparedStmts();
};

} // namespace nanos

#endif /* SQLITE3DBMANAGER_HPP_ */
