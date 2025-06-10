# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring that clearly defines the scope and purpose of the `chap_core.database` package, outlining its role in managing database interactions, schemas, and sessions. (Primary task).
# 2. **Expose Key Components via `__all__`**: Identify core classes and functions from the submodules (e.g., `SessionWrapper` from `database.py`, main ORM base classes from `base_tables.py`, or key table models) that form the public API of this package. Import and list them in `__all__` for convenient access.
# 3. **Centralized Database Initialization/Setup**: Consider providing a utility function here (e.g., `get_engine()`, `create_db_session()`) if there's a standard way the application initializes its database connection or sessions, abstracting away the details from other parts of the application.
# 4. **Overview of Database Schema Modules**: The module docstring could briefly describe the different schema modules within this package (e.g., `dataset_tables.py`, `model_spec_tables.py`, `tables.py`) and the types of data they represent in the database.
# 5. **Database-Related Constants**: If there are any package-wide constants related to the database (e.g., default schema names, common string lengths for certain fields, specific database dialect options if applicable), this `__init__.py` could be a suitable place to define them.

"""
Initialization file for the `chap_core.database` package.

This package is responsible for all database interactions within the CHAP-core
application. It includes modules for:
- Defining database table schemas using SQLModel (e.g., `base_tables.py`,
  `dataset_tables.py`, `model_spec_tables.py`, `tables.py`).
- Managing database sessions and engine creation (e.g., `database.py`).
- Caching mechanisms related to database operations (e.g., `local_db_cache.py`).
- Utility functions for database operations, potentially including debugging
  and session handling for specific formats like JSON (`debug.py`, `json_session.py`).

By organizing database-related code here, CHAP-core centralizes its data persistence logic.
"""

# Example of how commonly used components could be re-exported for convenience:
# from .database import SessionWrapper, get_engine, get_session
# from .tables import SomeCoreTableModel
#
# __all__ = [
#     'SessionWrapper',
#     'get_engine',
#     'get_session',
#     'SomeCoreTableModel',
# ]

# By default, an empty __init__.py simply makes the directory a package.
