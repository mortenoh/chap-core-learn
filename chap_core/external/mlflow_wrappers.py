# OBSOLETE FILE - TO BE DELETED
#
# Improvement Suggestions:
# 1. Mark as Obsolete: Add a prominent block comment and module docstring stating this file is obsolete and to be deleted. (Done)
# 2. Verify Code Migration: Before deletion, ensure all functionalities previously in this file have been successfully migrated.
# 3. Check for Imports: Search the codebase for any remaining imports from this file and update/remove them.
# 4. Update Documentation: If this file was mentioned in project documentation, update the documentation.
# 5. Confirm Deletion: After verification and updates, confirm with the team and delete this file from the repository.

"""
**OBSOLETE MODULE: TO BE DELETED**

This module, `chap_core.external.mlflow_wrappers`, previously contained
wrappers and utility functions for interacting with MLflow projects.

The code and functionalities formerly in this file have been moved to other
locations within the `chap_core` project or have been superseded by other
mechanisms.

This file is retained temporarily to note its obsolescence and should be
removed in a future cleanup. Please ensure no active parts of the system
are still importing from this module.
"""

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "Module `chap_core.external.mlflow_wrappers` is obsolete and scheduled for removal. "
    "Its contents have been moved or are no longer used."
)

# Original comment indicating obsolescence:
# this file can be removed in the future, code has moved
