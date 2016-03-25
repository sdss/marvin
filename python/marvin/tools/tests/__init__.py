from unittest import TestCase
import warnings
from marvin.tools.core.exceptions import MarvinSkippedTestWargning
from functools import wraps


# Decorator to skip a test if the session is None (i.e., if there is no DB)
def skipIfNoDB(test):
    @wraps(test)
    def wrapper(self, *args, **kwargs):
        if not self.session:
            return self.skipTest(test)
        else:
            return test(self, *args, **kwargs)
    return wrapper


class MarvinTest(TestCase):
    """Custom class for Marvin-tools tests."""

    def skipTest(self, test):
        """Issues a warning when we skip a test."""
        warnings.warn('Skipped test {0} because there is no DB connection.'
                      .format(test.__name__), MarvinSkippedTestWargning)
