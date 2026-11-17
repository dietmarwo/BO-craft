"""Optional-backend Bayesian optimization without an execution service."""

__version__ = "0.1.0"

from .api import capabilities, propose, recommend
from .specs import Error, validate_request

__all__ = ["Error", "capabilities", "propose", "recommend", "validate_request"]
