from typing import Optional, Tuple, Union

from text_selection_core.validation import ValidationErrBase

Success = bool
ChangedAnything = bool

ExecutionResult = Tuple[Success, Optional[ChangedAnything]]
ExecutionResult2 = Union[ValidationErrBase, Optional[ChangedAnything]]
