from typing import Optional, Tuple

from text_selection_core.validation import ValidationError

ChangedAnything = bool
ExecutionResult = Tuple[Optional[ValidationError], ChangedAnything]
