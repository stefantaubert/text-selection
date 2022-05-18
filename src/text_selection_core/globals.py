from typing import Optional, Tuple

from text_selection_core.validation import ValidationError

TQDM_LINE_UNIT = " line(s)"

ChangedAnything = bool
ExecutionResult = Tuple[Optional[ValidationError], ChangedAnything]
