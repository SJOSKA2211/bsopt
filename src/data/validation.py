import pandas as pd
from typing import NamedTuple

class ValidationReport(NamedTuple):
    passed: bool
    errors: list[str] = []

class OptionsDataValidator:
    def __init__(self, min_samples: int):
        self.min_samples = min_samples

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        # For now, a very basic validation to pass the test
        if df.empty:
            return ValidationReport(passed=False, errors=["DataFrame is empty"])
        if len(df) < self.min_samples:
            return ValidationReport(passed=False, errors=[f"Less than {self.min_samples} samples"])
        
        # Check for negative strike as seen in test_options_data_validation
        if 'strike' in df.columns and (df['strike'] < 0).any():
            return ValidationReport(passed=False, errors=["Negative strike found"])

        return ValidationReport(passed=True)