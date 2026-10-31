"""Guard the all-state manuscript input against partial or malformed replacements."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import REACT_ESR_reproducible as react


class StorageInputTest(unittest.TestCase):
    def test_all_states_required_and_values_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "storage.csv"
            frame = pd.DataFrame({"state": react.core.STATE_ORDER,
                                  react.STORAGE_COLUMN: np.arange(16, dtype=float)})
            frame.to_csv(path, index=False)
            self.assertEqual(len(react.load_storage_file(path)), 16)
            for invalid in (frame.iloc[:-1], pd.concat([frame, frame.iloc[[0]]]),
                            frame.assign(**{react.STORAGE_COLUMN: -1}),
                            frame.assign(**{react.STORAGE_COLUMN: np.inf})):
                invalid.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    react.load_storage_file(path)

    def test_default_panel_uses_archived_registry_for_every_state(self):
        values = react.load_storage_file(react.DEFAULT_STORAGE_FILE)
        panel = react._configured_static()
        np.testing.assert_array_equal(panel.loc[react.core.STATE_ORDER, "Storage_MWh"],
                                      values.reindex(react.core.STATE_ORDER))
        self.assertTrue((values > 0).all())


if __name__ == "__main__":
    unittest.main()
