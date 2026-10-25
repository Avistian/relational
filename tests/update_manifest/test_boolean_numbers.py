"""REL06-V1: JSON booleans must not compare equal to JSON numbers."""
import copy
import unittest

from tests.update_manifest import test_update_manifest as frozen


class BooleanNumbers(unittest.TestCase):
    def test_check_rejects_boolean_version_and_year_without_writes(self):
        fixture = frozen.UpdateManifest()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()

        # Both spellings are JSON numbers; object formatting is irrelevant.
        numeric = copy.deepcopy(fixture.expected)
        numeric['version'] = 1.0
        numeric['lessons'][0]['year'] = 1.0
        fixture.write(numeric)
        fixture.checked(True)

        for field in ('version', 'year'):
            with self.subTest(field=field):
                changed = copy.deepcopy(fixture.expected)
                target = changed if field == 'version' else changed['lessons'][0]
                target[field] = True
                fixture.write(changed)
                fixture.checked(False)


if __name__ == '__main__':
    unittest.main()
