"""Standard-library tests can run without importing another PyTorch process."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.deduplicate_prepared_images import deduplicate


class PreparedStorageTests(unittest.TestCase):
    def test_only_identical_inactive_generated_images_share_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name / 'dataset/images/train/a.png' for name in ('old-a', 'old-b', 'different', 'yolo-g10-active')]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True)
                path.write_bytes(b'original' if index != 2 else b'different')
            original_inodes = [p.stat().st_ino for p in paths]
            dry = deduplicate(root)
            assert dry['coalesced_paths'] == 1
            assert [p.stat().st_ino for p in paths] == original_inodes
            result = deduplicate(root, apply=True)
            assert result['coalesced_paths'] == 1
            assert paths[0].stat().st_ino == paths[1].stat().st_ino
            assert paths[3].stat().st_ino == original_inodes[3]
            assert [p.read_bytes() for p in paths] == [b'original', b'original', b'different', b'original']
            assert deduplicate(root, apply=True)['coalesced_paths'] == 0

    def test_failed_atomic_replacement_keeps_original_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a', 'b'):
                path = root / name / 'dataset/images/train/a.png'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'original')
            with patch('scripts.deduplicate_prepared_images.os.replace', side_effect=OSError('fixture')):
                with self.assertRaises(OSError):
                    deduplicate(root, apply=True)
            assert len(list(root.rglob('a.png'))) == 2
            assert all(p.read_bytes() == b'original' for p in root.rglob('a.png'))
            assert not list(root.rglob('.prepared-link-*'))


if __name__ == '__main__':
    unittest.main()
