"""Offline tests; these do not authenticate to or query Earth Engine."""
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
import zipfile

import numpy as np

from geodiff_gan.data.gee_pairs import (
    CLASSES, IndiaPairConfig, aligned_grids, block_assignment,
    export_datasets, make_selection, save_json,
)
from geodiff_gan.experiments.trust_moe import prepare_manifest


class IndiaPairTests(unittest.TestCase):
    def setUp(self):
        self.config = IndiaPairConfig(sizes=(20, 40, 60), test_count=10, val_count=5, lr_size=8)
        self.rows = []
        for split, count in [('train', 9), ('val', 1), ('test', 2)]:
            for label in CLASSES.values():
                for i in range(count):
                    pid = f'{split}_{label}_{i}'
                    self.rows.append(dict(pair_id=pid, location_id=pid, block=f'block_{split}',
                        split=split, scene_class=label, npz=f'master/{pid}.npz',
                        grid=dict(row=0, col=0, crs='EPSG:32644', hr_transform=[10, 0, 0, 0, -10, 0]),
                        valid_fraction=1., landsat='LS/' + pid, sentinel='S2/' + pid,
                        landsat_date='2024-01-01', sentinel_date='2024-01-02', gap_seconds=86400))

    def test_nested_and_fixed(self):
        selection = make_selection(self.rows, self.config)
        shuffled = list(self.rows)
        random.Random(1).shuffle(shuffled)
        self.assertEqual(selection, make_selection(shuffled, self.config))
        last_train = set()
        for n in (20, 40, 60):
            ids = selection[str(n)]
            self.assertEqual(len(ids), n)
            train = {p for p in ids if p.startswith('train_')}
            self.assertTrue(last_train <= train)
            self.assertEqual(len(train), n - 15)
            last_train = train
            for split in ('val_', 'test_'):
                self.assertEqual({p for p in ids if p.startswith(split)},
                                 {p for p in selection['20'] if p.startswith(split)})

    def test_shortfall_and_leakage(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            make_selection(self.rows[:-1], self.config)
        bad = [dict(r) for r in self.rows]
        bad[-1]['block'] = bad[0]['block']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            make_selection(bad, self.config)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            make_selection(self.rows + self.rows[:1], self.config)

    def test_configuration(self):
        IndiaPairConfig().validate()
        with self.assertRaises(ValueError):
            replace(self.config, test_count=5).validate()

    def test_native_grid(self):
        try:
            from pyproj import Transformer
        except ImportError:
            self.skipTest('pyproj is needed for coordinate tests')
        to_geo = Transformer.from_crs('EPSG:6933', 'EPSG:4326', always_xy=True)
        lon, lat = to_geo.transform(195 * 40000 + 20000, 70 * 40000 + 20000)
        native = [30, 0, 300000, 0, -30, 4000000]
        grid = aligned_grids(lon, lat, 'EPSG:32644', native, self.config)
        self.assertEqual(grid['lr_transform'][2::3], grid['hr_transform'][2::3])
        self.assertAlmostEqual((grid['lr_transform'][2] - native[2]) % 30, 0)
        self.assertEqual(grid['bounds'][2] - grid['bounds'][0], 30 * self.config.lr_size)
        lon, lat = to_geo.transform(195 * 40000 + 10, 70 * 40000 + 20000)
        with self.assertRaisesRegex(ValueError, 'guard'):
            aligned_grids(lon, lat, 'EPSG:32644', native, self.config)

    def test_block_is_location_only(self):
        self.assertEqual(block_assignment(100001, 200001, self.config),
                         block_assignment(100002, 200002, self.config))

    def test_export_and_training_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'master').mkdir()
            for i, row in enumerate(self.rows):
                np.savez_compressed(root / row['npz'],
                    lr=np.full((3, 8, 8), (i + 1) / 100, dtype=np.float32),
                    hr=np.full((3, 24, 24), (i + 1) / 100, dtype=np.float32),
                    valid_mask_lr=np.ones((1, 8, 8), dtype=np.float32),
                    valid_mask_hr=np.ones((1, 24, 24), dtype=np.float32))
                save_json(root / 'accepted' / f"{row['pair_id']}.json", row)
            save_json(root / 'fixed_selection.json', make_selection(self.rows, self.config))
            archives = export_datasets(root, self.config)
            self.assertEqual(len(archives), 3)
            self.assertEqual(archives, export_datasets(root, self.config))
            with zipfile.ZipFile(archives[0]) as handle:
                self.assertIsNone(handle.testzip())
                self.assertEqual(sum(n.endswith('.npz') for n in handle.namelist()), 20)
                handle.extractall(root / 'portable')
            report = prepare_manifest(root / 'portable/manifest.jsonl', root / 'audit/manifest.jsonl',
                                      spatial_audit=False, disjoint_tiles=True)
            self.assertTrue(report['disjoint_blocks'])
            self.assertEqual(report['counts'], {'train': 5, 'val': 5, 'test': 10})
            lock = json.loads((root / 'fixed_selection.json').read_text())
            lock['20'].reverse()
            save_json(root / 'fixed_selection.json', lock)
            with self.assertRaisesRegex(ValueError, 'fixed_selection'):
                export_datasets(root, self.config)


if __name__ == '__main__':
    unittest.main()
