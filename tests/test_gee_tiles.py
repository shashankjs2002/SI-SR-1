"""Offline tests for local tile cropping; no Earth Engine authentication."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
from rasterio.transform import Affine

from geodiff_gan.data.gee_pairs import IndiaPairConfig, save_json
from geodiff_gan.data.gee_tiles import crop_tiles, submit_exports


class TileTests(unittest.TestCase):
    def test_crop_tiffs_resume_and_shortfall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile = dict(tile_id='scene', landsat='LS', sentinel='S2', landsat_date='2024-01-01',
                        sentinel_date='2024-01-02', gap_seconds=86400, crs='EPSG:32644')
            save_json(root / 'tile_plan.json', [tile])
            tr = Affine(30, 0, 300000, 0, -30, 3000000)
            for kind, size, count in [('lr', 8, 4), ('hr', 24, 4), ('landcover', 8, 1)]:
                array = np.full((count, size, size), 10 if kind == 'landcover' else .1, dtype=np.float32)
                if count == 4:
                    array[3] = 1
                with rasterio.open(root / f'scene_{kind}.tif', 'w', driver='GTiff', width=size,
                        height=size, count=count, dtype='float32', crs=tile['crs'],
                        transform=tr * Affine.scale(1/3, 1/3) if kind == 'hr' else tr) as dst:
                    dst.write(array)
            cfg = IndiaPairConfig(lr_size=8, sizes=(20, 40, 60), val_count=5, test_count=10)
            grid = dict(row=0, col=0, crs=tile['crs'], block='train_block', split='train',
                        lr_transform=list(tr)[:6], hr_transform=list(tr * Affine.scale(1/3, 1/3))[:6])
            transformer = MagicMock()
            transformer.Transformer.from_crs.return_value.transform.side_effect = lambda x, y: (x, y)
            with patch.dict('sys.modules', pyproj=transformer), \
                    patch('geodiff_gan.data.gee_tiles.aligned_grids', return_value=grid):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, 'Insufficient'):
                        crop_tiles(root, root, cfg)
            receipts = list((root / 'accepted').glob('*.json'))
            self.assertEqual(len(receipts), 1)
            row = json.loads(receipts[0].read_text())
            with rasterio.open(root / row['hr_path']) as src:
                self.assertEqual(src.shape, (24, 24))
                np.testing.assert_allclose(src.read(), .1)
            self.assertFalse(list(root.rglob('*.npz')))

    def test_task_receipts_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dl = MagicMock()
            dl.root = Path(tmp)
            dl.ee.batch.Task.list.return_value = []
            dl.ee.data.getTaskStatus.return_value = [dict(state='COMPLETED')]
            for kind in ('lr', 'hr', 'landcover'):
                save_json(dl.root / 'tile_tasks' / f'scene_{kind}.json', dict(task_id=kind))
            reports = submit_exports(dl, [dict(tile_id='scene')], 'folder')
            self.assertEqual(len(reports), 3)
            dl.ee.batch.Export.image.toDrive.assert_not_called()


if __name__ == '__main__':
    unittest.main()
