"""Batch-export scene overlaps, then create fixed TIFF pairs locally."""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .gee_pairs import (CLASSES, IndiaPairConfig, IndiaPairDownloader, aligned_grids,
                        identity, make_selection, save_json, write_pair_tiff)


def plan_tiles(downloader, max_tiles=30, candidate_limit=150):
    """Persist a bounded scene plan. Existing plans are never silently replaced."""
    path = downloader.root / 'tile_plan.json'
    if path.exists():
        return json.loads(path.read_text())
    ee, cfg = downloader.ee, downloader.config
    plan, seen = [], set()
    # Interleave classes rather than completing a single category first.
    candidates = downloader.candidates()
    pools = [[s for s in candidates if s['code'] == code] for code in CLASSES]
    sites = [pool[i] for i in range(max(map(len, pools))) for pool in pools if i < len(pool)]
    for site in sites[:candidate_limit]:
        for pair in downloader.scene_pairs(site):
            tile_id = identity([pair['landsat'], pair['sentinel']])[:24]
            if tile_id in seen:
                continue
            crs, tr = downloader.pair_projection(pair)
            a, b, c, d, e, f = tr
            if (a, b, d, e) != (30, 0, 0, -30):
                raise ValueError('Expected native Landsat north-up 30 m grid')
            overlap = ee.Image(pair['landsat']).geometry().intersection(
                ee.Image(pair['sentinel']).geometry(), 30).intersection(downloader.india, 30)
            if downloader.retry(lambda: overlap.area(30).getInfo()) < 25e6:
                continue
            coords = downloader.retry(lambda: overlap.bounds(30, ee.Projection(crs)).coordinates().getInfo())[0]
            xs, ys = zip(*coords)
            col0, col1 = math.floor((min(xs)-c)/30), math.ceil((max(xs)-c)/30)
            row0, row1 = math.floor((f-max(ys))/30), math.ceil((f-min(ys))/30)
            x0, y0 = c+30*col0, f-30*row0
            width, height = col1-col0, row1-row0
            if max(width, height) * 3 > 32768:
                continue
            plan.append(dict(**pair, tile_id=tile_id, crs=crs,
                lr_transform=[30, 0, x0, 0, -30, y0],
                hr_transform=[10, 0, x0, 0, -10, y0], width=width, height=height,
                bounds=[x0, y0-height*30, x0+width*30, y0],
                discovery_class=site['scene_class']))
            seen.add(tile_id)
            print(f'Planned overlap {len(plan)}/{max_tiles}: {tile_id}', flush=True)
            break
        if len(plan) >= max_tiles:
            break
    if not plan:
        raise ValueError('No overlapping scenes found; inspect candidate dates/coverage')
    save_json(path, plan)
    return plan


def submit_exports(downloader, plan, drive_folder, max_active=2):
    """Submit only available queue slots. Rerun to progress; task IDs survive restarts."""
    if max_active < 1:
        raise ValueError('max_active must be positive')
    ee = downloader.ee
    directory = downloader.root / 'tile_tasks'
    directory.mkdir(exist_ok=True)
    all_tasks = ee.batch.Task.list()
    active = sum(t.status()['state'] in ('READY', 'RUNNING') for t in all_tasks)
    known = {t.id: t for t in all_tasks}
    reports = []
    for tile in plan:
        for kind in ('lr', 'hr', 'landcover'):
            prefix = f"{tile['tile_id']}_{kind}"
            receipt = directory / f'{prefix}.json'
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                status = ee.data.getTaskStatus(saved['task_id'])[0]
                reports.append(dict(prefix=prefix, **status))
                # Failed tasks are not automatically duplicated. Fix cause first.
                continue
            # Recover a task submitted before its local receipt was saved.
            recovered = [t for t in known.values() if t.config and t.config.get('description') == prefix]
            if recovered:
                task = recovered[-1]
                save_json(receipt, dict(task_id=task.id, prefix=prefix))
                reports.append(dict(prefix=prefix, **task.status()))
                continue
            if active >= max_active:
                continue
            region = ee.Geometry.Rectangle(tile['bounds'], proj=tile['crs'], geodesic=False)
            overlap = ee.Image(tile['landsat']).geometry().intersection(
                ee.Image(tile['sentinel']).geometry(), 30).intersection(downloader.india, 30)
            if kind == 'landcover':
                image = downloader.cover.clip(overlap).unmask(0, sameFootprint=False).toFloat()
            else:
                image = downloader.image(tile['landsat' if kind == 'lr' else 'sentinel'], kind == 'lr')
                image = image.clip(overlap).unmask(0, sameFootprint=False)
            task = ee.batch.Export.image.toDrive(image=image, description=prefix,
                folder=drive_folder, fileNamePrefix=prefix, region=region,
                crs=tile['crs'], crsTransform=tile['hr_transform' if kind == 'hr' else 'lr_transform'],
                fileFormat='GeoTIFF', fileDimensions=32768, maxPixels=2e9)
            task.start()
            # Recovery above uses the unique description if interrupted before this write.
            save_json(receipt, dict(task_id=task.id, prefix=prefix))
            active += 1
            reports.append(dict(prefix=prefix, **task.status()))
    return reports


def crop_tiles(root, tile_directory, config=IndiaPairConfig()):
    """Window-read aligned exported TIFFs; preserve deterministic spatial quotas."""
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    root, tile_directory = Path(root), Path(tile_directory)
    config.validate()
    plan = json.loads((root / 'tile_plan.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((root / 'accepted').glob('*.json'))]
    for r in rows:
        if any(not (root / r[k]).is_file() for k in ('lr_path', 'hr_path')):
            raise FileNotFoundError('An accepted TIFF is missing; restore it before resuming')
    seen = {r['pair_id'] for r in rows}
    counts = Counter((r['split'], r['scene_class']) for r in rows)
    locations = Counter(r['location_id'] for r in rows)
    quotas = dict(train=(max(config.sizes)-config.test_count-config.val_count)//5,
                  val=config.val_count//5, test=config.test_count//5)
    pending = []
    n = config.lr_size
    for tile in plan:
        paths = {k: list(tile_directory.glob(f"{tile['tile_id']}_{k}*.tif"))
                 for k in ('lr', 'hr', 'landcover')}
        if any(not ps for ps in paths.values()):
            pending.append(tile['tile_id'])
            # Processing only the ready prefix keeps selection independent of export speed.
            break
        if any(len(ps) != 1 for ps in paths.values()):
            raise ValueError('Multiple TIFF shards/duplicate exports detected; resolve them before cropping')
        with rasterio.open(paths['lr'][0]) as lr, rasterio.open(paths['hr'][0]) as hr, \
                rasterio.open(paths['landcover'][0]) as cover:
            if lr.count != 4 or hr.count != 4 or cover.count != 1:
                raise ValueError('Expected RGB+valid bands for sensors and one class band')
            if (hr.width, hr.height) != (lr.width*3, lr.height*3):
                raise ValueError('Export geometry is not exact 3x')
            if lr.crs != hr.crs or lr.crs != cover.crs or cover.transform != lr.transform:
                raise ValueError('Exported projections do not match')
            expected = lr.transform * rasterio.Affine.scale(1/3, 1/3)
            if not hr.transform.almost_equals(expected) or (cover.width, cover.height) != (lr.width, lr.height):
                raise ValueError('Export origins, sizes or transforms do not match')
            to_geo = Transformer.from_crs(lr.crs, 'EPSG:4326', always_xy=True)
            windows = [(r, c) for r in range(0, lr.height-n+1, n) for c in range(0, lr.width-n+1, n)]
            windows.sort(key=lambda rc: identity([config.seed, tile['tile_id'], *rc]))
            for r, c in windows:
                pid = identity([tile['tile_id'], r, c])[:24]
                if pid in seen:
                    continue
                x, y = lr.transform * (c+n/2, r+n/2)
                lon, lat = to_geo.transform(x, y)
                try:
                    # Move slightly inside the center pixel to avoid floating point floor errors.
                    lon_g, lat_g = to_geo.transform(*(lr.transform * (c+n//2+0.1, r+n//2+0.1)))
                    grid = aligned_grids(lon_g, lat_g, tile['crs'], list(lr.transform)[:6], config)
                except ValueError:
                    continue
                if grid['row'] != r or grid['col'] != c:
                    raise ValueError('Local window and guarded grid differ')
                location = identity([round(lon, 5), round(lat, 5)])[:20]
                if locations[location] >= config.max_pairs_per_location:
                    continue
                wc = cover.read(1, window=Window(c, r, n, n))
                fractions = {code: float((wc == code).mean()) for code in CLASSES}
                code = max(fractions, key=fractions.get)
                key = (grid['split'], CLASSES[code])
                if fractions[code] < config.minimum_class_fraction or counts[key] >= quotas[grid['split']]:
                    continue
                low = lr.read(window=Window(c, r, n, n))
                high = hr.read(window=Window(c*3, r*3, n*3, n*3))
                ml, mh = (low[3:4] >= .999), (high[3:4] >= .999)
                for image, mask in ((low, ml), (high, mh)):
                    mask &= (np.isfinite(image[:3]) & (image[:3]>=0) & (image[:3]<=1)).all(0, keepdims=True)
                mh &= ml.repeat(3, 1).repeat(3, 2)
                ml &= mh.reshape(1, n, 3, n, 3).all((2, 4))
                mh &= ml.repeat(3, 1).repeat(3, 2)
                if ml.mean() < config.minimum_valid:
                    continue
                record = dict(pair_id=pid, location_id=location, block=grid['block'], split=grid['split'],
                    scene_class=CLASSES[code], code=code, grid=grid, valid_fraction=float(mh.mean()),
                    class_fraction=fractions[code], lon=lon, lat=lat,
                    **{k: tile[k] for k in ('landsat', 'sentinel', 'landsat_date', 'sentinel_date', 'gap_seconds')})
                for sensor, image, mask in [('lr', low[:3], ml), ('hr', high[:3], mh)]:
                    path = root / 'master' / sensor.upper() / f'{pid}.tif'
                    write_pair_tiff(path, np.where(mask, image, 0), mask, record, sensor)
                    record[f'{sensor}_path'] = path.relative_to(root).as_posix()
                save_json(root / 'accepted' / f'{pid}.json', record)
                rows.append(record); seen.add(pid); counts[key] += 1; locations[location] += 1
            print('Processed tile:', tile['tile_id'], 'accepted:', len(rows), flush=True)
    print('Counts:', dict(counts), 'tiles awaiting files:', len(pending))
    # Never silently change a finished selection.
    selection = make_selection(rows, config)
    path = root / 'fixed_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Fixed selections differ; preserve the previous study')
    save_json(path, selection)
    return rows
