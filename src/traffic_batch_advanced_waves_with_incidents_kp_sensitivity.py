import os, re, glob, argparse, json, logging, itertools, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLS = [
    'row_id','vehicle_id','hour','minute','sec','lat1','lon1','x1','y1',
    'fixedlat','fixedlon','z','fixedKP','kp_dist','lane_no','speed',
    'heading','acc','f1','f2','fixedlane','misc','abs_flag'
]

DEFAULT_CONFIG = {
    "file_pattern": "*.csv",
    "chunksize": 400000,
    "speed_threshold": 5.0,
    "dwell_threshold": 120.0,
    "low_speed_threshold": 20.0,
    "cell_m": 100,
    "time_bin": "30s",
    "draw_overlay_sample_n": 120,
    "min_consecutive_bins": 3,
    "incident_harsh_brake_threshold": 0.10,
    "incident_stop_rate_threshold": 0.10,
    "incident_shockwave_threshold": -10.0,
    "incident_speed_anomaly_threshold": -15.0,
    "bottleneck_occurrence_threshold": 0.30,
    "bottleneck_speed_threshold": 40.0,
    "incident_time_buffer_min": 15,
    "incident_kp_tolerance_km": 0.5,
    "incident_target_names": ["事故", "故障車", "落下物", "車線規制", "工事", "渋滞"],
    "incident_route_filter": "高速３号大高線",
    "kp_bounds_by_direction": {
        "up":   {"kp_min": 1.9, "kp_max": 3.7, "anchor_kp": 3.7},
        "down": {"kp_min": 2.3, "kp_max": 3.8, "anchor_kp": 2.3}
    }
}

def load_config(config_path=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    return cfg

def load_sensitivity_grid(grid_path=None):
    if grid_path and os.path.exists(grid_path):
        with open(grid_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "dwell_threshold": [120.0, 150.0, 180.0],
        "low_speed_threshold": [15.0, 20.0, 25.0],
        "incident_shockwave_threshold": [-8.0, -10.0, -12.0],
        "incident_speed_anomaly_threshold": [-10.0, -15.0, -20.0],
        "incident_time_buffer_min": [10, 15, 20],
        "incident_kp_tolerance_km": [0.3, 0.5, 0.7]
    }

def generate_config_variants(base_cfg, grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    variants = []
    for i, combo in enumerate(itertools.product(*values), start=1):
        cfg = copy.deepcopy(base_cfg)
        label_parts = []
        for k, v in zip(keys, combo):
            cfg[k] = v
            label_parts.append(f"{k}={v}")
        cfg['_scenario_id'] = f'scenario_{i:03d}'
        cfg['_scenario_label'] = ';'.join(label_parts)
        variants.append(cfg)
    return variants

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'traffic_analysis.log')
    logger = logging.getLogger('traffic_batch_sensitivity_incidents_route_filtered_nohardcodeddate')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_path

def infer_direction_from_filename(path):
    name = os.path.basename(path)
    if any(k in name for k in ['上り', '上り方向', 'nobori', 'up']):
        return 'up'
    if any(k in name for k in ['下り', '下り方向', 'kudari', 'down']):
        return 'down'
    return None

def infer_date_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r'(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)', name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    raise ValueError(f"date not found in filename: {name}")

def infer_route_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name

def build_timestamp(df, base_date):
    if base_date is None or pd.isna(base_date):
        raise ValueError("base_date is required to build timestamps")
    return (
        pd.Timestamp(base_date)
        + pd.to_timedelta(df['hour'], unit='h')
        + pd.to_timedelta(df['minute'], unit='m')
        + pd.to_timedelta(df['sec'], unit='s')
    )

def normalize_direction_text(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if '上' in s or s == 'up' or 'nobori' in s:
        return 'up'
    if '下' in s or s == 'down' or 'kudari' in s:
        return 'down'
    return None

def safe_to_datetime(s):
    return pd.to_datetime(s, errors='coerce')

def safe_to_numeric(s):
    return pd.to_numeric(s, errors='coerce')

def get_kp_bounds(cfg, direction):
    d = cfg.get('kp_bounds_by_direction', {}).get(direction, {})
    kp_min = d.get('kp_min', None)
    kp_max = d.get('kp_max', None)
    anchor_kp = d.get('anchor_kp', None)
    return kp_min, kp_max, anchor_kp

def normalize_incident_one_file(csv_path, cfg, logger):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(csv_path, encoding='cp932')

    rename_map = {}
    route_count = 0
    dir_count = 0
    kp_count = 0

    for c in df.columns:
        cs = str(c).strip()
        if cs == '名称':
            rename_map[c] = 'incident_name'
        elif cs == '番号':
            rename_map[c] = 'incident_id'
        elif cs == '状態':
            rename_map[c] = 'incident_status'
        elif cs == '新規時刻':
            rename_map[c] = 'start_time'
        elif cs == '更新・削除時刻':
            rename_map[c] = 'end_time'
        elif cs == '機関区分':
            rename_map[c] = 'agency'
        elif cs == '道路区分':
            rename_map[c] = 'road_type'
        elif cs == '路線':
            route_count += 1
            rename_map[c] = 'route_primary' if route_count == 1 else 'route_secondary'
        elif cs == '方向':
            dir_count += 1
            rename_map[c] = 'direction_primary' if dir_count == 1 else 'direction_secondary'
        elif cs == 'キロポスト':
            kp_count += 1
            rename_map[c] = 'kp_primary' if kp_count == 1 else 'kp_secondary'

    df = df.rename(columns=rename_map).copy()

    if 'start_time' in df.columns:
        df['start_time'] = safe_to_datetime(df['start_time'])
    else:
        df['start_time'] = pd.NaT

    if 'end_time' in df.columns:
        df['end_time'] = safe_to_datetime(df['end_time'])
    else:
        df['end_time'] = pd.NaT

    for c in ['kp_primary', 'kp_secondary']:
        if c in df.columns:
            df[c] = safe_to_numeric(df[c])

    df['direction_norm'] = df['direction_primary'].apply(normalize_direction_text) if 'direction_primary' in df.columns else None
    df['route_norm'] = df['route_primary'].astype(str).str.strip() if 'route_primary' in df.columns else None
    df['kp_norm'] = np.where(
        df['kp_primary'].notna() if 'kp_primary' in df.columns else False,
        df['kp_primary'] if 'kp_primary' in df.columns else np.nan,
        df['kp_secondary'] if 'kp_secondary' in df.columns else np.nan
    )

    targets = set(cfg.get('incident_target_names', []))
    if 'incident_name' in df.columns and len(targets):
        df = df[df['incident_name'].astype(str).isin(targets)].copy()

    route_filter = cfg.get('incident_route_filter')
    if route_filter and 'route_norm' in df.columns:
        df = df[df['route_norm'].astype(str).str.strip() == route_filter].copy()

    df['incident_date'] = df['start_time'].dt.strftime('%Y-%m-%d')
    df['duration_min'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60.0
    df['source_file'] = os.path.basename(csv_path)

    keep_cols = [
        'incident_name', 'incident_id', 'incident_status',
        'start_time', 'end_time', 'incident_date', 'duration_min',
        'agency', 'road_type',
        'route_primary', 'direction_primary', 'kp_primary',
        'route_secondary', 'direction_secondary', 'kp_secondary',
        'route_norm', 'direction_norm', 'kp_norm',
        'source_file'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    logger.info(f'incident file loaded={os.path.basename(csv_path)} rows_after_filter={len(out)}')
    return out

def load_incident_dir(incident_dir, cfg, logger):
    if incident_dir is None or not os.path.isdir(incident_dir):
        logger.info('incident dir not provided or not found')
        return pd.DataFrame()

    files = sorted(glob.glob(os.path.join(incident_dir, '*.csv')))
    if len(files) == 0:
        logger.info('incident dir has no csv files')
        return pd.DataFrame()

    parts = []
    for f in files:
        try:
            one = normalize_incident_one_file(f, cfg, logger)
            if len(one):
                parts.append(one)
        except Exception:
            logger.exception(f'incident read error: {os.path.basename(f)}')

    if not parts:
        return pd.DataFrame()

    inc = pd.concat(parts, ignore_index=True)
    inc = inc.sort_values(['incident_date', 'start_time', 'incident_id'], na_position='last').reset_index(drop=True)
    logger.info(f'incident master rows loaded={len(inc)} route_filter={cfg.get("incident_route_filter")}')
    return inc

def extract_wavefronts(agg, cfg):
    if len(agg) == 0:
        return pd.DataFrame()
    a = agg.copy().sort_values(['time_bin', 'cell100'])
    a['low_speed_flag'] = a['mean_speed'] < cfg['low_speed_threshold']
    low = a[a['low_speed_flag']].copy()
    if len(low) == 0:
        return pd.DataFrame()

    front = low.groupby('time_bin').agg(
        up_cell=('cell100', 'min'),
        down_cell=('cell100', 'max'),
        mean_speed=('mean_speed', 'mean'),
        max_stop_rate=('stop_rate', 'max'),
        max_harsh_brake=('harsh_brake_rate', 'max'),
        mean_shockwave=('shockwave_kmh', 'mean'),
        n_low_cells=('cell100', 'nunique')
    ).reset_index().sort_values('time_bin')

    front['dt_s'] = pd.to_datetime(front['time_bin']).diff().dt.total_seconds().fillna(0)
    front['dup'] = front['up_cell'].diff().abs().fillna(0)
    front['episode_break'] = (front['dt_s'] > 120) | (front['dup'] > 3)
    front['episode_id'] = front['episode_break'].cumsum() + 1

    episodes = front.groupby('episode_id').agg(
        start_time=('time_bin', 'min'),
        end_time=('time_bin', 'max'),
        n_bins=('time_bin', 'count'),
        start_up_cell=('up_cell', 'first'),
        end_up_cell=('up_cell', 'last'),
        start_down_cell=('down_cell', 'first'),
        end_down_cell=('down_cell', 'last'),
        mean_low_speed=('mean_speed', 'mean'),
        max_stop_rate=('max_stop_rate', 'max'),
        max_harsh_brake=('max_harsh_brake', 'max'),
        mean_shockwave=('mean_shockwave', 'mean'),
        mean_n_low_cells=('n_low_cells', 'mean')
    ).reset_index()

    episodes = episodes[episodes['n_bins'] >= cfg['min_consecutive_bins']].copy()
    if len(episodes) == 0:
        return episodes

    episodes['duration_s'] = (
        pd.to_datetime(episodes['end_time']) - pd.to_datetime(episodes['start_time'])
    ).dt.total_seconds().clip(lower=1)

    episodes['upstream_front_speed_kmh'] = (
        (episodes['end_up_cell'] - episodes['start_up_cell']) * cfg['cell_m'] / episodes['duration_s']
    ) * 3.6

    episodes['downstream_front_speed_kmh'] = (
        (episodes['end_down_cell'] - episodes['start_down_cell']) * cfg['cell_m'] / episodes['duration_s']
    ) * 3.6

    episodes['spatial_span_m'] = (
        episodes[['start_down_cell', 'end_down_cell']].max(axis=1)
        - episodes[['start_up_cell', 'end_up_cell']].min(axis=1) + 1
    ) * cfg['cell_m']

    episodes['front_fixedness_cells'] = (episodes['end_up_cell'] - episodes['start_up_cell']).abs()
    return episodes

def classify_waves(episodes, cfg):
    if len(episodes) == 0:
        return episodes

    e = episodes.copy()
    e['wave_class'] = 'natural_congestion'

    incident_cond = (
        (e['front_fixedness_cells'] <= 2) &
        (
            (e['max_harsh_brake'] >= cfg['incident_harsh_brake_threshold']) |
            (e['max_stop_rate'] >= cfg['incident_stop_rate_threshold']) |
            (e['mean_shockwave'] < cfg['incident_shockwave_threshold'])
        ) &
        (e['duration_s'] <= 3600)
    )
    e.loc[incident_cond, 'wave_class'] = 'incident_like'

    weak_cond = (e['n_bins'] < cfg['min_consecutive_bins'] + 1) | (e['spatial_span_m'] < 200)
    e.loc[weak_cond, 'wave_class'] = 'weak_or_noise'
    return e

def process_one_file(input_csv, output_dir, cfg, logger, base_date=None, direction=None, produce_figures=True):
    os.makedirs(output_dir, exist_ok=True)
    direction = direction or infer_direction_from_filename(input_csv) or 'up'
    if base_date is None:
        raise ValueError(f"base_date is required for file: {os.path.basename(input_csv)}")
    route_name = infer_route_from_filename(input_csv)

    logger.info(f"[{cfg.get('_scenario_id','base')}] file={os.path.basename(input_csv)} date={base_date} direction={direction}")

    stop_bins = {}
    kp_min_cfg, kp_max_cfg, anchor_kp_cfg = get_kp_bounds(cfg, direction)

    if kp_min_cfg is not None and kp_max_cfg is not None:
        kp_lo = min(kp_min_cfg, kp_max_cfg)
        kp_hi = max(kp_min_cfg, kp_max_cfg)
        logger.info(
            f"[{cfg.get('_scenario_id','base')}] fixed KP range applied: "
            f"direction={direction} kp_min={kp_lo} kp_max={kp_hi} anchor_kp={anchor_kp_cfg}"
        )
    else:
        kp_global_max = None
        kp_global_min = None
        for chunk in pd.read_csv(input_csv, header=None, names=COLS, chunksize=cfg['chunksize'], low_memory=False):
            for c in ['hour', 'minute', 'sec', 'fixedKP', 'speed', 'acc', 'abs_flag']:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
            chunk = chunk.dropna(subset=['hour', 'minute', 'sec', 'fixedKP', 'speed'])
            if len(chunk) == 0:
                continue
            kp_global_max = chunk['fixedKP'].max() if kp_global_max is None else max(kp_global_max, chunk['fixedKP'].max())
            kp_global_min = chunk['fixedKP'].min() if kp_global_min is None else min(kp_global_min, chunk['fixedKP'].min())

        if kp_global_max is None or kp_global_min is None:
            raise ValueError(f"no valid fixedKP data found: {os.path.basename(input_csv)}")

        kp_lo = kp_global_min
        kp_hi = kp_global_max
        if anchor_kp_cfg is None:
            anchor_kp_cfg = kp_global_max if direction == 'up' else kp_global_min
        logger.info(
            f"[{cfg.get('_scenario_id','base')}] inferred KP range: "
            f"direction={direction} kp_min={kp_lo} kp_max={kp_hi} anchor_kp={anchor_kp_cfg}"
        )

    for chunk in pd.read_csv(input_csv, header=None, names=COLS, chunksize=cfg['chunksize'], low_memory=False):
        for c in ['hour', 'minute', 'sec', 'fixedKP', 'speed', 'acc', 'abs_flag']:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk = chunk.dropna(subset=['hour', 'minute', 'sec', 'fixedKP', 'speed'])

        if kp_lo is not None and kp_hi is not None:
            chunk = chunk[(chunk['fixedKP'] >= kp_lo) & (chunk['fixedKP'] <= kp_hi)].copy()
        if len(chunk) == 0:
            continue

        speed_median = chunk['speed'].median()
        chunk['speed_kmh'] = chunk['speed'] * 3.6 if speed_median < 60 else chunk['speed']
        chunk['is_stop'] = chunk['speed_kmh'] < cfg['speed_threshold']
        chunk['x_m'] = chunk['fixedKP'] * 1000.0
        anchor_m = anchor_kp_cfg * 1000.0

        if direction == 'up':
            chunk['dist_tmp'] = anchor_m - chunk['x_m']
        else:
            chunk['dist_tmp'] = chunk['x_m'] - anchor_m

        st = chunk[chunk['is_stop']].copy()
        st['pos_bin30'] = (st['dist_tmp'] // 30).astype('Int64')
        g = st.groupby('pos_bin30')['vehicle_id'].nunique()
        for k, v in g.items():
            stop_bins[int(k)] = stop_bins.get(int(k), 0) + int(v)

    loc = pd.DataFrame({
        'pos_bin30': list(stop_bins.keys()),
        'unique_veh_proxy': list(stop_bins.values())
    }).sort_values('pos_bin30')

    thr = loc['unique_veh_proxy'].quantile(0.97) if len(loc) else np.inf
    signal_bins = set(loc.loc[loc['unique_veh_proxy'] >= thr, 'pos_bin30'].astype(int).tolist())

    states = {}
    remove_ids = set()
    rows_before = 0
    rows_after = 0
    veh_ids = set()
    kept_ids = set()
    sum_speed_before = 0.0
    sum_speed_after = 0.0
    n_speed_before = 0
    n_speed_after = 0
    cell_records = []

    for chunk in pd.read_csv(input_csv, header=None, names=COLS, chunksize=cfg['chunksize'], low_memory=False):
        for c in ['hour', 'minute', 'sec', 'fixedKP', 'speed', 'acc', 'abs_flag']:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk = chunk.dropna(subset=['hour', 'minute', 'sec', 'fixedKP', 'speed'])

        if kp_lo is not None and kp_hi is not None:
            chunk = chunk[(chunk['fixedKP'] >= kp_lo) & (chunk['fixedKP'] <= kp_hi)].copy()
        if len(chunk) == 0:
            continue

        chunk['timestamp'] = build_timestamp(chunk, base_date)
        chunk = chunk.sort_values(['vehicle_id', 'timestamp'])

        speed_median = chunk['speed'].median()
        chunk['speed_kmh'] = chunk['speed'] * 3.6 if speed_median < 60 else chunk['speed']
        chunk['x_m'] = chunk['fixedKP'] * 1000.0
        anchor = anchor_kp_cfg * 1000.0

        if direction == 'up':
            chunk['dist_downstream_m'] = anchor - chunk['x_m']
        else:
            chunk['dist_downstream_m'] = chunk['x_m'] - anchor

        delta = chunk.groupby('vehicle_id')['dist_downstream_m'].agg(lambda s: s.iloc[-1] - s.iloc[0])
        valid_ids = delta[delta > 0].index
        chunk = chunk[chunk['vehicle_id'].isin(valid_ids)].copy()
        if len(chunk) == 0:
            continue

        chunk['pos_bin30'] = (chunk['dist_downstream_m'] // 30).astype('Int64')
        rows_before += len(chunk)
        veh_ids.update(chunk['vehicle_id'].dropna().unique().tolist())
        sum_speed_before += chunk['speed_kmh'].sum()
        n_speed_before += chunk['speed_kmh'].notna().sum()

        for vid, g in chunk.groupby('vehicle_id'):
            g = g.sort_values('timestamp')
            st = states.get(vid, {'active': False, 'last_t': None, 'last_bin': None, 'dwell': 0.0})
            for row in g.itertuples(index=False):
                t = row.timestamp
                posb = int(row.pos_bin30) if pd.notna(row.pos_bin30) else None
                low = row.speed_kmh < cfg['speed_threshold']
                near = posb in signal_bins if posb is not None else False

                if near and low:
                    if st['active'] and st['last_t'] is not None and (t - st['last_t']).total_seconds() <= 1.5 and abs(posb - st['last_bin']) <= 1:
                        st['dwell'] += max(0.0, (t - st['last_t']).total_seconds())
                    else:
                        st['active'] = True
                        st['dwell'] = 0.0
                    st['last_t'] = t
                    st['last_bin'] = posb
                    if st['dwell'] >= cfg['dwell_threshold']:
                        remove_ids.add(vid)
                else:
                    st = {'active': False, 'last_t': t, 'last_bin': posb, 'dwell': 0.0}

                states[vid] = st

        keep = chunk[~chunk['vehicle_id'].isin(remove_ids)].copy()
        if len(keep) == 0:
            continue

        rows_after += len(keep)
        kept_ids.update(keep['vehicle_id'].dropna().unique().tolist())
        sum_speed_after += keep['speed_kmh'].sum()
        n_speed_after += keep['speed_kmh'].notna().sum()

        keep['time_bin'] = keep['timestamp'].dt.floor(cfg['time_bin'])
        keep['cell100'] = (keep['dist_downstream_m'] // cfg['cell_m']).astype('Int64')
        keep['fixedKP_mean'] = keep['fixedKP']

        cell = keep.groupby(['time_bin', 'cell100']).agg(
            mean_speed=('speed_kmh', 'mean'),
            p15_speed=('speed_kmh', lambda s: np.percentile(s, 15) if len(s) else np.nan),
            mean_acc=('acc', 'mean'),
            harsh_brake_rate=('acc', lambda s: np.mean(s <= -2.0) if len(s) else np.nan),
            stop_rate=('speed_kmh', lambda s: np.mean(s < cfg['speed_threshold']) if len(s) else np.nan),
            abs_rate=('abs_flag', 'mean'),
            probe_veh=('vehicle_id', 'nunique'),
            fixedKP_mean=('fixedKP_mean', 'mean')
        ).reset_index()

        cell_records.append(cell)

    agg = pd.concat(cell_records, ignore_index=True) if cell_records else pd.DataFrame()

    if len(agg):
        agg = agg.groupby(['time_bin', 'cell100']).agg(
            mean_speed=('mean_speed', 'mean'),
            p15_speed=('p15_speed', 'mean'),
            mean_acc=('mean_acc', 'mean'),
            harsh_brake_rate=('harsh_brake_rate', 'mean'),
            stop_rate=('stop_rate', 'mean'),
            abs_rate=('abs_rate', 'mean'),
            probe_veh=('probe_veh', 'sum'),
            fixedKP_mean=('fixedKP_mean', 'mean')
        ).reset_index()

        agg['flow_probe_per_h'] = agg['probe_veh'] * (3600 / pd.to_timedelta(cfg['time_bin']).total_seconds())
        agg['density_proxy'] = agg['flow_probe_per_h'] / agg['mean_speed'].clip(lower=1.0)
        agg = agg.sort_values(['cell100', 'time_bin'])
        agg['prev_flow'] = agg.groupby('cell100')['flow_probe_per_h'].shift(1)
        agg['prev_den'] = agg.groupby('cell100')['density_proxy'].shift(1)

        agg['shockwave_kmh'] = (
            (agg['flow_probe_per_h'] - agg['prev_flow']) /
            (agg['density_proxy'] - agg['prev_den']).replace(0, np.nan)
        ).clip(-80, 80)

        agg['file_name'] = os.path.basename(input_csv)
        agg['route_name'] = route_name
        agg['date'] = base_date
        agg['direction'] = direction
    else:
        agg = pd.DataFrame(columns=[
            'time_bin', 'cell100', 'mean_speed', 'p15_speed', 'mean_acc',
            'harsh_brake_rate', 'stop_rate', 'abs_rate', 'probe_veh',
            'fixedKP_mean', 'flow_probe_per_h', 'density_proxy', 'prev_flow',
            'prev_den', 'shockwave_kmh', 'file_name', 'route_name', 'date', 'direction'
        ])

    waves = extract_wavefronts(agg, cfg)
    waves = classify_waves(waves, cfg)

    if len(waves):
        waves['file_name'] = os.path.basename(input_csv)
        waves['route_name'] = route_name
        waves['date'] = base_date
        waves['direction'] = direction

        cell_kp = agg.groupby('cell100')['fixedKP_mean'].median().reset_index().rename(columns={'fixedKP_mean': 'cell_kp'})
        waves = waves.merge(cell_kp.rename(columns={'cell100': 'start_up_cell', 'cell_kp': 'start_up_kp'}), on='start_up_cell', how='left')
        waves = waves.merge(cell_kp.rename(columns={'cell100': 'end_up_cell', 'cell_kp': 'end_up_kp'}), on='end_up_cell', how='left')
        waves = waves.merge(cell_kp.rename(columns={'cell100': 'start_down_cell', 'cell_kp': 'start_down_kp'}), on='start_down_cell', how='left')
        waves = waves.merge(cell_kp.rename(columns={'cell100': 'end_down_cell', 'cell_kp': 'end_down_kp'}), on='end_down_cell', how='left')
        waves['event_kp_est'] = waves[['start_up_kp', 'end_up_kp', 'start_down_kp', 'end_down_kp']].mean(axis=1)

    if produce_figures:
        agg.to_csv(os.path.join(output_dir, 'aggregated.csv'), index=False)
        waves.to_csv(os.path.join(output_dir, 'wave_events.csv'), index=False)
        loc.to_csv(os.path.join(output_dir, 'signal_candidate_locations.csv'), index=False)

    summary = pd.DataFrame([{
        'scenario_id': cfg.get('_scenario_id', 'base'),
        'scenario_label': cfg.get('_scenario_label', 'base'),
        'file_name': os.path.basename(input_csv),
        'route_name': route_name,
        'date': base_date,
        'direction': direction,
        'kp_min_used': kp_lo,
        'kp_max_used': kp_hi,
        'anchor_kp_used': anchor_kp_cfg,
        'signal_candidate_bins': len(signal_bins),
        'vehicle_ids_seen': len(veh_ids),
        'vehicle_ids_removed': len(remove_ids),
        'vehicle_ids_remaining_proxy': len(kept_ids),
        'rows_before': rows_before,
        'rows_after': rows_after,
        'avg_speed_before_kmh': sum_speed_before / n_speed_before if n_speed_before else np.nan,
        'avg_speed_after_kmh': sum_speed_after / n_speed_after if n_speed_after else np.nan,
        'avg_speed_diff_kmh': (sum_speed_after / n_speed_after - sum_speed_before / n_speed_before) if n_speed_before and n_speed_after else np.nan,
        'shockwave_mean_kmh': agg['shockwave_kmh'].replace([np.inf, -np.inf], np.nan).mean() if len(agg) else np.nan,
        'wave_events': len(waves),
        'incident_like_waves': int((waves['wave_class'] == 'incident_like').sum()) if len(waves) else 0,
        'natural_waves': int((waves['wave_class'] == 'natural_congestion').sum()) if len(waves) else 0,
        'weak_waves': int((waves['wave_class'] == 'weak_or_noise').sum()) if len(waves) else 0
    }])

    if produce_figures and len(agg):
        piv = agg.pivot_table(index='cell100', columns='time_bin', values='mean_speed')
        plt.figure(figsize=(12, 6))
        plt.imshow(piv.sort_index(ascending=False), aspect='auto', cmap='RdYlBu', vmin=0, vmax=100)
        plt.colorbar(label='Mean speed (km/h)')
        plt.title(f'Time-Space Diagram: {os.path.basename(input_csv)}')
        plt.xlabel('Time bins')
        plt.ylabel('100m cell index')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'timespace.png'), dpi=160)
        plt.close()

    logger.info(f"[{cfg.get('_scenario_id', 'base')}] done file={os.path.basename(input_csv)} removed={len(remove_ids)} waves={len(waves)}")
    return summary, agg, waves

def build_baseline_and_classify(all_agg, all_waves, output_dir, cfg, logger, save_outputs=True):
    if len(all_agg) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_agg = all_agg.copy()
    all_agg['tod'] = pd.to_datetime(all_agg['time_bin']).dt.strftime('%H:%M:%S')

    baseline = all_agg.groupby(['direction', 'tod', 'cell100']).agg(
        baseline_speed_kmh=('mean_speed', 'median'),
        baseline_stop_rate=('stop_rate', 'median'),
        baseline_harsh_brake_rate=('harsh_brake_rate', 'median'),
        baseline_shockwave=('shockwave_kmh', 'median'),
        days=('date', 'nunique')
    ).reset_index()

    merged = all_agg.merge(baseline, on=['direction', 'tod', 'cell100'], how='left')
    merged['speed_anomaly_kmh'] = merged['mean_speed'] - merged['baseline_speed_kmh']

    merged['incident_candidate_flag'] = (
        (merged['mean_speed'] < cfg['low_speed_threshold']) &
        (merged['speed_anomaly_kmh'] <= cfg['incident_speed_anomaly_threshold']) &
        (
            (merged['harsh_brake_rate'] > merged['baseline_harsh_brake_rate'].fillna(0) + 0.05) |
            (merged['stop_rate'] > merged['baseline_stop_rate'].fillna(0) + 0.05) |
            (merged['shockwave_kmh'] < merged['baseline_shockwave'].fillna(0) - 5)
        )
    )

    bottleneck = merged.groupby(['direction', 'cell100']).agg(
        days=('date', 'nunique'),
        low_speed_occurrence=('mean_speed', lambda s: np.mean(s < cfg['low_speed_threshold'])),
        median_speed=('mean_speed', 'median'),
        baseline_speed=('baseline_speed_kmh', 'median')
    ).reset_index()

    bottleneck['bottleneck_flag'] = (
        (bottleneck['low_speed_occurrence'] >= cfg['bottleneck_occurrence_threshold']) &
        (bottleneck['baseline_speed'] < cfg['bottleneck_speed_threshold'])
    )

    incidents = merged[merged['incident_candidate_flag']].copy()
    incidents = incidents.groupby(['date', 'direction', 'cell100']).agg(
        start_time=('tod', 'min'),
        end_time=('tod', 'max'),
        min_speed=('mean_speed', 'min'),
        mean_speed_anomaly=('speed_anomaly_kmh', 'mean'),
        mean_shockwave=('shockwave_kmh', 'mean'),
        peak_stop_rate=('stop_rate', 'max'),
        peak_harsh_brake=('harsh_brake_rate', 'max'),
        duration_bins=('tod', 'count')
    ).reset_index()

    incidents['duration_min'] = incidents['duration_bins'] * pd.to_timedelta(cfg['time_bin']).total_seconds() / 60.0

    wave_refined = all_waves.copy() if len(all_waves) else pd.DataFrame()
    if len(wave_refined):
        wave_refined['refined_class'] = wave_refined['wave_class']
        wave_refined.loc[
            (wave_refined['wave_class'] == 'natural_congestion') &
            (wave_refined['max_harsh_brake'] >= cfg['incident_harsh_brake_threshold'] + 0.02),
            'refined_class'
        ] = 'incident_like'

    daydir = merged.groupby(['date', 'direction']).agg(
        avg_speed=('mean_speed', 'mean'),
        mean_shockwave=('shockwave_kmh', 'mean'),
        low_speed_cells=('mean_speed', lambda s: np.sum(s < cfg['low_speed_threshold'])),
        incident_candidate_cells=('incident_candidate_flag', 'sum')
    ).reset_index()

    if save_outputs:
        baseline.to_csv(os.path.join(output_dir, 'natural_congestion_baseline.csv'), index=False)
        merged.to_csv(os.path.join(output_dir, 'all_aggregated_with_baseline.csv'), index=False)
        bottleneck.to_csv(os.path.join(output_dir, 'recurrent_bottlenecks.csv'), index=False)
        incidents.to_csv(os.path.join(output_dir, 'incident_candidates.csv'), index=False)
        daydir.to_csv(os.path.join(output_dir, 'daily_direction_processing_stats.csv'), index=False)
        if len(wave_refined):
            wave_refined.to_csv(os.path.join(output_dir, 'wave_events_all.csv'), index=False)
        logger.info(
            f"[{cfg.get('_scenario_id', 'base')}] baseline_rows={len(baseline)} "
            f"bottlenecks={int(bottleneck['bottleneck_flag'].sum()) if len(bottleneck) else 0} "
            f"incidents={len(incidents)}"
        )

    return baseline, bottleneck, incidents, wave_refined, daydir

def evaluate_with_incidents(waves_all, incident_df, output_dir, cfg, logger, save_outputs=True):
    if len(waves_all) == 0 or len(incident_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    w = waves_all.copy()
    inc = incident_df.copy()

    w['start_time'] = pd.to_datetime(w['start_time'])
    w['end_time'] = pd.to_datetime(w['end_time'])
    inc['start_time'] = pd.to_datetime(inc['start_time'])
    inc['end_time'] = pd.to_datetime(inc['end_time'])

    matches = []
    time_buf = pd.Timedelta(minutes=cfg.get('incident_time_buffer_min', 15))
    kp_tol = cfg.get('incident_kp_tolerance_km', 0.5)
    route_filter = cfg.get('incident_route_filter')

    if route_filter and 'route_norm' in inc.columns:
        inc = inc[inc['route_norm'].astype(str).str.strip() == route_filter].copy()

    for wr in w.itertuples(index=False):
        ws = wr.start_time
        we = wr.end_time
        wdir = getattr(wr, 'direction', None)
        wdate = getattr(wr, 'date', None)
        wkp = getattr(wr, 'event_kp_est', np.nan)

        cand = inc.copy()

        if wdate is not None and 'incident_date' in cand.columns:
            cand = cand[cand['incident_date'] == wdate]

        if wdir is not None and 'direction_norm' in cand.columns:
            cand = cand[(cand['direction_norm'].isna()) | (cand['direction_norm'] == wdir)]

        cand = cand[
            (cand['start_time'] <= we + time_buf) &
            (cand['end_time'] >= ws - time_buf)
        ].copy()

        if pd.notna(wkp) and 'kp_norm' in cand.columns:
            cand['kp_diff_km'] = (cand['kp_norm'] - wkp).abs()
            cand = cand[(cand['kp_diff_km'].isna()) | (cand['kp_diff_km'] <= kp_tol)]
        else:
            cand['kp_diff_km'] = np.nan

        if len(cand) == 0:
            continue

        cand['time_gap_min'] = cand.apply(
            lambda r: min(
                abs((r['start_time'] - ws).total_seconds()) if pd.notna(r['start_time']) else np.inf,
                abs((r['end_time'] - we).total_seconds()) if pd.notna(r['end_time']) else np.inf
            ) / 60.0,
            axis=1
        )

        cand = cand.sort_values(['kp_diff_km', 'time_gap_min'], na_position='last')
        best = cand.iloc[0]

        matches.append({
            'date': wdate,
            'direction': wdir,
            'wave_start_time': ws,
            'wave_end_time': we,
            'wave_class': getattr(wr, 'wave_class', None),
            'refined_class': getattr(wr, 'refined_class', None),
            'event_kp_est': wkp,
            'incident_name': best.get('incident_name'),
            'incident_id': best.get('incident_id'),
            'incident_status': best.get('incident_status'),
            'incident_start_time': best.get('start_time'),
            'incident_end_time': best.get('end_time'),
            'incident_route': best.get('route_norm'),
            'incident_direction': best.get('direction_norm'),
            'incident_kp': best.get('kp_norm'),
            'kp_diff_km': best.get('kp_diff_km'),
            'time_gap_min': best.get('time_gap_min'),
            'source_file': best.get('source_file')
        })

    matched = pd.DataFrame(matches)

    if len(matched):
        target_wave = w[w['wave_class'] == 'incident_like'].copy()
        tp = len(matched[matched['wave_class'] == 'incident_like'])
        fp = len(target_wave) - tp

        matched_ids = set(matched['incident_id'].dropna().tolist()) if 'incident_id' in matched.columns else set()
        target_ids = set(inc['incident_id'].dropna().tolist()) if 'incident_id' in inc.columns else set()
        fn = len(target_ids - matched_ids)

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        evaluation = pd.DataFrame([{
            'total_waves': len(w),
            'incident_like_waves': len(target_wave),
            'matched_waves': len(matched),
            'tp_incident_like': tp,
            'fp_incident_like': fp,
            'fn_incidents': fn,
            'precision_incident_like': precision,
            'recall_incident_like': recall,
            'mean_kp_diff_km': matched['kp_diff_km'].mean() if 'kp_diff_km' in matched.columns else np.nan,
            'mean_time_gap_min': matched['time_gap_min'].mean() if 'time_gap_min' in matched.columns else np.nan
        }])
    else:
        evaluation = pd.DataFrame([{
            'total_waves': len(w),
            'incident_like_waves': int((w['wave_class'] == 'incident_like').sum()),
            'matched_waves': 0,
            'tp_incident_like': 0,
            'fp_incident_like': int((w['wave_class'] == 'incident_like').sum()),
            'fn_incidents': len(inc),
            'precision_incident_like': 0.0,
            'recall_incident_like': 0.0,
            'mean_kp_diff_km': np.nan,
            'mean_time_gap_min': np.nan
        }])

    type_eval = pd.DataFrame()
    if len(matched) and 'incident_name' in matched.columns:
        rows = []
        for nm, g in matched.groupby('incident_name'):
            rows.append({
                'incident_name': nm,
                'matched_rows': len(g),
                'mean_kp_diff_km': g['kp_diff_km'].mean() if 'kp_diff_km' in g.columns else np.nan,
                'mean_time_gap_min': g['time_gap_min'].mean() if 'time_gap_min' in g.columns else np.nan
            })
        type_eval = pd.DataFrame(rows)

    if save_outputs:
        matched.to_csv(os.path.join(output_dir, 'matched_incidents.csv'), index=False)
        evaluation.to_csv(os.path.join(output_dir, 'wave_incident_evaluation.csv'), index=False)
        if len(type_eval):
            type_eval.to_csv(os.path.join(output_dir, 'wave_incident_type_evaluation.csv'), index=False)
        logger.info(
            f"incident evaluation done matched={len(matched)} "
            f"precision={evaluation.iloc[0]['precision_incident_like']} "
            f"recall={evaluation.iloc[0]['recall_incident_like']}"
        )

    return matched, evaluation, type_eval

def run_one_scenario(input_dir, output_dir, cfg, logger, incident_df=None, produce_figures=True):
    files = sorted(glob.glob(os.path.join(input_dir, cfg['file_pattern'])))
    all_summary = []
    all_agg = []
    all_waves = []

    for f in files:
        direction = infer_direction_from_filename(f)
        if direction is None:
            logger.warning(f"[{cfg.get('_scenario_id', 'base')}] skipped (direction not inferred): {os.path.basename(f)}")
            continue

        try:
            date = infer_date_from_filename(f)
        except Exception as e:
            logger.exception(f"[{cfg.get('_scenario_id', 'base')}] date inference failed: {os.path.basename(f)}")
            stem = os.path.splitext(os.path.basename(f))[0]
            outdir = os.path.join(output_dir, stem) if produce_figures else output_dir
            os.makedirs(outdir, exist_ok=True)
            pd.DataFrame([{
                'scenario_id': cfg.get('_scenario_id', 'base'),
                'file_name': os.path.basename(f),
                'direction': direction,
                'error': str(e)
            }]).to_csv(os.path.join(outdir, 'error.csv'), index=False)
            continue

        stem = os.path.splitext(os.path.basename(f))[0]
        outdir = os.path.join(output_dir, stem) if produce_figures else output_dir
        os.makedirs(outdir, exist_ok=True)

        try:
            summary, agg, waves = process_one_file(
                f, outdir, cfg, logger,
                base_date=date, direction=direction,
                produce_figures=produce_figures
            )
            all_summary.append(summary)
            all_agg.append(agg)
            if len(waves):
                all_waves.append(waves)
        except Exception as e:
            logger.exception(f"[{cfg.get('_scenario_id', 'base')}] error processing file={os.path.basename(f)}")
            pd.DataFrame([{
                'scenario_id': cfg.get('_scenario_id', 'base'),
                'file_name': os.path.basename(f),
                'date': date,
                'direction': direction,
                'error': str(e)
            }]).to_csv(os.path.join(outdir, 'error.csv'), index=False)

    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    agg_all = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    waves_all = pd.concat(all_waves, ignore_index=True) if all_waves else pd.DataFrame()

    baseline, bottleneck, incidents, wave_refined, daydir = build_baseline_and_classify(
        agg_all, waves_all, output_dir, cfg, logger, save_outputs=produce_figures
    )

    matched, evaluation, type_eval = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if incident_df is not None and len(incident_df):
        matched, evaluation, type_eval = evaluate_with_incidents(
            wave_refined if len(wave_refined) else waves_all,
            incident_df,
            output_dir,
            cfg,
            logger,
            save_outputs=produce_figures
        )

    return summary_all, agg_all, waves_all, baseline, bottleneck, incidents, wave_refined, daydir, matched, evaluation, type_eval

def run_sensitivity(input_dir, output_dir, base_cfg, grid, logger, incident_df=None):
    sens_dir = os.path.join(output_dir, 'sensitivity_analysis')
    os.makedirs(sens_dir, exist_ok=True)
    variants = generate_config_variants(base_cfg, grid)
    logger.info(f'sensitivity scenarios={len(variants)}')

    compare_rows = []
    detail_rows = []

    for cfg in variants:
        scen_dir = os.path.join(sens_dir, cfg['_scenario_id'])
        os.makedirs(scen_dir, exist_ok=True)
        logger.info(f"start sensitivity {cfg['_scenario_id']} {cfg['_scenario_label']}")

        summary_all, agg_all, waves_all, baseline, bottleneck, incidents, wave_refined, daydir, matched, evaluation, type_eval = run_one_scenario(
            input_dir, scen_dir, cfg, logger, incident_df=incident_df, produce_figures=False
        )

        if len(summary_all):
            summary_all.to_csv(os.path.join(scen_dir, 'all_files_summary.csv'), index=False)
        if len(daydir):
            daydir.to_csv(os.path.join(scen_dir, 'daily_direction_processing_stats.csv'), index=False)
        if len(matched):
            matched.to_csv(os.path.join(scen_dir, 'matched_incidents.csv'), index=False)
        if len(evaluation):
            evaluation.to_csv(os.path.join(scen_dir, 'wave_incident_evaluation.csv'), index=False)
        if len(type_eval):
            type_eval.to_csv(os.path.join(scen_dir, 'wave_incident_type_evaluation.csv'), index=False)

        compare_rows.append({
            'scenario_id': cfg['_scenario_id'],
            'scenario_label': cfg['_scenario_label'],
            'files_processed': len(summary_all),
            'avg_removed_vehicles': summary_all['vehicle_ids_removed'].mean() if len(summary_all) else np.nan,
            'avg_speed_after_kmh': summary_all['avg_speed_after_kmh'].mean() if len(summary_all) else np.nan,
            'total_wave_events': summary_all['wave_events'].sum() if len(summary_all) else 0,
            'total_incident_like_waves': summary_all['incident_like_waves'].sum() if len(summary_all) else 0,
            'total_natural_waves': summary_all['natural_waves'].sum() if len(summary_all) else 0,
            'recurrent_bottlenecks': int(bottleneck['bottleneck_flag'].sum()) if len(bottleneck) else 0,
            'incident_candidates': len(incidents),
            'matched_waves': evaluation.iloc[0]['matched_waves'] if len(evaluation) else np.nan,
            'precision_incident_like': evaluation.iloc[0]['precision_incident_like'] if len(evaluation) else np.nan,
            'recall_incident_like': evaluation.iloc[0]['recall_incident_like'] if len(evaluation) else np.nan
        })

        if len(daydir):
            d = daydir.copy()
            d['scenario_id'] = cfg['_scenario_id']
            d['scenario_label'] = cfg['_scenario_label']
            detail_rows.append(d)

        logger.info(f"end sensitivity {cfg['_scenario_id']}")

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(os.path.join(sens_dir, 'scenario_comparison_summary.csv'), index=False)

    if detail_rows:
        detail_df = pd.concat(detail_rows, ignore_index=True)
        detail_df.to_csv(os.path.join(sens_dir, 'scenario_daily_direction_detail.csv'), index=False)
    else:
        detail_df = pd.DataFrame()

    if len(compare_df):
        plt.figure(figsize=(12, 5))
        plt.bar(compare_df['scenario_id'], compare_df['total_incident_like_waves'])
        plt.xticks(rotation=60)
        plt.ylabel('Incident-like waves')
        plt.title('Sensitivity analysis: incident-like waves by scenario')
        plt.tight_layout()
        plt.savefig(os.path.join(sens_dir, 'scenario_incident_wave_comparison.png'), dpi=160)
        plt.close()

    return compare_df, detail_df

def run_batch(input_dir, output_dir, config_path=None, sensitivity_grid_path=None, incident_dir=None):
    cfg = load_config(config_path)
    logger, log_path = setup_logger(output_dir)

    logger.info('===== traffic batch analysis started =====')
    logger.info(f'base config={json.dumps(cfg, ensure_ascii=False)}')

    incident_df = load_incident_dir(incident_dir, cfg, logger)
    if len(incident_df):
        incident_df.to_csv(os.path.join(output_dir, 'incident_master_normalized.csv'), index=False)

    summary_all, agg_all, waves_all, baseline, bottleneck, incidents, wave_refined, daydir, matched, evaluation, type_eval = run_one_scenario(
        input_dir, output_dir, cfg, logger, incident_df=incident_df, produce_figures=True
    )

    if len(summary_all):
        summary_all.to_csv(os.path.join(output_dir, 'all_files_summary.csv'), index=False)
    if len(agg_all):
        agg_all.to_csv(os.path.join(output_dir, 'all_aggregated.csv'), index=False)
    if len(waves_all):
        waves_all.to_csv(os.path.join(output_dir, 'all_wave_events_raw.csv'), index=False)
    if len(matched):
        matched.to_csv(os.path.join(output_dir, 'matched_incidents.csv'), index=False)
    if len(evaluation):
        evaluation.to_csv(os.path.join(output_dir, 'wave_incident_evaluation.csv'), index=False)
    if len(type_eval):
        type_eval.to_csv(os.path.join(output_dir, 'wave_incident_type_evaluation.csv'), index=False)

    if len(daydir):
        daydir.to_csv(os.path.join(output_dir, 'daily_direction_processing_stats.csv'), index=False)
        for _, r in daydir.iterrows():
            logger.info(
                f"daydir date={r['date']} dir={r['direction']} "
                f"avg_speed={r['avg_speed']:.2f} mean_shockwave={r['mean_shockwave']:.2f} "
                f"low_speed_cells={int(r['low_speed_cells'])} incident_cells={int(r['incident_candidate_cells'])}"
            )

    with open(os.path.join(output_dir, 'used_config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    grid = load_sensitivity_grid(sensitivity_grid_path)
    with open(os.path.join(output_dir, 'sensitivity_grid_used.json'), 'w', encoding='utf-8') as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)

    compare_df, detail_df = run_sensitivity(input_dir, output_dir, cfg, grid, logger, incident_df=incident_df)

    logger.info('===== traffic batch analysis finished =====')

    return summary_all, baseline, bottleneck, incidents, wave_refined, daydir, matched, evaluation, type_eval, compare_df, detail_df, log_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic batch analysis with fixed KP bounds, incident directory loading, route filtering, sensitivity analysis, and no hardcoded date fallback.')
    parser.add_argument('--input_dir', required=True, help='Input folder containing ITS Connect CSV files')
    parser.add_argument('--output_dir', required=True, help='Output folder')
    parser.add_argument('--config', default=None, help='Path to base JSON config file')
    parser.add_argument('--sensitivity_grid', default=None, help='Path to JSON sensitivity grid file')
    parser.add_argument('--incident_dir', default=None, help='Path to incident_daily_csv directory')
    args = parser.parse_args()

    summary_all, baseline, bottleneck, incidents, wave_refined, daydir, matched, evaluation, type_eval, compare_df, detail_df, log_path = run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        sensitivity_grid_path=args.sensitivity_grid,
        incident_dir=args.incident_dir
    )

    print({
        'processed_files': len(summary_all),
        'baseline_rows': len(baseline),
        'bottleneck_rows': len(bottleneck),
        'incident_candidate_rows': len(incidents),
        'wave_rows': len(wave_refined),
        'matched_incident_rows': len(matched),
        'evaluation_rows': len(evaluation),
        'type_evaluation_rows': len(type_eval),
        'sensitivity_scenarios': len(compare_df),
        'log_file': log_path
    })