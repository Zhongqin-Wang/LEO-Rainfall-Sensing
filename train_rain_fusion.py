"""Train point rain occurrence and rain-rate heads with chronological group holdouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.rain_fusion import RainFusion


def load_data(path):
    with np.load(path, allow_pickle=False) as d:
        data = {k:d[k].copy() for k in d.files}
    x, cloud, y = data['csi'], data['cloud'], data['rain_rate_mm_h']
    if x.ndim != 3 or cloud.ndim != 4 or y.shape != (len(x),) or len(cloud) != len(x) or not len(x):
        raise ValueError('Expected CSI [N,T,F], cloud [N,C,H,W], rain [N]')
    if min(cloud.shape[-2:]) < 4 or x.shape[-1] < 2 or x.shape[1] < 1:
        raise ValueError('Invalid input dimensions')
    if not all(np.isfinite(a).all() for a in (x, cloud, y)) or np.any(y < 0):
        raise ValueError('Inputs/labels must be finite; rain must be nonnegative')
    if not np.isin(x[:, :, -1], [0, 1]).all() or not (x[:, :, -1].sum(axis=1) > 0).all():
        raise ValueError('Last CSI feature must be observed-bin mask, at least one bin per sample')
    for k in ('history_start_utc','issue_time_utc','target_time_utc','event_group'):
        if data[k].shape != (len(x),):
            raise ValueError('Invalid metadata: '+k)
    return data


def chronological_split(data, embargo_minutes=60):
    if embargo_minutes < 0:
        raise ValueError('Embargo must be nonnegative')
    issue = pd.to_datetime(data['issue_time_utc'], utc=True)
    target = pd.to_datetime(data['target_time_utc'], utc=True)
    start = pd.to_datetime(data['history_start_utc'], utc=True)
    if np.any(target <= issue) or np.any(start > issue):
        raise ValueError('Invalid forecast timing')
    group = data['event_group'].astype(str)
    unique = sorted(set(group), key=lambda g: issue[group == g].min())
    if len(unique) < 3:
        raise ValueError('Need at least three independent chronological event groups')
    # Entire groups assigned together; interleaved/overlapping groups are rejected.
    for a,b in zip(unique, unique[1:]):
        if issue[group == a].max() >= issue[group == b].min():
            raise ValueError('Interleaved event groups; regroup whole weather events')
    ntest = max(1, len(unique)//5)
    nval = max(1, len(unique)//5)
    train_groups, val_groups, test_groups = unique[:-ntest-nval], unique[-ntest-nval:-ntest], unique[-ntest:]
    masks = [np.isin(group, g) for g in (train_groups, val_groups, test_groups)]
    gap = pd.Timedelta(minutes=embargo_minutes)
    # Purge earlier samples whose label/history would overlap the next split.
    masks[0] &= target < start[masks[1]].min()-gap
    masks[1] &= target < start[masks[2]].min()-gap
    if not all(m.any() for m in masks):
        raise ValueError('Empty split after history/target embargo; collect more independent dates')
    return masks


def normalization(csi, train):
    values = csi[train, :, :-1][csi[train, :, -1] == 1]
    mean, scale = values.mean(axis=0), values.std(axis=0)
    return mean, np.maximum(scale, 1e-5)


def normalize(csi, mean, scale):
    x = csi.copy().astype('float32')
    x[:, :, :-1] = (x[:, :, :-1]-mean)/scale
    x[:, :, :-1] *= x[:, :, -1:]
    return x


def metrics(prob, rate, y, threshold):
    truth, pred = y >= threshold, prob >= .5
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    return dict(samples=len(y), rain_events=int(truth.sum()),
        brier=float(np.mean((prob-truth)**2)),
        csi_critical_success_index=tp/(tp+fp+fn) if tp+fp+fn else None,
        mae_mm_h=float(np.mean(abs(rate-y))), rmse_mm_h=float(np.sqrt(np.mean((rate-y)**2))))


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.rain_threshold <= 0 or args.rate_weight < 0:
        raise ValueError('Invalid training parameters')
    torch.set_num_threads(2)
    data = load_data(args.samples)
    masks = chronological_split(data, args.embargo_minutes)
    y = data['rain_rate_mm_h'].astype('float32')
    mean, scale = normalization(data['csi'], masks[0])
    x = normalize(data['csi'], mean, scale)
    cloud = data['cloud'].astype('float32')
    args.output.mkdir(parents=True, exist_ok=True)
    reports, prediction_tables = {}, []
    demo = bool(data.get('synthetic_demo', False))
    for mode in args.modes:
        torch.manual_seed(args.seed)
        model = RainFusion(x.shape[-1], cloud.shape[1], mode)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        def loss(logit, lograte, target):
            return nn.functional.binary_cross_entropy_with_logits(logit, (target >= args.rain_threshold).float()) + args.rate_weight*nn.functional.smooth_l1_loss(lograte, torch.log1p(target))
        tensors = [torch.from_numpy(a) for a in (x, cloud, y)]
        train_ids = np.flatnonzero(masks[0])
        loader = DataLoader(TensorDataset(*(a[train_ids] for a in tensors)), batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        val_ids = np.flatnonzero(masks[1])
        best, best_epoch, history = float('inf'), 0, []
        checkpoint = args.output/(mode+'.pt')
        for epoch in range(args.epochs):
            model.train(); total = 0.
            for bx, bc, by in loader:
                optimizer.zero_grad()
                value = loss(*model(bx, bc), by)
                value.backward(); optimizer.step()
                total += float(value.detach())*len(by)
            model.eval()
            with torch.no_grad():
                value = float(loss(*model(tensors[0][val_ids], tensors[1][val_ids]), tensors[2][val_ids]))
            if not np.isfinite(value):
                raise ValueError('Nonfinite validation loss')
            history.append(dict(epoch=epoch+1, train_loss=total/len(train_ids), validation_loss=value))
            if value < best:
                best, best_epoch = value, epoch+1
                torch.save(dict(state_dict=model.state_dict(), mode=mode,
                    csi_features=x.shape[-1], cloud_channels=cloud.shape[1],
                    csi_mean=torch.from_numpy(mean), csi_scale=torch.from_numpy(scale),
                    rain_threshold_mm_h=args.rain_threshold, synthetic_demo=demo,
                    label_semantics='future station-grid instantaneous rain rate, mm/h',
                    input_csi_steps=x.shape[1], input_cloud_shape=list(cloud.shape[1:]),
                    train_issue_times=data['issue_time_utc'][masks[0]].tolist()), checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['state_dict']); model.eval()
        with torch.no_grad():
            logits, lr = model(tensors[0], tensors[1])
            prob, rate = logits.sigmoid().numpy(), torch.expm1(lr).numpy()
        reports[mode] = dict(best_epoch=best_epoch, best_validation_loss=best, history=history)
        for name, mask in zip(('train','validation','test'), masks):
            reports[mode][name] = metrics(prob[mask], rate[mask], y[mask], args.rain_threshold)
            prediction_tables.append(pd.DataFrame(dict(mode=mode, split=name,
                issue_time_utc=data['issue_time_utc'][mask], target_time_utc=data['target_time_utc'][mask],
                actual_rain_rate_mm_h=y[mask], rain_probability=prob[mask], predicted_rain_rate_mm_h=rate[mask])))
        print(mode, json.dumps(reports[mode]['test']), flush=True)
    prior = float(np.mean(y[masks[0]] >= args.rain_threshold)); average = float(y[masks[0]].mean())
    report = dict(synthetic_demo=demo, models=reports, seed=args.seed,
        rain_threshold_mm_h=args.rain_threshold, embargo_minutes=args.embargo_minutes,
        label_semantics='future station-grid instantaneous rain rate, not accumulated rainfall',
        split_groups={name:sorted(set(data['event_group'][mask].tolist())) for name,mask in zip(('train','validation','test'),masks)},
        constant_training_baseline_test=metrics(np.full(masks[2].sum(), prior), np.full(masks[2].sum(), average), y[masks[2]], args.rain_threshold),
        notes=['Synthetic smoke tests do not establish rainfall forecast skill.' if demo else 'Offline experiment: verify source availability, event independence and CSI provenance.',
               'Best epoch selected using validation only; test never selects a checkpoint.',
               'Rain probability means rate >= configured threshold at target time, not any rain over an interval.'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    pd.concat(prediction_tables, ignore_index=True).to_csv(args.output/'predictions.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', choices=['csi','cloud','fusion'], default=['csi','cloud','fusion'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--rain-threshold', type=float, default=.1)
    p.add_argument('--rate-weight', type=float, default=1.)
    p.add_argument('--embargo-minutes', type=float, default=60)
    p.add_argument('--seed', type=int, default=42)
    train(p.parse_args())
