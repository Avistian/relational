"""Complete 13-model x 97-task released-score reconstruction; original identity stays gated."""
import csv,hashlib,json,math,sys
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b19b';E=P/'evidence/b19b'
MODELS={'TiRex':'TiRex','Toto-1.0':'Toto_Open_Base_1.0','Moirai-2':'Moirai2','TabPFN-TS':'tabpfn_ts','TimesFM-2.0':'timesfm_2_0_500m','Chronos-Bolt-Base':'chronos_bolt_base','Sundial-Base':'sundial_base_128m','PatchTST':'PatchTST','TFT':'tft','DeepAR':'deepar','Auto-Arima':'auto_arima','Auto-Theta':'auto_theta','Seasonal-Naive':'seasonal_naive'}
PUBLISHED={'TiRex':[2.515,.413,.642],'Toto-1.0':[4.237,.437,.673],'Moirai-2':[4.464,.436,.654],'TabPFN-TS':[5.392,.460,.692],'TimesFM-2.0':[5.454,.465,.680],'Chronos-Bolt-Base':[5.866,.485,.725],'Sundial-Base':[6.402,.472,.673],'PatchTST':[6.943,.496,.762],'TFT':[7.222,.511,.822],'DeepAR':[9.624,.721,1.206],'Auto-Arima':[10.284,.770,.964],'Auto-Theta':[10.923,1.051,.978],'Seasonal-Naive':[11.675,1.,1.]}

def replay(records):
    """Equal task weight, geometric ratios and average tied WQL ranks; no missing imputation."""
    roster=list(records);keys=sorted(records['Seasonal-Naive'])
    if len(roster)!=13 or len(keys)!=97 or any(set(records[m])!=set(keys) for m in roster):raise ValueError('Incomplete model/task support')
    for model in roster:
        for key in keys:
            if any(not math.isfinite(records[model][key][m]) or records[model][key][m]<=0 for m in ['mase','wql']):raise ValueError('Nonpositive/nonfinite score')
    result=[]
    for model in roster:
        ranks=[]
        for key in keys:
            v=records[model][key]['wql'];values=[records[m][key]['wql'] for m in roster]
            ranks.append(1+sum(x<v for x in values)+(sum(x==v for x in values)-1)/2)
        row=dict(model=model,mean_wql_rank=sum(ranks)/len(keys))
        for metric in ['wql','mase']:
            row['relative_'+metric]=math.exp(sum(math.log(records[model][k][metric]/records['Seasonal-Naive'][k][metric]) for k in keys)/len(keys))
        result.append(row)
    return dict(tasks=len(keys),models=len(roster),records=len(keys)*len(roster),rows=result,weighting='equal task weight; not independent datasets',confidence_intervals='NOT_RECONSTRUCTED_ORIGINAL_METHOD_UNESTABLISHED')

def main():
    if '--fresh' in sys.argv:raise SystemExit('NOT_RUN: approved scope is saved-score replay; historical environment/backend identity unresolved. See b19b-reproduction.md.')
    manifest=json.loads((S/'manifest.json').read_text())
    for f in manifest['files']:
        if hashlib.sha256((S/f['file']).read_bytes()).hexdigest()!=f['sha256']:raise ValueError('SOURCE_HASH_MISMATCH '+f['file'])
    records={};audit=[]
    for name,folder in MODELS.items():
        path=S/'wrapper/gift_eval/submission/all_results.csv' if folder is None else S/'gift/results'/folder/'all_results.csv'
        rows=list(csv.DictReader(path.open()));mapped={}
        for r in rows:
            key=r['dataset']
            if key in mapped:raise ValueError('Duplicate task '+key)
            mapped[key]=dict(mase=float(r['eval_metrics/MASE[0.5]']),wql=float(r['eval_metrics/mean_weighted_sum_quantile_loss']))
        records[name]=mapped;audit.append(dict(model=name,path=str(path.relative_to(S)),rows=len(rows)))
    report=replay(records)
    for row in report['rows']:
        row['published']=dict(zip(['mean_wql_rank','relative_wql','relative_mase'],PUBLISHED[row['model']]))
        row['matches_3_decimals']={m:round(row[m],3)==v for m,v in row['published'].items()}
    report.update(status='COMPLETE_RELEASED_SCORE_RECONSTRUCTION',historical_target='INCOMPLETE_SOURCE_PROTOCOL_GATE',matched_fields=sum(sum(r['matches_3_decimals'].values()) for r in report['rows']),published_fields=39,pins=manifest['pins'],files=audit,unresolved=['Original Figure4.1 score-file and task-weight identity','Original confidence interval procedure, resampling unit, repetitions and seed','Original run environment, backend versions and exact checkpoint bytes used'],fresh_inference='NOT_RUN',pretraining='NOT_RUN',whole_paper='NOT_RUN')
    report['source_findings']=['Figure includes Sundial and PatchTST and excludes AutoETS despite prose roster.','Wrapper repository submission is dated January2025; GIFT TabPFN-TS was updated May2025. Primary replay uses the single pinned GIFT snapshot for all13 models.','GIFT README documents a July2025 Naive/Seasonal Naive correction. Archived pre-correction repo file has only4 tasks and cannot replace97.','Candidate Jan2026 wrapper defaults to client mode, unbounded dependencies; GIFT notebook references v1.0.0 in local mode.','Appendix names regression checkpoint; source defaults spell regressor. No original run/checkpoint-byte equivalence assumed.']
    (E/'released-scores.json').write_text(json.dumps(records,indent=2)+'\n');(E/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n');(E/'portable-replay.json').write_text(json.dumps(replay(records),indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
