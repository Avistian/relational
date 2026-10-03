"""Full course experiment; functions are inlined into the portable notebook."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


def load187(packet, manifest):
    for name,digest in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Input hash mismatch: '+name)
    return {p.stem:pd.read_parquet(p) for p in sorted((packet/'db').glob('*.parquet'))}


def experiment187(db, config, owned_counts, bounded_histogram, release_scale):
    if config['caps']!=[1,5,20] or config['epsilons']!=[.5,1.,2.] or config['seeds']!=list(range(30)):
        raise ValueError('Changed frozen experiment')
    counts=owned_counts(db)
    events=db['results'];domain=sorted(db['constructors'].constructorId.tolist())
    raw=events.constructorId.value_counts().reindex(domain,fill_value=0).to_numpy(dtype=np.int64)
    releases=[];summaries=[];histograms={}
    for cap in config['caps']:
        clipped=bounded_histogram(events,domain,cap);histograms[str(cap)]=clipped.tolist()
        for epsilon in config['epsilons']:
            spec=release_scale(cap,epsilon,30);noise_mae=[];raw_mae=[]
            for seed in config['seeds']:
                rng=np.random.default_rng(np.random.SeedSequence([187,cap,int(epsilon*10),seed]))
                noise=rng.laplace(0,spec['scale'],len(domain));released=clipped+noise
                a=float(np.mean(np.abs(released-clipped)));b=float(np.mean(np.abs(released-raw)))
                releases.append({'cap':cap,'epsilon':epsilon,'seed':seed,'values':released.tolist(),
                                 'mae_clipped':a,'mae_raw':b})
                noise_mae.append(a);raw_mae.append(b)
            summaries.append({'cap':cap,'epsilon':epsilon,'scale':spec['scale'],
                'kept':int(clipped.sum()),'clipping_l1':int(np.abs(clipped-raw).sum()),
                'signed_bias_per_bin':float((clipped-raw).mean()),
                'noise_mae_mean':float(np.mean(noise_mae)),'noise_mae_sd':float(np.std(noise_mae,ddof=1)),
                'raw_mae_mean':float(np.mean(raw_mae)),'raw_mae_sd':float(np.std(raw_mae,ddof=1))})
    maximal=counts.sort_values(['owned_rows','driverId'],ascending=[False,True]).iloc[0]
    report={'experiment':config['experiment'],'status':'COMPLETE_COURSE_EXPERIMENT',
        'tables':{k:len(v) for k,v in db.items()},'drivers':len(counts),'owned_rows':int(counts.owned_rows.sum()),
        'driver_edges':int(counts.driver_incident_edges.sum()),'owned_edges':int(counts.owned_incident_edges.sum()),
        'maximum_owner':{k:int(v) for k,v in maximal.to_dict().items()},
        'domain':domain,'raw_histogram':raw.tolist(),'clipped_histograms':histograms,'summaries':summaries,
        'releases':len(releases),'released_coordinates':len(releases)*len(domain),
        'hypothetical_basic_composition_epsilon':float(30*3*sum(config['epsilons'])),
        'whole_paper':'NOT_RUN','production_dp':'NOT_ESTABLISHED','erasure':'NOT_ESTABLISHED',
        'learner':'PENDING_WRITTEN_DEFENSE','cloud_usd':0}
    return report,counts,releases


if __name__=='__main__':
    from relkit.privacy_l187 import owned_counts,bounded_histogram,release_scale
    P=Path(__file__).resolve().parent;E=P/'evidence/l187';packet=E/'packet'
    db=load187(packet,json.loads((E/'input-manifest.json').read_text()))
    report,counts,releases=experiment187(db,json.loads((packet/'config.json').read_text()),owned_counts,bounded_histogram,release_scale)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (E/'releases.json').write_text(json.dumps(releases,separators=(',',':'))+'\n')
    counts.to_csv(E/'driver-audit.csv',index=False)
    print({k:report[k] for k in ['status','drivers','owned_rows','owned_edges','maximum_owner','releases','released_coordinates']})
