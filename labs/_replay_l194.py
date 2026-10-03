"""Offline full evidence report. No backend calls, patches or downloads."""
import hashlib,json
from pathlib import Path
from html.parser import HTMLParser
from relkit.report_l194 import oriented_gap,evidence_license,summarize_comparisons


def build_report(packet, manifest, gap_fn, license_fn, summary_fn):
    import hashlib,json,itertools
    from collections import Counter
    from html.parser import HTMLParser
    for name,digest in manifest['files'].items():
        if Path(name).name != name or hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Packet integrity failure: '+name)
    read=lambda name:json.loads((packet/name).read_text())
    tasks=read('tasks.json');prior=read('prior-report.json');protocol=read('protocol.json');runs=read('runs.json')
    if len(tasks)!=21 or len({t['id'] for t in tasks})!=21:raise ValueError('Incomplete task set')
    prior_manifest=read('prior-manifest.json')
    for name,digest in prior_manifest['files'].items():
        if manifest['files'].get(name)!=digest:raise ValueError('Original packet identity changed')
    if protocol['seeds']!=[0,1,2] or protocol['depths']!=[2,3,4] or protocol['backends']!=['TabPFN-v2','TabPFN-v2.5','LimiX-16M']:raise ValueError('Protocol changed')
    expected=Counter((t['id'],s) for t in tasks for s in protocol['seeds'])
    if Counter((r['task'],r['seed']) for r in runs)!=expected:raise ValueError('Incomplete seed set')
    if any(r['status']!='NOT_RUN' or r['score'] is not None for r in runs):raise ValueError('Frozen unrun ledger changed')
    if prior['executed_model_evaluations']!=0 or prior['status']!='INCOMPLETE_SOURCE_PREPROCESSING_GATE':raise ValueError('Prior status changed')
    if Counter(r['task'] for r in prior['task_results'])!=Counter(t['id'] for t in tasks):raise ValueError('Prior task set mismatch')
    if any(r['mean'] is not None or r['sample_sd'] is not None or r['completed_seeds']!=0 for r in prior['task_results']):raise ValueError('Invented measured result')
    if len(read('validation-schedule.json'))!=567 or len(read('test-schedule.json'))!=63:raise ValueError('Schedule coverage changed')
    class Tables(HTMLParser):
        def __init__(self):super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
        def handle_starttag(self,tag,attrs):
            if tag=='table':self.table=[]
            if tag=='tr' and self.table is not None:self.row=[]
            if tag in ('td','th') and self.row is not None:self.cell=[]
        def handle_data(self,data):
            if self.cell is not None:self.cell.append(data)
        def handle_endtag(self,tag):
            if tag in ('td','th') and self.cell is not None:self.row.append(' '.join(''.join(self.cell).split()));self.cell=None
            if tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
            if tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None
    parser=Tables();parser.feed((packet/'paper.html').read_text());references={};table_id=0
    comparator=read('analysis-protocol.json')['comparator']
    if comparator!='AutoGluon+DFS':raise ValueError('Unapproved comparator')
    for table in parser.tables:
        header=next((r for r in table if 'Dataset' in r and 'RDBLearn' in r),None)
        if header is None:continue
        table_id+=1;dataset=''
        for row in table[table.index(header)+1:]:
            if len(row)!=len(header):continue
            dataset=row[0] or dataset;task=row[1].split(' (')[0].lower();key=(table_id,dataset.lower(),task)
            if key in references:raise ValueError('Duplicate paper row')
            references[key]=(float(row[header.index('RDBLearn')]),float(row[header.index(comparator)]))
    if len(references)!=21:raise ValueError('Paper extraction coverage')
    rows=[]
    for task in tasks:
        model,base=references[(task['paper_table'],task['dataset'],task['task'])]
        if model!=task['paper_score']:raise ValueError('Target mismatch')
        gap=gap_fn(task['metric'],model,base)
        rows.append(dict(task=task['id'],metric=task['metric'],group=task['group'],paper_model=model,paper_comparator=base,gap=gap,published_license=license_fn('PUBLISHED_TABLE'),fresh_gap=gap_fn(task['metric'],None,None),fresh_mean=None,fresh_sd=None,completed_seeds=0,status='NOT_RUN',performance_license=license_fn('UNRUN_BENCHMARK'),attribution='NOT_ESTABLISHED'))
    probe=read('preprocessing.json');observations=probe['observations']
    changed=[sum(a!=b for a,b in zip(o['before'],o['after'])) for o in observations]
    if changed!=[3,0,3,0] or any(o['numeric_before']!=o['numeric_after'] for o in observations):raise ValueError('Diagnostic mismatch')
    plans=[dict(axis='label_coverage',hypothesis='More available labeled support improves prediction',vary='Nested support sizes sampled only from labels available at each cutoff',hold_fixed='Same queries, features, checkpoint, depth, preprocessing, seed pairing and evaluation rule',measure='Per-task paired AUROC or MAE change across complete seeds',limitation='Context sensitivity is not pretraining scale or cross-database superiority'),dict(axis='schema_information',hypothesis='A specified join path supplies useful predictive information',vary='Remove that path under a declared feature intervention',hold_fixed='Same queries, support labels, checkpoint, preprocessing policy, seed pairing and remaining paths',measure='Per-task paired score change with and without the path',limitation='Ablation tests path information under this pipeline, not arbitrary schema robustness'),dict(axis='cold_start',hypothesis='Performance differs for entities unseen in support history',vary='Prespecified cold/warm strata using only past entity membership',hold_fixed='Same fitted predictor, task cutoff policy and scoring implementation',measure='Stratum sizes, label counts and separate scores; AUROC undefined for single-class strata',limitation='Observational subgroup comparison; degree, time and labels may confound the difference')]
    for plan in plans:plan.update(status='NOT_RUN',license=license_fn('PROPOSED_INTERVENTION'))
    return dict(name='L194 complete analysis and report',status=prior['status'],task_count=21,validation_slots=567,test_slots=63,executed_model_evaluations=0,comparator=comparator,reference_summary=summary_fn(rows,[t['id'] for t in tasks]),task_results=rows,source_diagnostic=dict(scope='Replay of L193 synthetic observations',changed_known_codes=changed,license=license_fn('SAVED_SOURCE_DIAGNOSTIC'),real_task_occurrence='NOT_ESTABLISHED',score_effect='NOT_ESTABLISHED'),intervention_plans=plans,regression_normalization='NOT_ESTABLISHED',fresh_inference='NOT_RUN',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',cloud_usd=0)


def report_markdown(report):
    lines=['# RDBLearn reproduction — analysis report','', '**Model reproduction: '+report['status']+'**. Completed tasks: 0/21; model evaluations: 0/630.','', '## Question and protocol','', 'Can the complete RDBLearn v1 Tables 1–3 column be reproduced? The frozen L193 plan uses all 21 tasks, three course seeds, depths 2/3/4 and TabPFNv2/v2.5/LimiX-16M: 567 validation candidates, 63 selected tests. Source commit b5b03ebf8091547285a6e06cba53d2d1a40cb171, release 0.1.2, FastDFS 0.2.1; official splits and support cap 10000. Course seeds are a disclosed extension.','', '## Results and uncertainty','', 'All fresh scores and their seed SDs remain null. Positive reference gaps favor RDBLearn; AUROC uses model minus comparator, MAE uses comparator minus model. Comparator fixed before analysis: AutoGluon+DFS. These are rounded published values, not fresh or paired measurements. No statistical significance or causal claim follows. No raw-MAE or mixed-metric average is computed.','', '| Task | Metric | Published RDBLearn | Published AutoGluon+DFS | Oriented reference gap | Fresh result | Attribution |','|---|---|---:|---:|---:|---|---|']
    for row in report['task_results']:lines.append(f"| {row['task']} | {row['metric']} | {row['paper_model']:.4f} | {row['paper_comparator']:.4f} | {row['gap']:+.4f} | NOT_RUN | NOT_ESTABLISHED |")
    s=report['reference_summary'];lines+=['',f"At printed precision: {s['higher']} favorable, {s['lower']} unfavorable, {s['equal_at_printed_precision']} equal. Counts give each task one vote; dependent tasks and heterogeneous methods prevent a causal or significance interpretation.",'','## Diagnostic and explanation boundary','','Recorded synthetic source observations change known category codes in two of four interventions; numeric controls are unchanged. This justifies the declared source-admission stop. It does not establish occurrence or score impact on any real task. L194 replays these observations; it does not freshly execute the upstream preprocessor.','','## Testable follow-up studies (all NOT_RUN)']
    for plan in report['intervention_plans']:lines+=['', '### '+plan['axis'], '', *[f"**{k.replace('_',' ').capitalize()}:** {plan[k]}" for k in ('hypothesis','vary','hold_fixed','measure','limitation')]]
    lines+=['','## Limitations and resumption','','Resolve source/preprocessing identity in a separately named repair protocol; audit raw data, labels, features and temporal availability; authenticate checkpoint bytes; establish regression denominators; implement and validate the backend executor; measure full-grid cost before dispatch. Preserve all tasks, seeds and candidates. No backend pretraining or comparator retraining is included. Whole-paper reproduction remains NOT_RUN and historical identity NOT_ESTABLISHED.','','L194 approved budget: $0 cloud/API and 1800 aggregate local execution seconds. L193’s $10 ceiling was not a demonstrated full-run cost estimate. Learner defense remains PENDING_WRITTEN_DEFENSE.','','Primary source: [RDBLearn v1](https://arxiv.org/html/2602.18495v1). Input hashes: [manifest](input-manifest.json). Structured result: [report](report.json).']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    E=Path(__file__).resolve().parent/'evidence/l194'
    report=build_report(E/'packet',json.loads((E/'input-manifest.json').read_text()),oriented_gap,evidence_license,summarize_comparisons)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n');(E/'report.md').write_text(report_markdown(report));print(report['reference_summary']);print(report['status'])
