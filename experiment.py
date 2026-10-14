"""Run: python experiment.py (requires numpy). Seeds and splits are fixed."""
import json, pathlib, time
import numpy as np
from engine import scenario,run,GraphPolicy

ROOT=pathlib.Path(__file__).parent
def main():
    start=time.time(); data=[]
    # Normal-only and stress-augmented models share initialization and epochs.
    models={}
    for name,modes in [('graph_normal',['normal']),('graph_stress',['normal','outage','delay','burst']),('edge_stress',['normal','outage','delay','burst'])]:
        data=[]
        for seed in range(64):
            sc=scenario(seed,mode=modes[seed%len(modes)],liquidity=[50,75,100,125][(seed//4)%4])
            data.extend(run(sc,'lookahead',collect=True)[1])
        model=GraphPolicy(use_graph=name!='edge_stress'); model.fit(data)
        models[name]=model
        print(name,'states',len(data),'loss',model.loss[-1],flush=True)
    policies=['fifo','edf','smallest','cycle','lookahead','graph_normal','graph_stress','edge_stress']
    rows=[]; demos={}
    for mode in ['normal','outage','delay','burst','ring','larger']:
        for seed in range(1000,1048):
            sc=scenario(seed,n=12 if mode=='larger' else 8,mode='normal' if mode=='larger' else mode,liquidity=75)
            for policy in policies:
                metrics,_,tr=run(sc,policy,model=models.get(policy),trace=seed==1000)
                rows.append(dict(mode=mode,seed=seed,policy=policy,**metrics))
                if seed==1000: demos[f'{mode}:{policy}']=dict(n=sc['n'],**metrics,**tr)
    summaries=[]
    for mode in ['normal','outage','delay','burst','ring','larger']:
        for policy in policies:
            rs=[r for r in rows if r['mode']==mode and r['policy']==policy]
            summary=dict(mode=mode,policy=policy)
            for key in ['settled','missed','censored_delay']:
                a=np.array([r[key] for r in rs]); summary[key]=float(a.mean())
                summary[key+'_se']=float(a.std(ddof=1)/np.sqrt(len(a)))
            summaries.append(summary)
    # Paired intervals over the same generated scenarios; descriptive synthetic evidence.
    comparisons=[]
    for mode in ['normal','outage','delay','burst','ring','larger']:
        for baseline in ['fifo','edf','smallest','cycle','lookahead','graph_normal','edge_stress']:
            a={r['seed']:r['settled'] for r in rows if r['mode']==mode and r['policy']=='graph_stress'}
            b={r['seed']:r['settled'] for r in rows if r['mode']==mode and r['policy']==baseline}
            d=np.array([a[s]-b[s] for s in sorted(a)])
            comparisons.append(dict(mode=mode,baseline=baseline,mean_delta=float(d.mean()),ci95_halfwidth=float(1.96*d.std(ddof=1)/np.sqrt(len(d)))))
    payload=dict(name='FlowForge',generated_at='2026-10-14',train_seeds=[0,63],test_seeds=[1000,1047],
                 rows=rows,summaries=summaries,comparisons=comparisons,demos=demos,models={k:m.export() for k,m in models.items()},elapsed_seconds=time.time()-start)
    (ROOT/'results.json').write_text(json.dumps(payload),encoding='utf8')
    (ROOT/'results.js').write_text('window.FLOWFORGE_RESULTS='+json.dumps(payload)+';',encoding='utf8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
