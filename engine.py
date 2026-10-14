"""FlowForge: reproducible synthetic RTGS scheduling research prototype.

Integer monetary units, sequential gross settlement, no credit creation.
Only arrived payments and observed outages enter scheduling features.
"""
from dataclasses import dataclass, asdict
import copy
import numpy as np

@dataclass
class Payment:
    id: int
    src: int
    dst: int
    amount: int
    arrival: int
    deadline: int
    settled: int = -1

def scenario(seed, n=8, mode='normal', liquidity=100):
    rng = np.random.default_rng(seed)
    payments=[]
    for i in range(100):
        s=int(rng.integers(n))
        d=(s+1)%n if mode=='ring' else int(rng.integers(n-1))
        if mode!='ring' and d>=s: d+=1
        t=int(rng.integers(40))
        if mode=='burst': t=(t//10)*10
        if mode=='delay' and s==0: t=min(39,t+12)
        payments.append(Payment(i,s,d,int(rng.integers(10,101)),t,t+int(rng.integers(3,13))))
    return dict(seed=seed,n=n,mode=mode,liquidity=liquidity,
                balances=[liquidity]*n,payments=payments,
                outage=0 if mode=='outage' else -1)

def available(sc, balances, t):
    return [p for p in sc['payments'] if p.settled<0 and p.arrival<=t
            and not(p.src==sc['outage'] and 12<=t<30)
            and balances[p.src]>=p.amount]

def reward(p,t):
    # Shared objective: completion + urgency, without future knowledge.
    return 1 + .05*min(20,t-p.arrival) + (1 if t>=p.deadline else 0)

def cycle_batch(sc,balances,t):
    """Find a feasible directed cycle of 2--4 whole payments, FIFO search.

    This baseline has a stronger simultaneous-netting mechanism than RTGS
    policies; report it separately rather than attributing its gain to ranking.
    """
    queued=sorted([p for p in sc['payments'] if p.settled<0 and p.arrival<=t
                   and not(p.src==sc['outage'] and 12<=t<30)],key=lambda p:(p.arrival,p.id))
    outgoing={i:[] for i in range(sc['n'])}
    for p in queued:outgoing[p.src].append(p)
    def search(start,node,path,visited):
        for q in outgoing[node]:
            if q.dst==start and path:
                batch=path+[q]; b=balances.copy()
                for p in batch:b[p.src]-=p.amount;b[p.dst]+=p.amount
                if np.all(b>=0):return batch
            elif len(path)<3 and q.dst not in visited:
                found=search(start,q.dst,path+[q],visited|{q.dst})
                if found:return found
        return None
    for i in range(sc['n']):
        found=search(i,i,[],{i})
        if found:return found
    return None

def teacher(sc, balances, t, candidates):
    """Two-step constrained lookahead teacher; never examines future arrivals."""
    values=[]
    for p in candidates:
        b=balances.copy(); b[p.src]-=p.amount; b[p.dst]+=p.amount
        nxt=[q for q in available(sc,b,t) if q.id!=p.id]
        values.append(reward(p,t)+max((reward(q,t) for q in nxt),default=0))
    return np.asarray(values)

def features(sc,balances,t,candidates):
    n=sc['n']; x=np.zeros((n,6)); adj=np.eye(n)
    queued=[p for p in sc['payments'] if p.settled<0 and p.arrival<=t]
    x[:,0]=np.asarray(balances)/100
    for p in queued:
        x[p.src,1]+=p.amount/100; x[p.dst,2]+=p.amount/100
        x[p.src,3]+=1; x[p.dst,4]+=1
        adj[p.src,p.dst]+=p.amount/100
        adj[p.dst,p.src]+=p.amount/100
    if sc['outage']>=0 and 12<=t<30: x[sc['outage'],5]=1
    x[:,1:5]/=10
    # One graph convolution, normalized neighbor messages, with self-loop.
    agg=(adj/adj.sum(axis=1,keepdims=True))@x
    e=np.asarray([[p.amount/100,(t-p.arrival)/20,
                   (p.deadline-t)/20,balances[p.src]/100,
                   balances[p.dst]/100] for p in candidates])
    ids=np.asarray([[p.src,p.dst] for p in candidates],dtype=int)
    return agg,e,ids

class GraphPolicy:
    """One-hop graph convolution + edge-ranking head, trained in NumPy."""
    def __init__(self,seed=1,use_graph=True):
        rng=np.random.default_rng(seed)
        self.W=rng.normal(0,.15,(6,12)); self.v=rng.normal(0,.1,29)
        self.loss=[]; self.use_graph=use_graph
    def scores(self,f):
        a,e,ids=f; a=a if self.use_graph else np.zeros_like(a); h=np.tanh(a@self.W)
        z=np.concatenate([h[ids[:,0]],h[ids[:,1]],e],axis=1)
        return z@self.v
    def fit(self,data,epochs=12):
        rng=np.random.default_rng(7)
        for epoch in range(epochs):
            losses=[]
            for ix in rng.permutation(len(data)):
                (a,e,ids),target=data[ix]
                a=a if self.use_graph else np.zeros_like(a)
                h=np.tanh(a@self.W)
                z=np.concatenate([h[ids[:,0]],h[ids[:,1]],e],axis=1)
                logits=z@self.v; logits-=logits.max()
                pred=np.exp(logits); pred/=pred.sum()
                losses.append(float(-np.sum(target*np.log(pred+1e-12))))
                dl=pred-target; dv=z.T@dl
                dh=np.zeros_like(h)
                np.add.at(dh,ids[:,0],dl[:,None]*self.v[:12])
                np.add.at(dh,ids[:,1],dl[:,None]*self.v[12:24])
                dw=a.T@(dh*(1-h*h))
                self.v-=.005*np.clip(dv,-5,5)
                self.W-=.005*np.clip(dw,-5,5)
            self.loss.append(float(np.mean(losses)))
    def export(self): return {'W':self.W.tolist(),'v':self.v.tolist(),'loss':self.loss,'use_graph':self.use_graph}

def run(source,policy='fifo',model=None,collect=False,trace=False):
    sc=copy.deepcopy(source); b=np.asarray(sc['balances'],dtype=int)
    total=int(b.sum()); data=[]; events=[]; states=[]
    for t in range(60):
        while True:
            if policy=='cycle':
                batch=cycle_batch(sc,b,t)
                if batch:
                    for p in batch:b[p.src]-=p.amount;b[p.dst]+=p.amount;p.settled=t
                    assert np.all(b>=0) and int(b.sum())==total
                    if trace:
                        for p in batch:events.append(dict(time=t,payment=p.id,src=p.src,dst=p.dst,amount=p.amount,batch=True))
                    continue
            c=available(sc,b,t)
            if not c: break
            if policy in ['fifo','cycle']: p=min(c,key=lambda p:(p.arrival,p.id))
            elif policy=='edf': p=min(c,key=lambda p:(p.deadline,p.id))
            elif policy=='smallest': p=min(c,key=lambda p:(p.amount,p.id))
            elif policy=='lookahead':
                values=teacher(sc,b,t,c)
                p=c[int(np.argmax(values))]
                if collect and len(c)>1:
                    targets=np.exp((values-values.max())*3); targets/=targets.sum()
                    data.append((features(sc,b,t,c),targets))
            else:
                if model is None: raise ValueError('graph policy requires model')
                p=c[int(np.argmax(model.scores(features(sc,b,t,c))))]
            before=b.copy(); b[p.src]-=p.amount; b[p.dst]+=p.amount; p.settled=t
            assert np.all(b>=0) and int(b.sum())==total
            assert p.arrival<=t
            if trace: events.append(dict(time=t,payment=p.id,src=p.src,dst=p.dst,amount=p.amount))
        if trace: states.append(dict(time=t,balances=b.tolist(),settled=sum(p.settled>=0 for p in sc['payments'])))
    done=[p for p in sc['payments'] if p.settled>=0]
    missed=sum(p.settled<0 or p.settled>p.deadline for p in sc['payments'])
    metrics=dict(settled=len(done),missed=missed,
                 mean_delay=float(np.mean([p.settled-p.arrival for p in done])) if done else None,
                 # Count all unfinished as delayed to horizon; avoids survivor bias.
                 censored_delay=float(np.mean([(p.settled if p.settled>=0 else 60)-p.arrival for p in sc['payments']])),
                 initial_liquidity=total)
    return metrics,data,dict(events=events,states=states,payments=[asdict(p) for p in sc['payments']])
