#!/usr/bin/env python3
import json,re,os,statistics as st,warnings,math
warnings.filterwarnings("ignore")
import jsonschema
R='results'
def load(base,tag=''):
    pat=re.compile('^'+re.escape(base)+r'(_off\d+)?'+(('_'+re.escape(tag)) if tag else '')+r'\.jsonl$')
    rows={}
    for f in sorted(os.listdir(R)):
        if pat.match(f):
            for ln in open(os.path.join(R,f)):
                ln=ln.strip()
                if ln: r=json.loads(ln); rows.setdefault(r['instance_id'],r)
    return rows
def leaves(o):
    if isinstance(o,dict): return sum(leaves(v) for v in o.values())
    if isinstance(o,list): return sum(leaves(v) for v in o)
    return 1
def pct(v,p,how='linear'):
    if not v: return float('nan')
    v=sorted(v)
    if how=='linear':
        k=(len(v)-1)*p; f=int(k); c=min(f+1,len(v)-1); return v[f]+(v[c]-v[f])*(k-f)
    if how=='higher':
        k=(len(v)-1)*p; return v[min(len(v)-1,math.ceil(k))]
    if how=='nearestrank':
        return v[min(len(v)-1,math.ceil(p*len(v))-1)]
def ran(rows): return {k:r for k,r in rows.items() if (r.get('time_taken') or 0)>=0.01}
SHARED_IDS={}
def score(rows,denom,fld='extracted',ids=None,how='linear'):
    rr=ran(rows)
    if ids is not None: rr={k:v for k,v in rr.items() if k in ids}
    ok=cont=to=0; times=[]; rs=0.0
    for r in rr.values():
        t=r['time_taken']; times.append(t)
        if t>=119.99: to+=1
        rs+=(r.get('resamples') or 0)
        try: doc=json.loads(r.get(fld) or r.get('extracted') or '')
        except Exception: continue
        sch=r.get('schema')
        try: sch=json.loads(sch) if isinstance(sch,str) else sch
        except Exception: continue
        try: jsonschema.validate(doc,sch)
        except Exception: continue
        ok+=1
        if leaves(doc)>=5: cont+=1
    P=lambda x:100.0*x/denom
    return dict(n=denom,ran=len(rr),schema=P(ok),content=P(cont),mean=st.mean(times) if times else float('nan'),
                p50=(st.median(times) if times else float('nan')),p95=pct(times,.95,how),p99=pct(times,.99,how),TO=to,rs=rs/denom)
SPEC={
 'easy':[('Vanilla',('vanilla_timed_jsb_easy_s0_t128',''),'extracted','full'),
         ('CD-CFG',('igcd_timed_jsb_easy_s0_t128',''),'autocompletion','own'),
         ('LAVE',('lave_timed_jsb_easy_s0_t128','easylave'),'extracted','shared'),
         ('no-DP',('v2_async_ac4_timed_jsb_easy_s0_t128','easybase'),'extracted','shared'),
         ('DPG',('dp_jsb_easy_s0_t128','easydp'),'extracted','shared')],
 'medium':[('Vanilla',('vanilla_timed_jsb_medium_s0_t128',''),'extracted','full'),
         ('CD-CFG',('igcd_timed_jsb_medium_s0_t128',''),'autocompletion','own'),
         ('LAVE',('lave_timed_jsb_medium_s0_t128','v6lave'),'extracted','shared'),
         ('no-DP',('v2_async_ac4_timed_jsb_medium_s0_t128','v6base'),'extracted','shared'),
         ('DPG',('dp_jsb_medium_s0_t128','v6dp'),'extracted','shared')],
 'hard':[('Vanilla',('vanilla_timed_jsb_hard_s0_t128',''),'extracted','full'),
         ('CD-CFG',('igcd_timed_jsb_hard_s0_t128',''),'autocompletion','own'),
         ('LAVE',('lave_timed_jsb_hard_s0_t128','hardlave'),'extracted','shared'),
         ('no-DP',('v2_async_ac4_timed_jsb_hard_s0_t128','hardbase3'),'extracted','shared'),
         ('DPG',('dp_jsb_hard_s0_t128','harddp3'),'extracted','shared')],
 'JME':[('Vanilla',('vanilla_timed_jsonschema_s0_t128',''),'extracted','full'),
         ('CD-CFG',('igcd_timed_jsonschema_s0_t128',''),'autocompletion','own'),
         ('LAVE',('lave_timed_jsonschema_s0_t128','jmlave'),'extracted','shared'),
         ('no-DP',('v2_async_ac4_timed_jsonschema_s0_t128','jmbase'),'extracted','shared'),
         ('DPG',('dp_jsonschema_s0_t128','jmdp'),'extracted','shared')],
}
DEN={'easy':558,'medium':511,'hard':269,'JME':272}
PAPER={
('easy','Vanilla'):(577,62.9,30.2,8.34,8.37,11.23,14.98,0,None),
('easy','CD-CFG'):(449,79.1,36.3,6.22,6.72,11.54,15.98,0,49.2),
('easy','LAVE'):(558,80.1,40.5,30.63,6.71,120.01,120.01,94,207.3),
('easy','no-DP'):(558,97.5,45.2,7.58,7.58,11.47,15.09,0,20.7),
('easy','DPG'):(558,97.7,49.3,5.31,3.40,15.11,29.06,0,0.6),
('medium','Vanilla'):(586,49.8,47.1,14.89,14.36,21.82,31.67,0,None),
('medium','CD-CFG'):(383,64.8,58.0,22.71,14.16,112.24,120.00,19,57.3),
('medium','LAVE'):(511,78.3,74.0,38.50,25.70,120.00,120.01,60,253.5),
('medium','no-DP'):(511,91.8,74.6,13.45,13.40,21.43,29.99,0,33.2),
('medium','DPG'):(511,96.7,84.9,16.75,15.05,36.21,63.24,0,1.2),
('hard','Vanilla'):(368,20.7,20.4,44.64,35.43,103.48,120.00,4,None),
('hard','CD-CFG'):(245,32.2,29.8,57.61,41.53,120.00,120.15,72,50.2),
('hard','LAVE'):(269,39.8,39.4,40.47,38.35,120.00,120.00,30,115.8),
('hard','no-DP'):(269,74.3,53.9,31.41,26.69,61.91,101.51,0,56.0),
('hard','DPG'):(269,87.7,71.7,42.66,35.24,96.58,115.96,2,1.5),
}
# shared ids per split = instances the llguidance arms actually ran
for sp,(b,t) in [('easy',('v2_async_ac4_timed_jsb_easy_s0_t128','easybase')),
                 ('medium',('v2_async_ac4_timed_jsb_medium_s0_t128','v6base')),
                 ('hard',('v2_async_ac4_timed_jsb_hard_s0_t128','hardbase3')),
                 ('JME',('v2_async_ac4_timed_jsonschema_s0_t128','jmbase'))]:
    SHARED_IDS[sp]=set(ran(load(b,t)))
import sys
HOW=sys.argv[1] if len(sys.argv)>1 else 'linear'
cols=['n','schema','content','mean','p50','p95','p99','TO','rs']
print(f"percentile method: {HOW}")
print(f"{'split':<7}{'arm':<8}"+"".join(f"{c:>10}" for c in cols))
print('-'*100)
nbad=0
for sp,arms in SPEC.items():
    for name,(base,tag),fld,mode in arms:
        rows=load(base,tag)
        ids=None
        if mode=='full': denom=len(rows)
        elif mode=='own': denom=len(ran(rows))
        else: denom=DEN[sp]; ids=SHARED_IDS[sp]
        d=score(rows,denom,fld,ids,HOW)
        pap=PAPER.get((sp,name))
        got=[d['n'],d['schema'],d['content'],d['mean'],d['p50'],d['p95'],d['p99'],d['TO'],d['rs']]
        line=f"{sp:<7}{name:<8}"; flags=[]
        for i,c in enumerate(cols):
            g=got[i]
            if pap is None or pap[i] is None:
                line+=f"{(f'{g:.2f}' if isinstance(g,float) else g):>10}"; continue
            p=pap[i]; tol=0.5 if c=='n' else (0.5 if c=='TO' else (0.06 if c in('schema','content','rs') else 0.02+abs(p)*0.004))
            bad=abs(g-p)>tol
            if bad: flags.append(f"{c}: paper {p} vs data {g:.2f}"); nbad+=1
            line+=f"{(f'{g:.2f}' if isinstance(g,float) else g):>9}"+('*' if bad else ' ')
        print(line)
        for f in flags: print(f"{'':<15}  ! {f}")
print(f"\n  mismatches: {nbad}")
