"""Graph parsing + hand-made chemistry features (no chemistry libs)."""
import numpy as np, pandas as pd, hashlib, collections
ATOMS=['C','N','O','S','F','Cl','Br']; A2I={a:i for i,a in enumerate(ATOMS)}
BT={'1':0,'2':1,'3':2,'a':3}; BORD={'1':1.0,'2':2.0,'3':3.0,'a':1.5}
VAL={'C':4,'N':3,'O':2,'S':2,'F':1,'Cl':1,'Br':1}

def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little')

def parse(atoms,bonds):
    at=atoms.split(); n=len(at)
    E=[]
    for b in bonds.split(';'):
        i,j,t=b.split('-'); E.append((int(i),int(j),t))
    return at,E

def bridges(n,adj):
    # iterative Tarjan; returns set of frozenset edges that are bridges
    disc=[-1]*n; low=[0]*n; t=0; br=set()
    for s in range(n):
        if disc[s]!=-1: continue
        stack=[(s,-1,iter(adj[s]))]; disc[s]=low[s]=t; t+=1
        while stack:
            u,p,it=stack[-1]
            for v,_ in it:
                if v==p: continue
                if disc[v]==-1:
                    disc[v]=low[v]=t; t+=1; stack.append((v,u,iter(adj[v]))); break
                low[u]=min(low[u],disc[v])
            else:
                stack.pop()
                if p!=-1:
                    low[p]=min(low[p],low[u])
                    if low[u]>disc[p]: br.add(frozenset((u,p)))
    return br

def smallest_ring(n,adj,ringbond):
    # smallest ring size per atom (0 if none) via BFS excluding edge
    rs=[0]*n
    for (u,v) in ringbond:
        # BFS u->v w/o edge u-v
        dist={u:0}; q=[u]; found=None
        while q and found is None:
            nq=[]
            for x in q:
                for y,_ in adj[x]:
                    if (x==u and y==v) or (x==v and y==u): continue
                    if y not in dist:
                        dist[y]=dist[x]+1
                        if y==v: found=dist[y]+1; break
                        nq.append(y)
                if found: break
            q=nq
        for a in (u,v):
            if found and (rs[a]==0 or found<rs[a]): rs[a]=found
    return rs

def mol_info(atoms,bonds):
    at,E=parse(atoms,bonds); n=len(at)
    adj=[[] for _ in range(n)]
    for i,j,t in E: adj[i].append((j,t)); adj[j].append((i,t))
    br=bridges(n,adj)
    ringb=[(i,j) for i,j,t in E if frozenset((i,j)) not in br]
    inring=[0]*n
    for i,j in ringb: inring[i]=inring[j]=1
    rs=smallest_ring(n,adj,ringb)
    deg=[len(adj[i]) for i in range(n)]
    nh=[]; arom=[]
    for i in range(n):
        s=sum(BORD[t] for _,t in adj[i]); arom.append(int(any(t=='a' for _,t in adj[i])))
        nh.append(int(max(0,VAL[at[i]]-int(np.floor(s+1e-6))) if s<=VAL[at[i]] else 0))
    return dict(at=at,E=E,adj=adj,br=br,inring=inring,rs=rs,deg=deg,nh=nh,arom=arom,n=n)

def atom_feats(m):
    """per-atom categorical features -> one-hot vector"""
    n=m['n']; F=[]
    for i in range(n):
        v=np.zeros(7+6+5+2+2+5,np.float32)
        v[A2I[m['at'][i]]]=1; v[7+min(m['deg'][i],5)]=1; v[13+min(m['nh'][i],4)]=1
        v[18+m['arom'][i]]=1; v[20+m['inring'][i]]=1
        r=m['rs'][i]; v[22+{0:0,3:1,4:1,5:2,6:3}.get(r,4)]=1
        F.append(v)
    return np.stack(F)

def ecfp(m,radius=3):
    """Morgan/ECFP-like WL hashing with bond types. returns list of (radius,key)"""
    n=m['n']
    h=[h64(f"{m['at'][i]}|{m['deg'][i]}|{m['nh'][i]}|{m['arom'][i]}|{m['inring'][i]}|{m['rs'][i]}") for i in range(n)]
    keys=[(0,x) for x in h]
    for r in range(1,radius+1):
        nh_=[]
        for i in range(n):
            nb=sorted((t,h[j]) for j,t in m['adj'][i])
            nh_.append(h64(f"{r}|{h[i]}|"+",".join(f"{t}{x}" for t,x in nb)))
        h=nh_; keys+= [(r,x) for x in h]
    return keys

def fragments(m):
    """cut bridge bonds (acyclic) -> ring systems & chains; hash each fragment via WL on its subgraph"""
    n=m['n']; br=m['br']
    # components after removing bridges = ring systems (size>1 w/ rings) or single acyclic atoms
    comp=[-1]*n; c=0
    for s in range(n):
        if comp[s]!=-1: continue
        comp[s]=c; st=[s]
        while st:
            u=st.pop()
            for v,_ in m['adj'][u]:
                if comp[v]==-1 and frozenset((u,v)) not in br: comp[v]=c; st.append(v)
        c+=1
    return comp,c

def ringsys_hashes(m):
    comp,c=fragments(m)
    groups=collections.defaultdict(list)
    for i,x in enumerate(comp): groups[x].append(i)
    out=[]
    for g,mem in groups.items():
        if len(mem)<3: continue
        S=set(mem)
        h={i:h64(m['at'][i]+str(m['arom'][i])) for i in mem}
        for r in range(len(mem)):
            h2={}
            for i in mem:
                nb=sorted((t,h[j]) for j,t in m['adj'][i] if j in S)
                ext=sum(1 for j,t in m['adj'][i] if j not in S)
                h2[i]=h64(f"{h[i]}|{ext}|{nb}")
            h=h2
            if r>=4: break
        out.append(h64(str(sorted(h.values()))))
    return out

def sub_hash(m,mem,mark=()):
    S=set(mem); mark=set(mark)
    h={i:h64(m['at'][i]+str(m['arom'][i])+('*' if i in mark else '')) for i in mem}
    for r in range(min(len(mem),8)):
        h2={}
        for i in mem:
            nb=sorted((t,h[j]) for j,t in m['adj'][i] if j in S)
            h2[i]=h64(f"{h[i]}|{nb}")
        h=h2
    return h64(str(sorted(h.values())))

def decompose(m):
    """scaffold = ring system with >=3 external bridges that is the tree centroid; blocks = subtrees."""
    comp,c=fragments(m); n=m['n']
    groups=collections.defaultdict(list)
    for i,x in enumerate(comp): groups[x].append(i)
    best=None
    for g,mem in groups.items():
        if len(mem)<3: continue
        S=set(mem); exts=[(i,j) for i in mem for j,t in m['adj'][i] if j not in S]
        if len(exts)<3: continue
        subs=[]
        for i,j in exts:
            seen={j}; st=[j]
            while st:
                u=st.pop()
                for v,_ in m['adj'][u]:
                    if v not in S and v not in seen: seen.add(v); st.append(v)
            subs.append((len(seen),i,j,seen))
        subs.sort(key=lambda x:-x[0])
        # sizes of 3 largest; small substituents (<=2 atoms) treated as part of scaffold
        big=[s for s in subs if s[0]>=3]
        score=(abs(len(big)-3), max(s[0] for s in subs))
        if best is None or score<best[0]: best=(score,mem,subs)
    if best is None: return None
    _,mem,subs=best
    big=subs[:3]; small=subs[3:]
    scaf=set(mem)
    for s in small: scaf|=s[3]
    attach=[s[1] for s in big]
    sh=sub_hash(m,sorted(scaf),attach)
    blocks=sorted(sub_hash(m,sorted(s[3]),[s[2]]) for s in big)
    return sh,blocks

def decompose_parts(m):
    """like decompose but returns per-atom part id (0=scaffold,1..3 blocks, ordered by block hash) + hashes"""
    comp,c=fragments(m)
    groups=collections.defaultdict(list)
    for i,x in enumerate(comp): groups[x].append(i)
    best=None
    for g,mem in groups.items():
        if len(mem)<3: continue
        S=set(mem); exts=[(i,j) for i in mem for j,t in m['adj'][i] if j not in S]
        if len(exts)<3: continue
        subs=[]
        for i,j in exts:
            seen={j}; st=[j]
            while st:
                u=st.pop()
                for v,_ in m['adj'][u]:
                    if v not in S and v not in seen: seen.add(v); st.append(v)
            subs.append((len(seen),i,j,seen))
        subs.sort(key=lambda x:-x[0])
        big=[s for s in subs if s[0]>=3]
        score=(abs(len(big)-3), max(s[0] for s in subs))
        if best is None or score<best[0]: best=(score,mem,subs)
    part=np.zeros(m['n'],np.int64)
    if best is None: return part,0,[0,0,0]
    _,mem,subs=best
    big=subs[:3]; scaf=set(mem)
    for s in subs[3:]: scaf|=s[3]
    sh=sub_hash(m,sorted(scaf),[s[1] for s in big])
    bl=sorted([(sub_hash(m,sorted(s[3]),[s[2]]),s) for s in big],key=lambda x:x[0])
    for k,(h,s) in enumerate(bl):
        for a in s[3]: part[a]=k+1
    return part,sh,[h for h,_ in bl]

def featurize_df(df,radius=3):
    out=[]
    for a,b in zip(df.atoms,df.bonds):
        m=mol_info(a,b)
        part,sh,bh=decompose_parts(m)
        X=atom_feats(m)
        E=np.array([(i,j,BT[t]) for i,j,t in m['E']],np.int64)
        keys=ecfp(m,radius)
        n=m['n']
        K=np.array([k for _,k in keys],np.uint64).reshape(radius+1,n)  # [r, atom]
        out.append(dict(x=X,e=E,part=part,scaf=sh,blocks=bh,keys=K))
    return out
