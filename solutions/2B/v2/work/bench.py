import sys, time, torch, numpy as np
sys.argv=['x']; import img_cv as I
X=torch.randint(0,255,(64,3,96,96),dtype=torch.uint8)
y=torch.randint(0,5,(64,)).to(I.DEV)
rng=np.random.default_rng(0)
cfg=dict(bj=.2,cj=.2,shift=0,blur_p=.3,blur_lo=.8,blur_hi=1.8)
for arch,res,bs in [('tiny24',96,64),('tiny48',96,64),('r18',96,64),('r18',160,32),('r18nm',96,32),('r34',96,64),('effb0',128,32),('mnv3',160,32),('rgy8',128,32),('cnxt',96,32),('dn121',96,32),('r50',96,32)]:
  try:
    torch.cuda.reset_peak_memory_stats(); m=I.make_model(arch).to(I.DEV); p=I.Prep(res); opt=torch.optim.AdamW(m.parameters(),1e-3)
    for i in range(12):
      if i==2: torch.cuda.synchronize(); t=time.time()
      x=p.norm(I.augment(p(X[:bs]),rng,cfg)); l=torch.nn.functional.cross_entropy(m(x),y[:bs]); opt.zero_grad(); l.backward(); opt.step()
    torch.cuda.synchronize(); dt=(time.time()-t)/10
    print(arch,res,bs,'%.1f ms/step -> %.0f s/epoch(3200)'%(dt*1000,dt*3200/bs),'mem %.0fMB'%(torch.cuda.max_memory_allocated()/2**20),flush=True)
    del m,opt; torch.cuda.empty_cache()
  except Exception as e: print(arch,res,'ERR',str(e)[:100]); torch.cuda.empty_cache()
