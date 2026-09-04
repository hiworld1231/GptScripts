#!/usr/bin/env python3
import argparse,json,math,os,re,subprocess
import cv2,numpy as np

def newest():
 f=[x for x in os.listdir('.') if re.fullmatch(r'session_\d+\.mkv',x)]
 if not f: raise FileNotFoundError('Не найден session_*.mkv')
 return max(f,key=lambda x:int(re.search(r'\d+',x).group()))

def events(p):
 q=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',p,'-map','0:s:0','-f','ass','-'],capture_output=True,text=True)
 out=[]
 for l in q.stdout.splitlines():
  if not l.startswith('Dialogue:'): continue
  z=l.split(',',9)
  if len(z)<10: continue
  m=re.match(r'(\d+):(\d+):(\d+)[.:](\d+)',z[1])
  if not m: continue
  h,mi,s,cs=map(int,m.groups()); t=h*3600+mi*60+s+cs/100
  k=re.search(r'(LMB_DOWN|LMB_UP|SPACE_DOWN)',z[9])
  if k: out.append((t,k.group(1)))
 return sorted(out)

def probe(p):
 q=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration','-of','json',p],capture_output=True,text=True)
 return json.loads(q.stdout)['streams'][0]

def geom(cap,fps,sample):
 rows=[]; i=0; step=max(1,round(fps/sample))
 while 1:
  ok,f=cap.read()
  if not ok: break
  if i%step==0:
   g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
   c=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,dp=1.2,minDist=35,param1=90,param2=28,minRadius=55,maxRadius=80)
   cs=[] if c is None else c[0]; cs=[x for x in cs if 140<=x[0]<=180 and 140<=x[1]<=185]
   x=min(cs,key=lambda z:abs(z[0]-160)+abs(z[1]-163)+.4*abs(z[2]-68)) if cs else (160,163,68)
   rows.append((i/fps,f,x[0],x[1],x[2]))
  i+=1
 return rows

def ring_maps(f,cx,cy,r0,r1,bins=144):
 hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV); s=hsv[:,:,1].astype(np.float32); v=hsv[:,:,2].astype(np.float32)
 h,w=v.shape; yy,xx=np.indices((h,w)); dx=xx-cx; dy=yy-cy; r=np.sqrt(dx*dx+dy*dy); keep=(r>=r0)&(r<=r1)
 a=(np.arctan2(-dy,dx)+2*np.pi)%(2*np.pi); ix=(a[keep]*bins/(2*np.pi)).astype(np.int32)
 def hist(x):
  y=x[keep]; sm=np.bincount(ix,weights=y,minlength=bins); n=np.bincount(ix,minlength=bins); return sm/np.maximum(n,1)
 sat=np.percentile(s[keep],[25,50,75]); val=np.percentile(v[keep],[25,50,75])
 white=np.clip((v-(val[1]+10))/max(1,255-(val[1]+10)),0,1)*np.clip((sat[2]+25-s)/max(1,sat[2]+25),0,1)
 black=np.clip(((val[1]-20)-v)/max(1,val[1]+20),0,1)*np.clip((s+35)/100,0,1)
 return hist(white),hist(black)

def smooth(x,n=3):
 k=np.ones(2*n+1,np.float32)/(2*n+1); return np.convolve(np.r_[x[-n:],x,x[:n]],k,'same')[n:-n]

def peak(h):
 h=smooth(h); i=int(np.argmax(h)); m=float(h.mean()); sd=float(h.std())+1e-6
 return (i+.5)*2*np.pi/len(h),max(0,(float(h[i])-m)/sd),float(h[i])

def build(rows,ring):
 out=[]
 for t,f,cx,cy,r in rows:
  wh,bh=ring_maps(f,cx,cy,r+ring[0],r+ring[1]); wa,ws,wp=peak(wh); ba,bs,bp=peak(bh)
  out.append((t,wa,ws,wp,ba,bs,bp))
 return out

def predict(a,t):
 a=np.asarray(a,float); t=np.asarray(t,float)
 if len(a)<6:return None
 a=np.unwrap(a); n=min(15,len(a)); t=t[-n:]; a=a[-n:]
 try:v,b=np.polyfit(t,a,1)
 except:return None
 if abs(v)<.15:return None
 x=a[-1]; cand=[]
 for k in range(-3,4): cand.append((abs((x+2*np.pi*k-b)/v-t[-1]),(x+2*np.pi*k-b)/v,v))
 d,z,v=min(cand,key=lambda x:x[0])
 if d>=.6:return None
 return z,v

def evaluate(feat,spaces,priority):
 t=np.array([x[0] for x in feat]); best=[]; total=0; err=0
 for s in spaces:
  mask=(t>=s-.5)&(t<=s+.05); ix=np.where(mask)[0]
  if len(ix)<6: best.append({'space':s,'result':'UNKNOWN','error':None,'predicted':None,'speed':None}); continue
  wa=np.array([feat[i][1] for i in ix]); ba=np.array([feat[i][4] for i in ix]); tt=t[ix]
  wp=predict(wa,tt); bp=predict(ba,tt)
  wc=None if wp is None else abs(wp[0]-s); bc=None if bp is None else abs(bp[0]-s)
  if priority=='WHITE' and wc is not None and wc<=.25:r='WHITE';e=wc;p=wp
  elif bc is not None and (wc is None or bc<=wc):r='BLACK';e=bc;p=bp
  elif wc is not None:r='WHITE';e=wc;p=wp
  else:r='UNKNOWN';e=None;p=None
  if e is not None: total+=1;err+=e
  best.append({'space':s,'result':r,'error':e,'predicted':None if p is None else p[0],'speed':None if p is None else p[1]})
 score=100*total/max(1,len(spaces))-45*min(1,err/max(1,total))
 return score,best

def main():
 ap=argparse.ArgumentParser();ap.add_argument('file',nargs='?');ap.add_argument('--sample-fps',type=int,default=30);a=ap.parse_args(); p=a.file or newest(); info=probe(p); ev=events(p); d=[x[0] for x in ev if x[1]=='LMB_DOWN']; off=d[0] if d else 0; spaces=[max(0,x[0]-off) for x in ev if x[1]=='SPACE_DOWN']
 print(f'Файл: {p}');print(f"Видео: {info.get('width')}x{info.get('height')} fps={info.get('r_frame_rate')}");print('SPACE:',len(spaces),'->',', '.join(f'{x:.3f}' for x in spaces));print('Извлекаю кадры и геометрию...')
 cap=cv2.VideoCapture(p);fps=cap.get(cv2.CAP_PROP_FPS) or 60; rows=geom(cap,fps,a.sample_fps);cap.release();print('Кадров для анализа:',len(rows));print('Адаптивный анализ оттенков: WHITE/BLACK (не точное RGB)')
 ranked=[]
 for ring in [(-6,5),(-2,7),(0,10),(3,14)]:
  f=build(rows,ring)
  for pr in ['WHITE','BLACK']:
   s,det=evaluate(f,spaces,pr);ranked.append((s,ring,pr,det))
 ranked.sort(key=lambda x:x[0],reverse=True);print('Тестов:',len(ranked));print('\n========== TOP 8 ==========')
 for i,(s,r,pr,d) in enumerate(ranked[:8],1):
  c={k:sum(x['result']==k for x in d) for k in ['WHITE','BLACK','UNKNOWN']};print(f'#{i} score={s:.2f} ring={r} priority={pr} WHITE={c["WHITE"]} BLACK={c["BLACK"]} UNKNOWN={c["UNKNOWN"]}')
 s,r,pr,det=ranked[0];print('\n========== BEST ==========');print(f'Score: {s:.2f}');print('Приоритет: WHITE TOP 1 -> BLACK TOP 2')
 for i,x in enumerate(det,1):
  if x.get('error') is None:print(f'#{i} SPACE {x["space"]:.3f} -> UNKNOWN')
  else:print(f'#{i} SPACE {x["space"]:.3f} -> {x["result"]} predicted={x["predicted"]:.3f} error={x["error"]*1000:.1f}ms speed={math.degrees(x["speed"]):.1f}deg/s')
 report={'file':p,'space_times':spaces,'tests':len(ranked),'best_score':s,'best_ring':r,'priority':pr,'best':det,'top':[{'score':x[0],'ring':x[1],'priority':x[2],'details':x[3]} for x in ranked]}
 json.dump(report,open('analysis_report.json','w',encoding='utf8'),ensure_ascii=False,indent=2);print('\nОтчёт сохранён: analysis_report.json')
if __name__=='__main__':main()
