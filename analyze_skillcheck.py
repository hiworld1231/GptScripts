#!/usr/bin/env python3
import argparse,json,math,os,re,subprocess
import cv2,numpy as np

def newest():
    fs=[x for x in os.listdir('.') if re.fullmatch(r'session_\d+\.mkv',x)]
    if not fs: raise FileNotFoundError('Не найден session_*.mkv')
    return max(fs,key=lambda x:int(re.search(r'\d+',x).group()))

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

def circle_candidates(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(7,7),1.5)
    cs=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=35,param1=90,param2=25,minRadius=50,maxRadius=85)
    if cs is None: return []
    return [(float(x),float(y),float(r)) for x,y,r in cs[0]]

def ring_features(frame,cx,cy,r,r0=-5,r1=8,bins=144):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    s=hsv[:,:,1].astype(np.float32); v=hsv[:,:,2].astype(np.float32)
    h,w=v.shape; yy,xx=np.indices((h,w)); rr=np.sqrt((xx-cx)**2+(yy-cy)**2)
    keep=(rr>=r+r0)&(rr<=r+r1)
    if keep.sum()<100: return None
    ang=(np.arctan2(-(yy-cy),xx-cx)+2*np.pi)%(2*np.pi)
    bi=(ang[keep]*bins/(2*np.pi)).astype(np.int32)
    def hist(score):
        z=score[keep]; sm=np.bincount(bi,weights=z,minlength=bins); n=np.bincount(bi,minlength=bins)
        return sm/np.maximum(n,1)
    sv=s[keep]; vv=v[keep]
    qv=np.percentile(vv,[20,35,50,65,80]); qs=np.percentile(sv,[20,50,80])
    white=np.clip((v-(qv[3]+4))/max(1,255-(qv[3]+4)),0,1)*np.clip((qs[2]+30-s)/max(1,qs[2]+30),0,1)
    black=np.clip((qv[1]-v)/max(1,qv[1]-qv[0]+20),0,1)*np.clip((s+25)/100,0,1)
    red=np.clip((hsv[:,:,0].astype(np.float32)-170)/20,0,1)+np.clip((10-hsv[:,:,0].astype(np.float32))/10,0,1)
    return hist(white),hist(black),hist(np.clip(red,0,1))

def smooth(x,n=3):
    k=np.ones(2*n+1,np.float32)/(2*n+1)
    return np.convolve(np.r_[x[-n:],x,x[:n]],k,'same')[n:-n]

def peak(h):
    h=smooth(h); i=int(np.argmax(h)); m=float(h.mean()); sd=float(h.std())+1e-6
    return (i+.5)*2*np.pi/len(h),max(0,(float(h[i])-m)/sd),float(h[i])

def candidate_score(frame,c):
    cx,cy,r=c; f=ring_features(frame,cx,cy,r)
    if f is None:return -1,None
    wh,bh,rh=f; wa,ws,wp=peak(wh); ba,bs,bp=peak(bh); ra,rs,rp=peak(rh)
    score=max(ws,bs)+0.35*rs+0.15*max(wp,bp)
    return score,(wa,ws,wp,ba,bs,bp,ra,rs,rp)

def local_check(cap,fps,start,end,sample):
    cap.set(cv2.CAP_PROP_POS_MSEC,max(0,start*1000))
    rows=[]; step=max(1,round(fps/sample)); frame_i=int(max(0,round(start*fps))); end_i=int(end*fps)
    last=None
    while frame_i<=end_i:
        ok,frame=cap.read()
        if not ok: break
        if (frame_i-int(round(start*fps)))%step!=0:
            frame_i+=1; continue
        cs=circle_candidates(frame)
        ranked=[]
        for c in cs:
            sc,feat=candidate_score(frame,c)
            if sc<0: continue
            if last is not None:
                lx,ly,lr=last; x,y,r=c
                sc-=0.08*(abs(x-lx)+abs(y-ly))+0.04*abs(r-lr)
            ranked.append((sc,c,feat))
        if ranked:
            ranked.sort(key=lambda z:z[0],reverse=True); sc,c,feat=ranked[0]
            if last is None or math.hypot(c[0]-last[0],c[1]-last[1])<45:
                last=c
                rows.append({'t':frame_i/fps,'x':c[0],'y':c[1],'r':c[2],'score':sc,'white_angle':feat[0],'white_strength':feat[1],'white_peak':feat[2],'black_angle':feat[3],'black_strength':feat[4],'black_peak':feat[5],'red_angle':feat[6],'red_strength':feat[7],'red_peak':feat[8]})
        frame_i+=1
    return rows

def angular_fit(rows,key,space):
    if len(rows)<6:return None
    t=np.array([x['t'] for x in rows],float); a=np.unwrap(np.array([x[key] for x in rows],float))
    keep=t<=space+0.02
    t=t[keep]; a=a[keep]
    if len(t)<6:return None
    n=min(18,len(t)); t=t[-n:]; a=a[-n:]
    try:
        v,b=np.polyfit(t,a,1)
    except Exception:return None
    if abs(v)<0.1:return None
    x=a[-1]; best=None
    for k in range(-5,6):
        z=(x+2*np.pi*k-b)/v; d=z-space
        if best is None or abs(d)<abs(best[0]): best=(d,z,v)
    if best is None or abs(best[0])>1.0:return None
    return {'predicted':best[1],'error':abs(best[0]),'signed_error':best[0],'speed_deg_s':math.degrees(v)}

def summarize(rows,space):
    if not rows:return {'result':'NO_CIRCLE','frames':0}
    wf=angular_fit(rows,'white_angle',space); bf=angular_fit(rows,'black_angle',space)
    last=rows[-1]
    result='UNKNOWN'; chosen=None
    if wf and wf['error']<=0.35: result='WHITE'; chosen=wf
    elif bf and bf['error']<=0.35: result='BLACK'; chosen=bf
    elif wf and (not bf or wf['error']<=bf['error']): result='WHITE'; chosen=wf
    elif bf: result='BLACK'; chosen=bf
    return {'result':result,'frames':len(rows),'first_t':rows[0]['t'],'last_t':last['t'],'center_mean':[float(np.mean([x['x'] for x in rows])),float(np.mean([x['y'] for x in rows]))],'radius_mean':float(np.mean([x['r'] for x in rows])),'white':wf,'black':bf,'chosen':chosen,'last_white_angle_deg':math.degrees(last['white_angle']),'last_black_angle_deg':math.degrees(last['black_angle']),'last_red_angle_deg':math.degrees(last['red_angle'])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('file',nargs='?'); ap.add_argument('--sample-fps',type=int,default=30); a=ap.parse_args(); p=a.file or newest()
    info=probe(p); ev=events(p); downs=[x[0] for x in ev if x[1]=='LMB_DOWN']; off=downs[0] if downs else 0; spaces=[max(0,x[0]-off) for x in ev if x[1]=='SPACE_DOWN']
    print(f'Файл: {p}'); print(f"Видео: {info.get('width')}x{info.get('height')} fps={info.get('r_frame_rate')}"); print('SPACE:',len(spaces),'->',', '.join(f'{x:.3f}' for x in spaces)); print('Режим: каждый SPACE анализируется отдельно, позиция круга не фиксируется')
    cap=cv2.VideoCapture(p); fps=cap.get(cv2.CAP_PROP_FPS) or 60; checks=[]
    for i,s in enumerate(spaces,1):
        start=max(0,s-1.8); end=s+0.08; rows=local_check(cap,fps,start,end,a.sample_fps); z=summarize(rows,s); z['space']=s; z['index']=i; z['trajectory']=rows; checks.append(z)
        c=z.get('center_mean'); print(f"#{i} SPACE={s:.3f} frames={z['frames']} center=" + ('%.1f,%.1f'%tuple(c) if c else 'none') + f" -> {z['result']}")
        for name in ['white','black']:
            q=z.get(name)
            if q: print(f"   {name.upper()}: predicted={q['predicted']:.3f} error={q['error']*1000:.1f}ms speed={q['speed_deg_s']:.1f}deg/s")
    cap.release()
    valid=[x for x in checks if x['result']!='NO_CIRCLE']; whites=sum(x['result']=='WHITE' for x in checks); blacks=sum(x['result']=='BLACK' for x in checks); unknown=sum(x['result']=='UNKNOWN' for x in checks)
    report={'file':p,'space_times':spaces,'sample_fps':a.sample_fps,'rules':{'white':'TOP 1','black':'TOP 2','position':'random/per-check detection','color':'adaptive, not exact RGB'},'summary':{'checks':len(checks),'with_circle':len(valid),'white':whites,'black':blacks,'unknown':unknown},'checks':checks}
    json.dump(report,open('analysis_report.json','w',encoding='utf8'),ensure_ascii=False,indent=2)
    print('\n========== ИТОГ =========='); print(f'Checks: {len(checks)} | circle: {len(valid)} | WHITE: {whites} | BLACK: {blacks} | UNKNOWN: {unknown}'); print('Полные покадровые траектории сохранены в analysis_report.json')
if __name__=='__main__': main()
