#!/usr/bin/env python3
import argparse,json,math,os,re,subprocess
import cv2,numpy as np

REPORT_BASE='analysis_report'
REPORT_LIMIT=500

def newest():
    fs=[x for x in os.listdir('.') if re.fullmatch(r'session_\d+\.mkv',x)]
    if not fs: raise FileNotFoundError('Не найден session_*.mkv')
    return max(fs,key=lambda x:int(re.search(r'\d+',x).group()))

def save_report(report):
    raw=json.dumps(report,ensure_ascii=False,separators=(',',':'))
    parts=[]
    pos=0
    while pos<len(raw):
        lo,hi=1,min(REPORT_LIMIT-30,len(raw)-pos)
        while lo<hi:
            mid=(lo+hi+1)//2
            probe=json.dumps({'chunk':raw[pos:pos+mid]},ensure_ascii=False,separators=(',',':'))
            if len(probe)<=REPORT_LIMIT: lo=mid
            else: hi=mid-1
        parts.append(json.dumps({'chunk':raw[pos:pos+lo]},ensure_ascii=False,separators=(',',':')))
        pos+=lo
    existing=[]
    for name in os.listdir('.'):
        m=re.fullmatch(r'analysis_report(?:_(\d+))?\.json',name)
        if m: existing.append(0 if m.group(1) is None else int(m.group(1)))
    start=max(existing)+1 if existing else 0
    for i,part in enumerate(parts):
        suffix='' if start+i==0 else f'_{start+i}'
        with open(f'{REPORT_BASE}{suffix}.json','w',encoding='utf8') as f:f.write(part)
    print(f'Отчёт сохранён: {len(parts)} JSON-файл(ов), максимум {REPORT_LIMIT} символов')

def events(p):
    q=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',p,'-map','0:s:0','-f','ass','-'],capture_output=True,text=True)
    out=[]
    for l in q.stdout.splitlines():
        if not l.startswith('Dialogue:'):continue
        z=l.split(',',9)
        if len(z)<10:continue
        m=re.match(r'(\d+):(\d+):(\d+)[.:](\d+)',z[1])
        if not m:continue
        h,mi,s,cs=map(int,m.groups());t=h*3600+mi*60+s+cs/100
        k=re.search(r'(LMB_DOWN|LMB_UP|SPACE_DOWN)',z[9])
        if k:out.append((t,k.group(1)))
    return sorted(out)

def probe(p):
    q=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration','-of','json',p],capture_output=True,text=True)
    return json.loads(q.stdout)['streams'][0]

def hough(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    variants=[cv2.GaussianBlur(gray,(5,5),1.0),cv2.GaussianBlur(gray,(7,7),1.5),cv2.medianBlur(gray,5)]
    out=[]
    for g in variants:
        for dp,p1,p2 in ((1.0,80,20),(1.15,90,23),(1.2,100,25),(1.3,110,28),(1.4,120,31),(1.0,100,28)):
            cs=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,dp=dp,minDist=30,param1=p1,param2=p2,minRadius=40,maxRadius=100)
            if cs is not None:out.extend((float(x),float(y),float(r)) for x,y,r in cs[0])
    unique=[]
    for c in out:
        if not any(math.hypot(c[0]-q[0],c[1]-q[1])<7 and abs(c[2]-q[2])<7 for q in unique):unique.append(c)
    return unique

def ring_features(frame,cx,cy,r,bins=180):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h=hsv[:,:,0].astype(np.float32);s=hsv[:,:,1].astype(np.float32);v=hsv[:,:,2].astype(np.float32)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
    H,W=v.shape;yy,xx=np.indices((H,W));rr=np.hypot(xx-cx,yy-cy)
    keep=(rr>=r-7)&(rr<=r+10)
    if keep.sum()<150:return None
    ang=(np.arctan2(-(yy-cy),xx-cx)+2*np.pi)%(2*np.pi);bi=(ang[keep]*bins/(2*np.pi)).astype(np.int32)
    n=np.bincount(bi,minlength=bins)
    def hist(a):return np.bincount(bi,weights=a[keep],minlength=bins)/np.maximum(n,1)
    vv=v[keep];ss=s[keep];hh=h[keep]
    v20,v35,v50,v65,v80=np.percentile(vv,[20,35,50,65,80]);s20,s50,s80=np.percentile(ss,[20,50,80])
    white=np.clip((v-(v65+3))/max(1,255-v65),0,1)*np.clip((s80+25-s)/max(20,s80+25),0,1)
    black=np.clip((v35-v)/max(20,v35-v20+20),0,1)*(0.35+0.65*np.clip((s+15)/100,0,1))
    red=(np.minimum(np.abs(hh-0),np.abs(hh-180))<16).astype(np.float32)*np.clip((ss-s20)/max(20,100-s20),0,1)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3);edge=np.hypot(gx,gy)
    edgehist=hist(np.clip(edge/80,0,1))
    return hist(white),hist(black),hist(red),edgehist

def smooth(x,n=5):
    k=np.ones(2*n+1,np.float32)/(2*n+1);return np.convolve(np.r_[x[-n:],x,x[:n]],k,'same')[n:-n]

def peak(a):
    a=smooth(a);i=int(np.argmax(a));m=float(a.mean());sd=float(a.std())+1e-6
    return (i+.5)*2*np.pi/len(a),max(0,(float(a[i])-m)/sd),float(a[i])

def candidate_score(frame,c,last=None):
    f=ring_features(frame,*c)
    if f is None:return -1,None
    wh,bh,rh,eh=f;wa,ws,wp=peak(wh);ba,bs,bp=peak(bh);ra,rs,rp=peak(rh);ea,es,ep=peak(eh)
    x,y,r=c;score=max(ws,bs)+0.35*rs+0.25*es+0.15*max(wp,bp)
    if last is not None:
        lx,ly,lr=last;score-=0.045*math.hypot(x-lx,y-ly)+0.06*abs(r-lr)
    return score,(wa,ws,wp,ba,bs,bp,ra,rs,rp,ea,es,ep)

def detect_frame(frame,last=None):
    ranked=[]
    for c in hough(frame):
        score,feat=candidate_score(frame,c,last)
        if score>=0:ranked.append((score,c,feat))
    if not ranked:return None
    ranked.sort(key=lambda x:x[0],reverse=True)
    return ranked[0]

def local_check(cap,fps,start,end,sample):
    cap.set(cv2.CAP_PROP_POS_MSEC,max(0,start*1000));rows=[]
    first=int(round(start*fps));last_frame=int(round(end*fps));step=max(1,round(fps/sample));frame_i=first;previous=None;miss=0
    while frame_i<=last_frame:
        ok,frame=cap.read()
        if not ok:break
        if (frame_i-first)%step:
            frame_i+=1;continue
        hit=detect_frame(frame,previous)
        if hit is not None:
            score,c,feat=hit
            if previous is None or math.hypot(c[0]-previous[0],c[1]-previous[1])<70:
                previous=c;miss=0
                rows.append({'t':frame_i/fps,'x':c[0],'y':c[1],'r':c[2],'score':score,'white_angle':feat[0],'white_strength':feat[1],'white_peak':feat[2],'black_angle':feat[3],'black_strength':feat[4],'black_peak':feat[5],'red_angle':feat[6],'red_strength':feat[7],'red_peak':feat[8],'edge_angle':feat[9],'edge_strength':feat[10],'edge_peak':feat[11]})
            else:miss+=1
        else:miss+=1
        if miss>=max(4,round(sample*0.25)):previous=None
        frame_i+=1
    return rows

def angular_fit(rows,key,space):
    if len(rows)<6:return None
    t=np.array([x['t'] for x in rows]);a=np.unwrap(np.array([x[key] for x in rows]));keep=t<=space+0.03;t=t[keep];a=a[keep]
    if len(t)<6:return None
    n=min(30,len(t));t=t[-n:];a=a[-n:]
    best=None
    for degree in (1,2):
        if len(t)<degree+2:continue
        try:
            coef=np.polyfit(t-t[-1],a,degree)
            for dt in np.linspace(-0.5,0.25,301):
                z=np.polyval(coef,dt)+a[-1];d=space-(t[-1]+dt)
                target=a[-1]+(z-a[-1]);err=abs(np.arctan2(np.sin(target-a[-1]),np.cos(target-a[-1])))
                if best is None or err<best[0]:best=(err,t[-1]+dt,coef)
        except Exception:pass
    if best is None:return None
    _,pred,coef=best
    if abs(pred-space)>1.0:return None
    vel=np.polyval(np.polyder(coef),0)
    return {'predicted':float(pred),'error':float(abs(pred-space)),'signed_error':float(pred-space),'speed_deg_s':float(math.degrees(vel))}

def summarize(rows,space):
    if not rows:return {'result':'NO_CIRCLE','frames':0}
    wf=angular_fit(rows,'white_angle',space);bf=angular_fit(rows,'black_angle',space);last=rows[-1]
    if wf and wf['error']<=0.35:result='WHITE';chosen=wf
    elif bf and bf['error']<=0.35:result='BLACK';chosen=bf
    elif wf and (not bf or wf['error']<=bf['error']):result='WHITE';chosen=wf
    elif bf:result='BLACK';chosen=bf
    else:result='UNKNOWN';chosen=None
    return {'result':result,'frames':len(rows),'first_t':rows[0]['t'],'last_t':last['t'],'center_mean':[float(np.mean([x['x'] for x in rows])),float(np.mean([x['y'] for x in rows]))],'radius_mean':float(np.mean([x['r'] for x in rows])),'white':wf,'black':bf,'chosen':chosen,'last_white_angle_deg':math.degrees(last['white_angle']),'last_black_angle_deg':math.degrees(last['black_angle']),'last_red_angle_deg':math.degrees(last['red_angle'])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('file',nargs='?');ap.add_argument('--sample-fps',type=int,default=60);a=ap.parse_args();p=a.file or newest()
    info=probe(p);ev=events(p);downs=[x[0] for x in ev if x[1]=='LMB_DOWN'];off=downs[0] if downs else 0;spaces=[max(0,x[0]-off) for x in ev if x[1]=='SPACE_DOWN']
    print(f'Файл: {p}');print(f"Видео: {info.get('width')}x{info.get('height')} fps={info.get('r_frame_rate')}");print('SPACE:',len(spaces),'->',', '.join(f'{x:.3f}' for x in spaces));print('Режим: 60 FPS, несколько Hough-проходов, адаптивные цвета, случайная позиция круга, WHITE > BLACK')
    cap=cv2.VideoCapture(p);fps=cap.get(cv2.CAP_PROP_FPS) or 60;checks=[]
    for i,s in enumerate(spaces,1):
        start=max(0,s-2.5);end=s+0.12;rows=local_check(cap,fps,start,end,a.sample_fps);z=summarize(rows,s);z['space']=s;z['index']=i;z['trajectory']=rows;checks.append(z);c=z.get('center_mean');print(f"#{i} SPACE={s:.3f} frames={z['frames']} center="+('%.1f,%.1f'%tuple(c) if c else 'none')+f" -> {z['result']}")
        for name in ['white','black']:
            q=z.get(name)
            if q:print(f"   {name.upper()}: predicted={q['predicted']:.3f} error={q['error']*1000:.1f}ms speed={q['speed_deg_s']:.1f}deg/s")
    cap.release();valid=[x for x in checks if x['result']!='NO_CIRCLE'];whites=sum(x['result']=='WHITE' for x in checks);blacks=sum(x['result']=='BLACK' for x in checks);unknown=sum(x['result']=='UNKNOWN' for x in checks)
    report={'file':p,'space_times':spaces,'sample_fps':a.sample_fps,'rules':{'white':'TOP 1','black':'TOP 2','position':'random/per-check detection','color':'adaptive, not exact RGB'},'summary':{'checks':len(checks),'with_circle':len(valid),'white':whites,'black':blacks,'unknown':unknown},'checks':checks}
    save_report(report)
    print(f'\n========== ИТОГ ==========\nChecks: {len(checks)} | circle: {len(valid)} | WHITE: {whites} | BLACK: {blacks} | UNKNOWN: {unknown}')

if __name__=='__main__':main()
