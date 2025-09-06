from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
from uuid import uuid4
from typing import Dict

from app.pipeline import transcribe_video, burn_subtitles

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Subtitle Birth: VideoComm", docs_url=None, redoc_url=None)
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------- Job storage --------
JOBS: Dict[str, dict] = {}

# -------- Background task --------

def _background_burn(job_id: str, video_path: Path, srt_path: Path, target_lang: str):
    try:
        out_path = burn_subtitles(video_path, srt_path, STORAGE_DIR, target_lang)
        JOBS[job_id].update({"status": "done", "output": out_path.name})
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})

# -------- Routes --------

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile, lang: str = Form("en")):
    if not file.content_type or not file.content_type.startswith("video"):
        raise HTTPException(400, "Please upload a video file")

    job_id = str(uuid4())
    input_path = STORAGE_DIR / f"{job_id}_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Immediate transcription (synchronous) for instant user feedback
    try:
        segments, srt_path = transcribe_video(input_path, STORAGE_DIR, target_lang=lang)
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(500, f"Transcription failed: {e}")

    transcript_text = "\n".join(seg["text"] for seg in segments)

    JOBS[job_id] = {"status": "burning", "transcript": transcript_text, "output": None}

    # Burn subtitles in background
    background_tasks.add_task(_background_burn, job_id, input_path, srt_path, lang)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/download/{file_name}")
async def download(file_name: str):
    file_path = STORAGE_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path=file_path, filename=file_name, media_type="video/mp4")


# -------- Frontend --------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Subtitle Birth: VideoComm</title>
        <link href='https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap' rel='stylesheet'>
        <style>
            :root{--brand:#e6007e;--bg:#ffeaf4;--card:#fff}
            body{
                font-family:'Poppins',sans-serif;
                background:var(--bg);
                color:#333;
                margin:0;
                display:flex;
                flex-direction:column;
                align-items:center;
                gap:1.5rem;
                padding:2rem;
                min-height:100vh;
            }
            h1{
                color:var(--brand);
                margin:0;
                font-size:2.5rem;
                font-family:'M PLUS Rounded 1c',sans-serif;
                letter-spacing:1.5px;
            }
            #drop-area{
                border:2px dashed var(--brand);
                border-radius:16px;
                padding:3rem;
                width:90%;
                max-width:600px;
                text-align:center;
                cursor:pointer;
                transition:background .2s;
                background:rgba(255,255,255,0.6);
            }
            #drop-area.hover{background:#fff3f9}
            #drop-area svg{width:4rem;height:auto;fill:var(--brand)}
            
            .btn{
                background:var(--brand);
                color:#fff;
                border:none;
                padding:.6rem 1.2rem;
                border-radius:8px;
                font-size:1rem;
                cursor:pointer;
                transition:transform .2s;
            }
            .btn:hover{transform:translateY(-2px)}
            
            #langSelect{
                padding:.5rem;
                border-radius:8px;
                border:1px solid #ccc;
                font-size:1rem;
            }
            
            #status{text-align:center;color:#555;min-height:1.4em}
            
            #transcriptWrap{
                width:90%;
                max-width:800px;
                background:var(--card);
                border-radius:12px;
                border:1px solid #ffd0e6;
                padding:1rem;
                display:none;
                max-height:30vh;
                overflow:hidden;
            }
            #transcriptWrap.expanded{max-height:70vh}
            #transcript{white-space:pre-wrap;margin:0;line-height:1.5}
            
            #seeMore{
                margin-top:1rem;
                padding:.5rem 1rem;
                background:var(--brand);
                color:#fff;
                border:none;
                border-radius:20px;
                cursor:pointer;
            }
            
            #tvFrame{position:relative;display:none}
            #tvFrame img{width:90%;max-width:800px;height:auto;display:block}
            #resultVideo{
                position:absolute;
                top:13.5%;
                left:12.5%;
                width:75%;
                height:61%;
                border:none;
                border-radius:4px;
                object-fit:cover;
                z-index:1;
                background:#000;
            }
            
            #downloadBtn{
                margin-top:1rem;
                padding:.6rem 1.4rem;
                background:var(--brand);
                color:#fff;
                border:none;
                border-radius:20px;
                font-weight:600;
                cursor:pointer;
                display:none;
            }
            
            #procFooter{
                position:fixed;
                bottom:20px;
                left:50%;
                transform:translateX(-50%);
                width:300px;
                height:8px;
                background:#ffd0e6;
                border-radius:4px;
                overflow:hidden;
                display:none;
            }
            #procBar{
                width:40%;
                height:100%;
                background:var(--brand);
                animation:moveBar 1s linear infinite;
            }
            @keyframes moveBar{
                from{transform:translateX(-100%)}
                to{transform:translateX(100%)}
            }
        </style>
    </head>
    <body>
        <h1>Subtitle Birth: VideoComm</h1>

        <div id='drop-area'>
            <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>
                <path d='M12 16l4-5h-3V4h-2v7H8l4 5z'/>
                <path d='M20 18H4v2h16v-2z'/>
            </svg>
            <p>Drop a video here or click to select</p>
        </div>

        <select id='langSelect'>
            <option value='en' selected>English</option>
            <option value='ur'>Urdu</option>
            <option value='hi'>Hindi</option>
            <option value='ar'>Arabic</option>
            <option value='fr'>French</option>
        </select>

        <button id='uploadBtn' class='btn'>Upload Video</button>
        <input type='file' id='fileElem' accept='video/*' style='display:none'>

        <div id='status'></div>

        <div id='transcriptWrap'>
            <pre id='transcript'></pre>
            <button id='seeMore'>See more</button>
        </div>

        <div id='tvFrame'>
            <img src='/static/tv.png' alt='TV frame'>
            <video id='resultVideo' controls></video>
        </div>
        <button id='downloadBtn'>Download video</button>

        <div id='procFooter'><div id='procBar'></div></div>

<script>
const dropArea=document.getElementById('drop-area');
const langSelect=document.getElementById('langSelect');
const uploadBtn=document.getElementById('uploadBtn');
const fileInput=document.getElementById('fileElem');
const statusEl=document.getElementById('status');
const transcriptEl=document.getElementById('transcript');
const transcriptWrap=document.getElementById('transcriptWrap');
const seeMoreBtn=document.getElementById('seeMore');
const videoEl=document.getElementById('resultVideo');
const tvFrame=document.getElementById('tvFrame');
const dlBtn=document.getElementById('downloadBtn');
const procFooter=document.getElementById('procFooter');
let pollInterval=null, expanded=false;

function setStatus(t){statusEl.textContent=t;}

function handleFile(file){
    setStatus('Uploading…');
    transcriptWrap.style.display='none';
    videoEl.style.display='none';
    const fd=new FormData(); 
    fd.append('file',file);
    fd.append('lang', langSelect.value);
    fetch('/upload',{method:'POST',body:fd})
      .then(r=>r.json())
      .then(({job_id})=>{
        setStatus('Transcribing…');
        pollInterval=setInterval(()=>checkStatus(job_id),1200);
      }).catch(e=>setStatus('Error: '+e.message));
}

function applyCollapsed(){
    if(!expanded){
        transcriptWrap.classList.remove('expanded');
        seeMoreBtn.textContent='See more';
    }else{
        transcriptWrap.classList.add('expanded');
        seeMoreBtn.textContent='See less';
    }
}

seeMoreBtn.addEventListener('click',()=>{
    expanded=!expanded;
    applyCollapsed();
});

function checkStatus(job_id){
    fetch('/status/'+job_id)
      .then(r=>r.json())
      .then(d=>{
        if(d.status==='burning'){
            transcriptEl.textContent=d.transcript||'';
            procFooter.style.display='block';
            transcriptWrap.style.display='block';
            expanded=false; applyCollapsed();
            setStatus('Adding subtitles to video…');
        }else if(d.status==='done'){
            clearInterval(pollInterval);
            setStatus('Complete!');
            transcriptWrap.style.display='none';
            procFooter.style.display='none';
            videoEl.src='/storage/'+d.output;
            tvFrame.style.display='block';
            dlBtn.style.display='inline-block';
            dlBtn.onclick=()=>window.open('/storage/'+d.output,'_blank');
        }else if(d.status==='error'){
            clearInterval(pollInterval);
            setStatus('Error: '+d.error);
        }
      }).catch(()=>{});
}

/* Drag & click */
['dragenter','dragover'].forEach(evt=>dropArea.addEventListener(evt,e=>{e.preventDefault();dropArea.classList.add('hover');}));
['dragleave','drop'].forEach(evt=>dropArea.addEventListener(evt,e=>{e.preventDefault();dropArea.classList.remove('hover');}));
dropArea.addEventListener('drop',e=>{const f=e.dataTransfer.files?.[0]; if(f) handleFile(f);});
dropArea.addEventListener('click',e=>{if(e.target!==langSelect && e.target!==uploadBtn) fileInput.click();});
uploadBtn.addEventListener('click',(e)=>{
    e.preventDefault();
    e.stopPropagation();
    console.log('Upload button clicked');
    fileInput.click();
});
fileInput.addEventListener('change',e=>{const f=e.target.files?.[0]; if(f) handleFile(f);});
</script>
    </body>
    </html>
    """