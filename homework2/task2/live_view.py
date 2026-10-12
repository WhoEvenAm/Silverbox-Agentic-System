"""Read-only localhost view of actual agent events, for observing and recording."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Autonomous focusing experiment</title>
<style>body{margin:0;background:#eef3f8;color:#172a40;font:15px system-ui}header{padding:18px 26px;background:#17354e;color:white}
h1{font-size:24px;margin:0 0 5px}main{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px}
section{background:white;border-radius:12px;padding:18px;min-width:0}h2{margin:0 0 12px;font-size:18px}.muted{color:#607488;font-size:13px}
pre{white-space:pre-wrap;overflow:auto;max-height:210px;font:13px ui-monospace,monospace;background:#f2f6fa;padding:12px;border-radius:6px}
img{width:100%;max-height:380px;object-fit:contain}table{width:100%;font-size:13px;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #e1e9f0;text-align:left}
#status{font-size:17px;margin:12px 0}progress{width:100%}#log{max-height:165px;overflow:auto;font-size:13px}.event{padding:5px;border-bottom:1px solid #edf1f4}
</style></head><body><header><h1>Autonomous focusing experiment</h1><div>Four-role agent · Binary-amplitude DMD · Noisy simulated camera</div></header>
<main><section><h2>Agent decision and tool call</h2><div id="status">Waiting for the initial request</div><div id="decision"></div>
<pre id="settings">Experiment settings will be selected by the agent.</pre><progress id="progress" max="1" value="0"></progress>
<p class="muted" id="measurement">No measurement yet</p><h2>Actual event stream</h2><div id="log"></div></section>
<section><h2>Live camera measurement</h2><img id="camera" alt="Camera image appears when the executor runs the experiment">
<p class="muted">Fresh camera noise is included in every shot. Cyan marks the target region.</p><h2>Completed experiments</h2>
<table><thead><tr><th>Macro size</th><th>Iterations</th><th>Enhancement ± SEM</th></tr></thead><tbody id="results"></tbody></table></section></main>
<script>let count=-1;const $=id=>document.getElementById(id);
async function refresh(){try{const rows=await(await fetch('/events.json',{cache:'no-store'})).json();if(rows.length===count)return;count=rows.length;
const latest=rows.at(-1);if(!latest)return;const starts=rows.filter(r=>r.kind==='model_started');const choice=rows.filter(r=>r.kind==='decision').at(-1);
if(choice){$('decision').textContent='Decision '+choice.data.number+': '+choice.data.next_role+' — '+choice.data.reason}
const call=rows.filter(r=>r.kind==='tool_call').at(-1);if(call)$('settings').textContent=JSON.stringify(call.data.arguments,null,2);
const p=rows.filter(r=>r.kind==='experiment_progress').at(-1);if(p){$('progress').max=p.data.iterations;$('progress').value=p.data.iteration;
$('measurement').textContent='Iteration '+p.data.iteration+'/'+p.data.iterations+' · measured enhancement '+p.data.retained_measurement.toFixed(4);
$('camera').src='/artifact?path='+encodeURIComponent(p.data.camera_image)+'&v='+count}
const complete=rows.filter(r=>r.kind==='tool_completed');$('results').replaceChildren();for(const r of complete){const tr=document.createElement('tr');
for(const v of [r.data.settings.macropixel_size+' px',r.data.settings.iterations,r.data.final.enhancement.toFixed(4)+' ± '+r.data.final.enhancement_sem.toFixed(4)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('results').append(tr)}
// Keep the most recent actual camera event visible while the next model thinks.
const cameraEvent=rows.filter(r=>r.kind==='experiment_progress'||r.kind==='tool_completed').at(-1);
if(cameraEvent){const path=cameraEvent.kind==='tool_completed'?cameraEvent.data.camera_images.final:cameraEvent.data.camera_image;
$('camera').src='/artifact?path='+encodeURIComponent(path)+'&v='+count}
$('status').textContent=latest.kind==='run_ended'?'Run '+latest.data.status:latest.kind==='experiment_progress'?'Executor: focus_beam is measuring':'Latest event: '+latest.kind;
$('log').replaceChildren();for(const row of rows.slice(-6)){const div=document.createElement('div');div.className='event';
div.textContent=row.time.slice(11,19)+' '+row.kind+(row.data.role?' · '+row.data.role:'');$('log').append(div)}
}catch(e){$('status').textContent='Waiting for run events…'}}refresh();setInterval(refresh,500);
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    if not all(character.isalnum() or character in "_-" for character in args.run_name):
        parser.error("Use a simple run folder name")
    root = (Path(__file__).resolve().parent / "output/runs" / args.run_name).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            if request.path == "/":
                content, kind = PAGE.encode(), "text/html; charset=utf-8"
            elif request.path == "/events.json":
                path = root / "events.json"
                content, kind = (path.read_bytes() if path.exists() else b"[]"), "application/json"
            elif request.path == "/artifact":
                path = Path(parse_qs(request.query).get("path", [""])[0]).resolve()
                # Restrict image serving to this run; never expose arbitrary local files.
                if not path.is_relative_to(root) or path.suffix != ".png" or not path.is_file():
                    self.send_error(404)
                    return
                content, kind = path.read_bytes(), "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    print(f"Live experiment view: http://127.0.0.1:{args.port}/", flush=True)
    # Loopback only: this is an observer for a local experiment, not public hosting.
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
