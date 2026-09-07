
const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];

const COLORS = {
  LOW:'#42ce8b',
  MODERATE:'#f3c849',
  HIGH:'#ff944d',
  CRITICAL:'#ff5f66'
};

const state = {
  currentView:'overview',
  summary:null,
  analytics:null,
  risk:[],
  alerts:[],
  reports:[],
  roads:[],
  system:null,
  maps:{overview:null,full:null},
  layers:{overview:null,full:null,reports:null,roads:null},
  charts:{}
};

function esc(value){
  return String(value ?? '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'
  }[c]));
}
function fmt(value,d=3){
  const n=Number(value);
  return Number.isFinite(n) ? n.toFixed(d) : '—';
}
function pct(value,d=0){
  const n=Number(value);
  return Number.isFinite(n) ? `${(n*100).toFixed(d)}%` : '—';
}
function riskBadge(level){
  const c=COLORS[level] || '#9aafc4';
  return `<span class="badge" style="background:${c}18;color:${c}">${esc(level)}</span>`;
}
function toast(title,message=''){
  const node=document.createElement('div');
  node.className='toast';
  node.innerHTML=`<strong>${esc(title)}</strong><small>${esc(message)}</small>`;
  $('#toastStack').appendChild(node);
  setTimeout(()=>node.remove(),4200);
}
async function api(url, options){
  const res=await fetch(url,options);
  if(!res.ok){
    let msg=await res.text();
    throw new Error(msg || `${res.status} ${res.statusText}`);
  }
  const ct=res.headers.get('content-type')||'';
  return ct.includes('application/json') ? res.json() : res.text();
}

function setView(name){
  state.currentView=name;
  $$('.view').forEach(v=>v.classList.toggle('active',v.id===`view-${name}`));
  $$('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.view===name));
  const titles={
    overview:'Command Overview',
    map:'Live Risk Map',
    alerts:'Prototype Alerts',
    reports:'Field Reporting',
    roads:'Road Connectivity',
    analytics:'Decision Analytics',
    system:'System Status'
  };
  $('#pageTitle').textContent=titles[name]||'NER Landslide Intelligence';
  $('#sidebar').classList.remove('open');
  window.scrollTo({top:0,behavior:'smooth'});
  if(name==='map' && state.maps.full) setTimeout(()=>state.maps.full.invalidateSize(),120);
  if(name==='overview' && state.maps.overview) setTimeout(()=>state.maps.overview.invalidateSize(),120);
  if(window.lucide) lucide.createIcons();
}
function initNavigation(){
  $$('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
  $$('[data-view-target]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.viewTarget)));
  $('#sidebarToggle').addEventListener('click',()=>$('#sidebar').classList.toggle('open'));
  $('#presentationBtn').addEventListener('click',()=>{
    document.body.classList.toggle('presentation');
    setTimeout(()=>{
      state.maps.overview?.invalidateSize();
      state.maps.full?.invalidateSize();
    },250);
  });
  $('#dismissBanner').addEventListener('click',()=>$('.global-banner').remove());
}

function tileLayer(){
  return L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
    maxZoom:18,
    attribution:'© OpenStreetMap contributors'
  });
}
function initMaps(){
  if(!window.L){
    $('#overviewMap').innerHTML='<div style="padding:30px;color:#7890a9">Interactive map library unavailable. Tables and APIs still work.</div>';
    $('#fullMap').innerHTML='<div style="padding:30px;color:#7890a9">Interactive map library unavailable. Tables and APIs still work.</div>';
    return;
  }
  state.maps.overview=L.map('overviewMap',{zoomControl:false,attributionControl:false}).setView([26.25,92.7],6);
  tileLayer().addTo(state.maps.overview);
  L.control.zoom({position:'bottomright'}).addTo(state.maps.overview);
  state.layers.overview=L.layerGroup().addTo(state.maps.overview);

  state.maps.full=L.map('fullMap',{zoomControl:false}).setView([26.25,92.7],6);
  tileLayer().addTo(state.maps.full);
  L.control.zoom({position:'bottomright'}).addTo(state.maps.full);
  state.layers.full=L.layerGroup().addTo(state.maps.full);
  state.layers.reports=L.layerGroup().addTo(state.maps.full);
  state.layers.roads=L.layerGroup().addTo(state.maps.full);
}

function markerStyle(r, compact=false){
  const p=Number(r.risk_probability)||0;
  const color=COLORS[r.risk_level]||'#9aafc4';
  return {
    radius:compact ? 3.4 + p*4.5 : 4 + p*6,
    color,
    fillColor:color,
    fillOpacity:.76,
    weight:r.prototype_alert ? 1.7 : .8,
    opacity:.95
  };
}
function renderRiskMarkers(rows){
  if(!window.L) return;
  state.layers.full.clearLayers();
  state.layers.overview.clearLayers();

  rows.forEach(r=>{
    const lat=Number(r.latitude),lon=Number(r.longitude);
    if(!Number.isFinite(lat)||!Number.isFinite(lon)) return;

    L.circleMarker([lat,lon],markerStyle(r,false))
      .on('click',()=>showDetail(r))
      .bindTooltip(`${esc(r.state)} • ${fmt(r.risk_probability)}`,{direction:'top'})
      .addTo(state.layers.full);

    L.circleMarker([lat,lon],markerStyle(r,true))
      .on('click',()=>{setView('map');setTimeout(()=>{state.maps.full.setView([lat,lon],10);showDetail(r)},160)})
      .addTo(state.layers.overview);
  });
}

function showDetail(r){
  $('#drawerEmpty').classList.add('hidden');
  const box=$('#drawerContent');
  box.classList.remove('hidden');
  box.innerHTML=`
    <div class="drawer-meta">${esc(r.grid_id)} • ${esc(r.state)}</div>
    <div class="drawer-title">${esc(r.state)} risk point</div>
    ${riskBadge(r.risk_level)}
    <div class="drawer-score">
      <span class="panel-kicker">AI RISK PROBABILITY</span>
      <strong>${fmt(r.risk_probability,3)}</strong>
      <div class="drawer-meta">${r.prototype_alert?'Prototype alert threshold exceeded':'Below prototype alert threshold'}</div>
    </div>
    <div class="feature-grid">
      ${feature('24h rainfall',`${fmt(r.rainfall_24h_mm,2)} mm`)}
      ${feature('72h rainfall',`${fmt(r.rainfall_72h_mm,2)} mm`)}
      ${feature('7d rainfall',`${fmt(r.rainfall_7d_mm,2)} mm`)}
      ${feature('Slope',`${fmt(r.slope_deg,1)}°`)}
      ${feature('Elevation',`${fmt(r.elevation_m,0)} m`)}
      ${feature('NDVI',fmt(r.ndvi_model_value,3))}
      ${feature('Land cover',r.land_cover || '—')}
      ${feature('Road distance',`${fmt(r.distance_to_road_m,0)} m`)}
      ${feature('River distance',`${fmt(r.distance_to_river_m,0)} m`)}
      ${feature('Soil moisture',fmt(r.soil_moisture_surface_m3_m3,3))}
    </div>
    <div class="drawer-disclaimer">Model-generated prototype risk assessment. Not an official government warning.</div>
  `;
  if(window.lucide) lucide.createIcons();
}
function feature(k,v){return `<div class="feature"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`}

function chartDefaults(){
  Chart.defaults.color='#829ab3';
  Chart.defaults.borderColor='rgba(138,164,196,.12)';
  Chart.defaults.font.family='Inter,system-ui,sans-serif';
}
function makeCharts(){
  if(!window.Chart || !state.analytics) return;
  chartDefaults();
  Object.values(state.charts).forEach(c=>c?.destroy?.());

  const counts=state.analytics.risk_counts;
  state.charts.donut=new Chart($('#riskDonut'),{
    type:'doughnut',
    data:{labels:['Low','Moderate','High','Critical'],datasets:[{data:[counts.LOW,counts.MODERATE,counts.HIGH,counts.CRITICAL],backgroundColor:[COLORS.LOW,COLORS.MODERATE,COLORS.HIGH,COLORS.CRITICAL],borderWidth:0,hoverOffset:5}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'72%',plugins:{legend:{position:'bottom',labels:{boxWidth:9,usePointStyle:true,padding:16}}}}
  });

  const ss=[...state.analytics.state_summary].sort((a,b)=>b.mean_risk-a.mean_risk);
  state.charts.stateBars=new Chart($('#stateBars'),{
    type:'bar',
    data:{labels:ss.map(x=>x.state),datasets:[{label:'Mean risk',data:ss.map(x=>x.mean_risk),backgroundColor:'rgba(77,163,255,.72)',borderRadius:7}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,max:1},x:{grid:{display:false},ticks:{maxRotation:45,minRotation:0}}}}
  });

  state.charts.stateRisk=new Chart($('#stateRiskChart'),{
    type:'bar',
    data:{labels:ss.map(x=>x.state),datasets:[
      {label:'Mean risk',data:ss.map(x=>x.mean_risk),backgroundColor:'rgba(77,163,255,.68)',borderRadius:6},
      {label:'Max risk',data:ss.map(x=>x.max_risk),backgroundColor:'rgba(255,95,102,.68)',borderRadius:6}
    ]},
    options:{responsive:true,maintainAspectRatio:false,scales:{y:{beginAtZero:true,max:1},x:{grid:{display:false}}}}
  });

  state.charts.alerts=new Chart($('#alertsChart'),{
    type:'bar',
    data:{labels:ss.map(x=>x.state),datasets:[{label:'Prototype alerts',data:ss.map(x=>x.prototype_alerts),backgroundColor:'rgba(159,122,234,.72)',borderRadius:6}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false}},y:{beginAtZero:true}}}
  });

  state.charts.rainRisk=new Chart($('#rainRiskChart'),{
    type:'scatter',
    data:{datasets:[{label:'State mean',data:ss.map(x=>({x:x.mean_rainfall_24h,y:x.mean_risk,label:x.state})),backgroundColor:'#46d7dc',pointRadius:6,pointHoverRadius:8}]},
    options:{responsive:true,maintainAspectRatio:false,scales:{x:{title:{display:true,text:'Mean 24h rainfall (mm)'}},y:{beginAtZero:true,max:1,title:{display:true,text:'Mean risk probability'}}},plugins:{tooltip:{callbacks:{label:(c)=>`${c.raw.label}: rain ${fmt(c.raw.x,1)} mm, risk ${fmt(c.raw.y,3)}`}}}}
  });
}

function renderWatchlist(rows){
  const top=[...rows].sort((a,b)=>b.risk_probability-a.risk_probability).slice(0,8);
  $('#watchlist').innerHTML=top.map((r,i)=>`
    <div class="watch-item">
      <div class="watch-rank">${String(i+1).padStart(2,'0')}</div>
      <div><div class="watch-title">${esc(r.state)} • ${esc(r.grid_id)}</div><div class="watch-sub">${riskBadge(r.risk_level)} ${r.prototype_alert?'• alert active':''}</div></div>
      <div class="risk-number" style="color:${COLORS[r.risk_level]}">${fmt(r.risk_probability)}</div>
    </div>`).join('');
}
function renderSummary(){
  const s=state.summary;
  $('#statMonitored').textContent=s.total_monitored_zones;
  $('#statCritical').textContent=s.critical_zones;
  $('#statHigh').textContent=s.high_risk_zones;
  $('#statAlerts').textContent=s.prototype_alerts;
  $('#navAlertCount').textContent=s.prototype_alerts;
  $('#heroMaxRisk').textContent=fmt(s.maximum_risk_probability,3);
  $('#heroTopState').textContent=`Highest mean risk: ${s.most_affected_state_by_mean_risk || 'NER'}`;
  $('#inferenceDate').textContent=s.inference_date || 'Cached snapshot';
}
function renderAlerts(filter=''){
  const q=($('#alertSearch')?.value||'').toLowerCase();
  const rows=state.alerts.filter(r=>(!filter||r.status===filter)&&(!q||String(r.grid_id).toLowerCase().includes(q)||String(r.state).toLowerCase().includes(q)));
  $('#alertCountLabel').textContent=`${rows.length} alerts`;
  $('#alertsTable').innerHTML=rows.map(r=>`
    <tr>
      <td><strong>${esc(r.grid_id)}</strong></td>
      <td>${esc(r.state)}</td>
      <td>${fmt(r.risk_probability)}</td>
      <td>${riskBadge(r.risk_level)}</td>
      <td><span class="status-badge status-${esc(r.status)}">${esc(r.status)}</span></td>
      <td>${r.status==='ACTIVE'?`<button class="mini-btn" onclick="ackAlert(${r.id})">Acknowledge</button>`:'—'}</td>
    </tr>`).join('');
}
function renderReports(){
  $('#reportCountLabel').textContent=`${state.reports.length} reports`;
  $('#reportsList').innerHTML=state.reports.length ? state.reports.map(r=>`
    <div class="activity-item">
      <div class="activity-icon"><span data-lucide="camera"></span></div>
      <div><div class="activity-title">${esc(titleCase(r.report_type))}</div><div class="activity-sub">${esc(r.state||'Unknown state')} • ${fmt(r.latitude,4)}, ${fmt(r.longitude,4)}</div></div>
      <div class="activity-actions"><span class="status-badge status-${esc(r.status)}">${esc(r.status)}</span>${r.status!=='RESOLVED'?`<button class="mini-btn" onclick="advanceReport(${r.id},'${r.status}')">Update</button>`:''}</div>
    </div>`).join('') : emptyState('No field reports yet','Submit the first geo-tagged field observation.');
  if(window.lucide) lucide.createIcons();
}
function renderRoads(){
  $('#roadsList').innerHTML=state.roads.length ? state.roads.map(r=>`
    <div class="activity-item">
      <div class="activity-icon"><span data-lucide="route"></span></div>
      <div><div class="activity-title">${esc(r.road_name)}</div><div class="activity-sub">${esc(r.state||'State not set')} ${r.notes?`• ${esc(r.notes)}`:''}</div></div>
      <div class="activity-actions"><span class="status-badge ${r.status==='BLOCKED'?'status-ACTIVE':'status-ACKNOWLEDGED'}">${esc(r.status)}</span></div>
    </div>`).join('') : emptyState('No road updates yet','Add current connectivity information for your demo.');
  if(window.lucide) lucide.createIcons();
}
function emptyState(title,msg){return `<div style="padding:40px 18px;text-align:center;color:#6f879f"><strong style="display:block;color:#afc2d5;margin-bottom:5px">${esc(title)}</strong><small>${esc(msg)}</small></div>`}
function titleCase(s){return String(s||'').replace(/\b\w/g,c=>c.toUpperCase())}
function renderSystem(){
  const labels={application:'Application',model:'ML model',risk_map:'Risk map',rainfall:'Rainfall',soil_moisture:'Soil moisture',satellite:'Satellite / NDVI',terrain:'Terrain',osm:'Roads / rivers',database:'Runtime database'};
  $('#systemGrid').innerHTML=Object.entries(labels).map(([k,label])=>{
    const v=state.system[k]||'UNKNOWN';
    return `<div class="system-card"><div class="system-top"><span class="system-name">${esc(label)}</span><strong class="system-status sys-${esc(v)}">${esc(v)}</strong></div><div class="system-note">${v==='CACHED'?'Processed Stage 10 snapshot used for reliable demo operation.':v==='LIVE'?'Service available to the application.':'Status not available.'}</div></div>`;
  }).join('');
}

async function loadMapRisk(){
  const params=new URLSearchParams();
  if($('#stateFilter').value) params.set('state',$('#stateFilter').value);
  if($('#levelFilter').value) params.set('level',$('#levelFilter').value);
  if($('#alertsOnly').checked) params.set('alerts_only','true');
  if($('#mapSearch').value.trim()) params.set('q',$('#mapSearch').value.trim());
  params.set('limit','5000');
  const rows=await api(`/api/risk?${params}`);
  state.risk=rows;
  renderRiskMarkers(rows);
  renderWatchlist(rows);
  toast('Map updated',`${rows.length} locations displayed`);
}
async function ackAlert(id){
  await api(`/api/alerts/${id}/acknowledge`,{method:'POST'});
  state.alerts=await api('/api/alerts');
  renderAlerts($('[data-alert-filter].active')?.dataset.alertFilter||'');
  toast('Alert acknowledged','Status updated for the presentation workflow.');
}
async function advanceReport(id,status){
  const next={NEW:'VERIFIED',VERIFIED:'IN_PROGRESS',IN_PROGRESS:'RESOLVED'}[status]||'RESOLVED';
  await api(`/api/reports/${id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:next})});
  state.reports=await api('/api/reports');
  renderReports();
  toast('Report updated',`Status changed to ${next}.`);
}
window.ackAlert=ackAlert;
window.advanceReport=advanceReport;

function initForms(){
  $('#reportForm').addEventListener('submit',async e=>{
    e.preventDefault();
    try{
      await api('/api/reports',{method:'POST',body:new FormData(e.currentTarget)});
      e.currentTarget.reset();
      state.reports=await api('/api/reports');
      renderReports();
      toast('Field report submitted','Saved as NEW / unverified.');
    }catch(err){toast('Could not submit report',err.message)}
  });

  $('#locateMe').addEventListener('click',()=>{
    if(!navigator.geolocation){toast('Location unavailable','Browser geolocation is not supported.');return}
    navigator.geolocation.getCurrentPosition(
      p=>{$('#reportLat').value=p.coords.latitude.toFixed(6);$('#reportLon').value=p.coords.longitude.toFixed(6);toast('Location captured','Coordinates added to the report form.');},
      ()=>toast('Location permission denied','Enter coordinates manually.')
    );
  });

  $('#roadForm').addEventListener('submit',async e=>{
    e.preventDefault();
    const payload={
      road_name:$('#roadName').value,
      state:$('#roadState').value,
      status:$('#roadStatus').value,
      latitude:$('#roadLat').value ? Number($('#roadLat').value) : null,
      longitude:$('#roadLon').value ? Number($('#roadLon').value) : null,
      notes:$('#roadNotes').value
    };
    try{
      await api('/api/roads',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      e.currentTarget.reset();
      state.roads=await api('/api/roads');
      renderRoads();
      toast('Road status saved',`${payload.road_name} added to the connectivity feed.`);
    }catch(err){toast('Could not save road status',err.message)}
  });
}

function initFilters(){
  $('#applyMapFilters').addEventListener('click',loadMapRisk);
  $('#mapSearch').addEventListener('keydown',e=>{if(e.key==='Enter')loadMapRisk()});
  $('#drawerClose').addEventListener('click',()=>{ $('#drawerContent').classList.add('hidden');$('#drawerEmpty').classList.remove('hidden')});
  let alertFilter='';
  $$('[data-alert-filter]').forEach(b=>b.addEventListener('click',()=>{
    $$('[data-alert-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');alertFilter=b.dataset.alertFilter;renderAlerts(alertFilter)
  }));
  $('#alertSearch').addEventListener('input',()=>renderAlerts(alertFilter));
}

async function boot(){
  try{
    initNavigation();
    initMaps();
    initForms();
    initFilters();

    const [summary,analytics,risk,alerts,reports,roads,system,states] = await Promise.all([
      api('/api/summary'),api('/api/analytics'),api('/api/risk?limit=5000'),api('/api/alerts'),api('/api/reports'),api('/api/roads'),api('/api/system/status'),api('/api/states')
    ]);

    Object.assign(state,{summary,analytics,risk,alerts,reports,roads,system});
    $('#stateFilter').innerHTML='<option value="">All states</option>'+states.map(s=>`<option>${esc(s)}</option>`).join('');

    renderSummary();
    renderRiskMarkers(risk);
    renderWatchlist(risk);
    renderAlerts();
    renderReports();
    renderRoads();
    renderSystem();
    makeCharts();

    if(window.lucide) lucide.createIcons();

    if('serviceWorker' in navigator){
      navigator.serviceWorker.register('/sw.js').catch(()=>{});
    }
  }catch(err){
    console.error(err);
    toast('Startup error',err.message);
  }
}
boot();
