const CACHE='ner-landslide-shell-v4';
const SHELL=['/','/static/styles.css','/static/app.js','/manifest.webmanifest'];

self.addEventListener('install',event=>{
  event.waitUntil(
    caches.open(CACHE).then(cache=>cache.addAll(SHELL)).catch(()=>{})
  );
  self.skipWaiting();
});

self.addEventListener('activate',event=>{
  event.waitUntil(
    caches.keys().then(keys=>
      Promise.all(keys.filter(key=>key!==CACHE).map(key=>caches.delete(key)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch',event=>{
  const req=event.request;
  if(req.method!=='GET') return;

  const url=new URL(req.url);

  // Same-origin app/API requests use network-first so a new deployment
  // is picked up immediately when online, with cache as offline fallback.
  if(url.origin===location.origin){
    event.respondWith(
      fetch(req).then(res=>{
        const copy=res.clone();
        caches.open(CACHE).then(cache=>cache.put(req,copy)).catch(()=>{});
        return res;
      }).catch(async()=>{
        const cached=await caches.match(req);
        if(cached) return cached;
        if(req.mode==='navigate') return caches.match('/');
        throw new Error('Offline and resource not cached');
      })
    );
    return;
  }

  // External map/CDN resources: cache-first for resilience.
  event.respondWith(
    caches.match(req).then(cached=>cached || fetch(req).then(res=>{
      const copy=res.clone();
      caches.open(CACHE).then(cache=>cache.put(req,copy)).catch(()=>{});
      return res;
    }))
  );
});
