// V7 stability mode.
// Offline service worker is intentionally disabled while the interactive map
// is being validated for the SIH presentation.
self.addEventListener('install',()=>self.skipWaiting());
self.addEventListener('activate',event=>{
  event.waitUntil(self.registration.unregister());
});
