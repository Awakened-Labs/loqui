//! A lazily loaded model with an idle timeout.
//!
//! The first request loads the model while holding the slot's lock, so
//! concurrent first requests wait for one load instead of starting several.
//! A failed load is remembered for a few seconds so a burst of requests
//! does not become a burst of multi-second load attempts. A model idle past
//! its TTL is dropped, freeing its memory (and VRAM), but never while a
//! request still holds it.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::Error;

/// How long a load failure is replayed before the next attempt.
const FAILURE_BACKOFF: Duration = Duration::from_secs(5);

type Loader<T> = Box<dyn Fn() -> Result<T, Error> + Send + Sync>;

pub(crate) struct Slot<T> {
    state: Mutex<State<T>>,
    ttl: Option<Duration>,
    load: Loader<T>,
}

struct State<T> {
    model: Option<Arc<T>>,
    last_used: Instant,
    failed: Option<(Instant, String)>,
}

impl<T> Slot<T> {
    pub fn new(ttl: Option<Duration>, load: Loader<T>) -> Self {
        Self { state: Mutex::new(State { model: None, last_used: Instant::now(), failed: None }), ttl, load }
    }

    fn lock(&self) -> MutexGuard<'_, State<T>> {
        // A panic while loading leaves nothing half-built in the state (the
        // model is only stored on success), so a poisoned lock is safe to use.
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The loaded model, loading it first if needed.
    pub fn get(&self) -> Result<Arc<T>, Error> {
        let mut state = self.lock();
        state.last_used = Instant::now();
        if let Some(model) = &state.model {
            return Ok(Arc::clone(model));
        }
        if let Some((when, why)) = &state.failed
            && when.elapsed() < FAILURE_BACKOFF
        {
            return Err(Error::Unavailable(why.clone()));
        }
        match (self.load)() {
            Ok(model) => {
                let model = Arc::new(model);
                state.model = Some(Arc::clone(&model));
                state.failed = None;
                state.last_used = Instant::now();
                Ok(model)
            }
            Err(e) => {
                state.failed = Some((Instant::now(), e.to_string()));
                Err(e)
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.lock().model.is_some()
    }

    /// Drops the model if nothing is using it. Returns whether it was dropped.
    pub fn unload(&self) -> bool {
        let mut state = self.lock();
        match &state.model {
            Some(model) if Arc::strong_count(model) == 1 => {
                state.model = None;
                true
            }
            _ => false,
        }
    }

    /// Unloads the model if it has been idle longer than the TTL.
    pub fn reap(&self) -> bool {
        let Some(ttl) = self.ttl else { return false };
        let idle = self.lock().last_used.elapsed();
        idle > ttl && self.unload()
    }

    pub fn ttl(&self) -> Option<Duration> {
        self.ttl
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn counting(fail: bool) -> (Arc<AtomicUsize>, Slot<u32>) {
        let loads = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&loads);
        let slot = Slot::new(
            Some(Duration::from_millis(1)),
            Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                if fail { Err(Error::Unavailable("boom".into())) } else { Ok(7) }
            }),
        );
        (loads, slot)
    }

    #[test]
    fn loads_once_and_shares() {
        let (loads, slot) = counting(false);
        let a = slot.get().unwrap();
        let b = slot.get().unwrap();
        assert_eq!((*a, *b), (7, 7));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn concurrent_first_requests_load_once() {
        let (loads, slot) = counting(false);
        let slot = Arc::new(slot);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let s = Arc::clone(&slot);
                std::thread::spawn(move || *s.get().unwrap())
            })
            .collect();
        assert!(threads.into_iter().all(|t| t.join().unwrap() == 7));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failures_back_off() {
        let (loads, slot) = counting(true);
        assert!(slot.get().is_err());
        assert!(slot.get().is_err());
        assert_eq!(loads.load(Ordering::SeqCst), 1, "the second call replays the failure");
    }

    #[test]
    fn idle_models_are_reaped_but_not_while_in_use() {
        let (loads, slot) = counting(false);
        let held = slot.get().unwrap();
        std::thread::sleep(Duration::from_millis(5));
        assert!(!slot.reap(), "a model in use must stay loaded");
        drop(held);
        assert!(slot.reap());
        assert!(!slot.is_loaded());
        slot.get().unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }
}
