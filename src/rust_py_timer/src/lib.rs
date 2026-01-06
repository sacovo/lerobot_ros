use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Threshold below which we spin-wait instead of sleeping for maximum precision
const SPIN_THRESHOLD: Duration = Duration::from_micros(5000);

/// Precise sleep that uses spin-waiting for the final microseconds
#[inline]
fn precise_sleep_until(target: Instant) {
    let now = Instant::now();
    if target <= now {
        return;
    }

    let remaining = target - now;

    // If we have more than SPIN_THRESHOLD remaining, sleep for the bulk of it
    if remaining > SPIN_THRESHOLD {
        thread::sleep(remaining - SPIN_THRESHOLD);
    }

    // Spin-wait for the remaining time for maximum precision
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

/// Convert an Instant to a Unix timestamp (f64 seconds since epoch).
/// This synchronizes Instant (monotonic) with SystemTime (wall clock) at startup.
struct TimeSync {
    instant_base: Instant,
    system_base: f64,
}

impl TimeSync {
    fn new() -> Self {
        let instant_base = Instant::now();
        let system_base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        Self {
            instant_base,
            system_base,
        }
    }

    /// Convert an Instant to a Unix timestamp based on the synchronized base.
    fn instant_to_timestamp(&self, instant: Instant) -> f64 {
        let elapsed = instant.duration_since(self.instant_base).as_secs_f64();
        self.system_base + elapsed
    }
}

/// Try to set the current thread to high priority (best-effort, Linux only)
#[cfg(target_os = "linux")]
fn try_set_high_priority() {
    

    // Try to set SCHED_FIFO with priority 1 (requires CAP_SYS_NICE or root)
    // If this fails, we just continue with normal priority
    unsafe {
        let thread_id = libc::pthread_self();
        let mut param: libc::sched_param = std::mem::zeroed();
        param.sched_priority = 1;
        let _ = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);
    }
}

#[cfg(not(target_os = "linux"))]
fn try_set_high_priority() {
    // No-op on other platforms
}

/// A frame containing messages collected during one interval.
/// Maps topic names to lists of messages (as Python objects).
type Frame = HashMap<String, Vec<PyObject>>;

/// Internal state shared between threads
struct CollectorState {
    /// Current frame being collected
    current_frame: Frame,
    /// Last message for each topic (for continuity between frames)
    last_messages: HashMap<String, PyObject>,
    /// Topic names that we're collecting
    topic_names: Vec<String>,
}

impl CollectorState {
    fn new(topic_names: Vec<String>) -> Self {
        let mut current_frame = HashMap::new();
        let last_messages = HashMap::new();
        for name in &topic_names {
            current_frame.insert(name.clone(), Vec::new());
        }
        Self {
            current_frame,
            last_messages,
            topic_names,
        }
    }

    /// Create a new empty frame
    fn new_frame(&self) -> Frame {
        let mut frame = HashMap::new();
        for name in &self.topic_names {
            frame.insert(name.clone(), Vec::new());
        }
        frame
    }

    /// Take a snapshot of the current frame and reset it.
    /// Returns the frame with all collected messages, plus last_messages for topics with no new data.
    fn snapshot(&mut self, py: Python<'_>) -> Frame {
        // Create new frame first to avoid borrow issues
        let new_frame = self.new_frame();
        // Take the current frame
        let mut frame = std::mem::replace(&mut self.current_frame, new_frame);

        // Update last_messages and fill in empty topics
        for (topic, messages) in &mut frame {
            if let Some(last) = messages.last() {
                // Update last message for this topic
                self.last_messages.insert(topic.clone(), last.clone_ref(py));
            } else if let Some(last_msg) = self.last_messages.get(topic) {
                // No new messages, use the last known message
                messages.push(last_msg.clone_ref(py));
            }
        }

        frame
    }
}

/// Signal sent from timer thread to trigger frame capture
struct CaptureSignal {
    timestamp: f64,
}

/// A precise frame collector that captures messages at fixed intervals.
///
/// The timing of frame capture is done in Rust for precision.
/// Messages are stored as Python objects and sent to Python for processing.
#[pyclass]
struct FrameCollector {
    state: Arc<RwLock<CollectorState>>,
    callback: Arc<Mutex<Option<PyObject>>>,
    running: Arc<Mutex<bool>>,
}

#[pymethods]
impl FrameCollector {
    /// Create a new FrameCollector.
    ///
    /// Args:
    ///     topic_names: List of topic names to collect messages for
    ///     fps: Frames per second (how often to capture frames)
    #[new]
    fn new(py: Python<'_>, topic_names: Vec<String>, fps: f64) -> PyResult<Self> {
        let state = Arc::new(RwLock::new(CollectorState::new(topic_names)));
        let callback: Arc<Mutex<Option<PyObject>>> = Arc::new(Mutex::new(None));
        let running = Arc::new(Mutex::new(true));

        // Channel for sending capture signals from timer thread
        let (tx, rx) = channel::<CaptureSignal>();

        let interval = Duration::from_secs_f64(1.0 / fps);

        // Timer thread - precise timing for frame capture signals
        let timer_running = Arc::clone(&running);
        let timer_tx = tx.clone();

        py.allow_threads(|| {
            thread::spawn(move || {
                // Try to set high priority for the timer thread
                try_set_high_priority();

                // Synchronize Instant (monotonic) with SystemTime (wall clock) once at start
                // This avoids calling SystemTime::now() on every tick which can drift
                let time_sync = TimeSync::new();

                let mut next_tick = Instant::now() + interval;

                loop {
                    // Check if we should stop (do this before sleeping, not after)
                    {
                        let running = timer_running.lock().unwrap();
                        if !*running {
                            break;
                        }
                    }

                    // Use precise sleep with spin-waiting for final approach
                    precise_sleep_until(next_tick);

                    let timestamp = time_sync.instant_to_timestamp(Instant::now());

                    // Send capture signal
                    let _ = timer_tx.send(CaptureSignal { timestamp });

                    // Schedule next tick
                    next_tick += interval;

                    // If we're falling behind, skip frames to catch up
                    // but don't adjust next_tick to current time (that would cause drift)
                    let now = Instant::now();
                    while next_tick < now {
                        // We missed this tick, skip it
                        next_tick += interval;
                    }
                }
            });
        });

        // Capture/callback thread - handles Python interaction
        let cap_state = Arc::clone(&state);
        let cap_callback = Arc::clone(&callback);
        let cap_running = Arc::clone(&running);

        py.allow_threads(|| {
            thread::spawn(move || {
                loop {
                    // Try to receive a capture signal (with timeout to check running flag)
                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(signal) => {
                            Python::with_gil(|py| {
                                // Take snapshot while holding GIL (needed for clone_ref)
                                let frame = {
                                    let mut state = cap_state.write().unwrap();
                                    state.snapshot(py)
                                };

                                // Get the callback
                                let cb = {
                                    let cb_lock = cap_callback.lock().unwrap();
                                    cb_lock.as_ref().map(|c| c.clone_ref(py))
                                };

                                if let Some(callback) = cb {
                                    // Convert Frame to Python dict
                                    let py_frame = frame.into_py_any(py).unwrap();

                                    if let Err(e) = callback.call1(py, (py_frame, signal.timestamp))
                                    {
                                        eprintln!("Error calling frame callback: {:?}", e);
                                    }
                                }
                            });
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            // Check if we should stop
                            let running = cap_running.lock().unwrap();
                            if !*running {
                                break;
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            break;
                        }
                    }
                }
            });
        });

        Ok(Self {
            state,
            callback,
            running,
        })
    }

    /// Add a message to the current frame for a specific topic.
    ///
    /// Args:
    ///     topic_name: The name of the topic
    ///     message: The ROS message (as a Python object)
    fn add_message(&self, topic_name: String, message: PyObject) -> PyResult<()> {
        let mut state = self.state.write().unwrap();
        if let Some(messages) = state.current_frame.get_mut(&topic_name) {
            messages.push(message);
        }
        Ok(())
    }

    /// Register a callback to be called with each frame.
    ///
    /// The callback should accept two arguments:
    ///     - frame: Dict[str, List[Any]] mapping topic names to lists of messages
    ///     - timestamp: float, the precise timestamp when the frame was captured
    fn register_callback(&self, callback: PyObject) -> PyResult<()> {
        let mut cb = self.callback.lock().unwrap();
        *cb = Some(callback);
        Ok(())
    }

    /// Stop the collector threads.
    fn stop(&self) -> PyResult<()> {
        let mut running = self.running.lock().unwrap();
        *running = false;
        Ok(())
    }

    /// Check if the collector is running.
    fn is_running(&self) -> bool {
        let running = self.running.lock().unwrap();
        *running
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_py_timer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FrameCollector>()?;
    Ok(())
}
