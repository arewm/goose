//! Queued next-step messages for interactive sessions.
//!
//! While a turn runs, typed-ahead lines are captured by a lightweight
//! reader (see [`spawn_typeahead_reader`]) and queued. When the turn
//! completes, the drain loop auto-sends queued messages one per turn,
//! FIFO, behind a short hold window that can pause the queue. `/queue
//! edit` opens the queue as a rebase-style plan file in the user's
//! editor; mid-turn `/queue` requests are intercepted by the reader and
//! served at the turn boundary, before any queued message fires.
//!
//! An in-progress line the user was typing when the turn ended is
//! handed off as the next prompt's draft (persisted alongside the
//! queue) instead of being discarded.
//!
//! While the user types, animated progress output is muted (see
//! [`crate::session::output`]): the reader owns the terminal's input
//! row, and releases it after a short typing idle so the spinner can
//! resume from a clean line.
//!
//! Queue state is persisted as session metadata in a sidecar file so it
//! survives compaction (which only rewrites the conversation) and
//! process restarts. The sidecar is removed once the queue is fully
//! drained and unheld.

use crate::session::input::InputResult;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Hold-window duration before a queued message auto-sends.
const HOLD_WINDOW_MS: i32 = 3000;

/// Set while another component (elicitation, tool confirmation) owns
/// stdin; the type-ahead reader parks — restoring the terminal — until
/// it is cleared, so the two never fight over keystrokes.
static STDIN_YIELDED: AtomicBool = AtomicBool::new(false);

/// Set by the type-ahead reader once it has parked and fully restored
/// the terminal, so a yielding prompt knows stdin is exclusively its
/// own before touching terminal modes.
static READER_PARKED: AtomicBool = AtomicBool::new(false);

/// Set while a type-ahead reader thread is alive, so yielders can skip
/// the park wait entirely when there is no reader to park.
static READER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Guard that marks stdin as yielded to another prompt while alive.
pub struct StdinYield;

impl StdinYield {
    pub fn acquire() -> Self {
        STDIN_YIELDED.store(true, Ordering::Release);
        // Wait briefly for the reader to park and restore the terminal
        // before the prompt changes terminal modes itself. Skipped
        // entirely when no reader thread is alive.
        if READER_ACTIVE.load(Ordering::Acquire) {
            for _ in 0..250 {
                if READER_PARKED.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        StdinYield
    }
}

impl Drop for StdinYield {
    fn drop(&mut self) {
        STDIN_YIELDED.store(false, Ordering::Release);
    }
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct QueueState {
    messages: VecDeque<String>,
    held: bool,
    /// A `/queue edit` was requested mid-turn and must be served at the
    /// next turn boundary, before any queued message auto-sends.
    pending_edit: bool,
    /// An in-progress line the user was typing when the reader stopped at
    /// a turn boundary; seeded into the next prompt instead of discarded.
    #[serde(default)]
    pending_draft: Option<String>,
}

impl QueueState {
    fn is_idle(&self) -> bool {
        self.messages.is_empty() && !self.held && !self.pending_edit && self.pending_draft.is_none()
    }
}

#[derive(Clone)]
pub struct SharedQueue {
    state: Arc<Mutex<QueueState>>,
    path: Option<PathBuf>,
}

impl SharedQueue {
    /// Load the persisted queue for a session. Missing or corrupt files
    /// start an empty queue (a corrupt sidecar must never block a
    /// session).
    pub fn load(session_id: &str) -> Self {
        let path = queue_path(session_id);
        let state = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self {
            state: Arc::new(Mutex::new(state)),
            path: Some(path),
        }
    }

    pub fn detached() -> Self {
        Self {
            state: Arc::new(Mutex::new(QueueState::default())),
            path: None,
        }
    }

    fn with_state<R>(&self, f: impl FnOnce(&mut QueueState) -> R) -> R {
        let mut state = self.state.lock().expect("queue lock poisoned");
        let result = f(&mut state);
        self.persist(&state);
        result
    }

    fn persist(&self, state: &QueueState) {
        let Some(path) = &self.path else { return };
        if state.is_idle() {
            // A fully drained, unheld queue leaves no sidecar behind.
            let _ = std::fs::remove_file(path);
            return;
        }
        if let Ok(text) = serde_json::to_string(state) {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(path, text);
        }
    }

    pub fn enqueue(&self, message: String) -> usize {
        self.with_state(|state| {
            state.messages.push_back(message);
            state.messages.len()
        })
    }

    /// The message at 1-based `position`, without removing it.
    pub fn message_at(&self, position: usize) -> Option<String> {
        if position == 0 {
            return None;
        }
        self.state
            .lock()
            .expect("queue lock poisoned")
            .messages
            .get(position - 1)
            .cloned()
    }

    /// Replace the message at 1-based `position` (an edited browse).
    pub fn replace_at(&self, position: usize, message: String) {
        self.with_state(|state| {
            if position >= 1 && position <= state.messages.len() {
                state.messages[position - 1] = message;
            }
        });
    }

    /// Remove and return the message at 1-based `position`.
    pub fn remove_at(&self, position: usize) -> Option<String> {
        if position == 0 {
            return None;
        }
        self.with_state(|state| state.messages.remove(position - 1))
    }

    pub fn len(&self) -> usize {
        self.state
            .lock()
            .expect("queue lock poisoned")
            .messages
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_held(&self) -> bool {
        self.state.lock().expect("queue lock poisoned").held
    }

    /// Whether auto-send may proceed (a held queue stays parked).
    pub fn ready_to_send(&self) -> bool {
        let state = self.state.lock().expect("queue lock poisoned");
        !state.held && !state.messages.is_empty() && !state.pending_edit
    }

    pub fn take_pending_edit(&self) -> bool {
        self.with_state(|state| std::mem::take(&mut state.pending_edit))
    }

    /// Save an in-progress line so the next prompt resumes from it.
    pub fn set_pending_draft(&self, draft: String) {
        self.with_state(|state| state.pending_draft = Some(draft));
    }

    /// Remove and return the saved in-progress line for prompt seeding.
    pub fn take_pending_draft(&self) -> Option<String> {
        self.with_state(|state| state.pending_draft.take())
    }

    pub fn request_edit(&self) {
        self.with_state(|state| state.pending_edit = true);
    }

    pub fn hold(&self) {
        self.with_state(|state| state.held = true);
    }

    pub fn unhold(&self) {
        self.with_state(|state| state.held = false);
    }

    /// Pop the head message for sending.
    pub fn pop_head(&self) -> Option<String> {
        self.with_state(|state| state.messages.pop_front())
    }

    /// Replace the queue contents (after `/queue edit`). Applying an
    /// edit releases a held queue: the user just reviewed the plan.
    pub fn replace_all(&self, messages: Vec<String>) {
        self.with_state(|state| {
            state.messages = messages.into();
            state.pending_edit = false;
            state.held = false;
        });
    }

    pub fn head(&self) -> Option<String> {
        self.state
            .lock()
            .expect("queue lock poisoned")
            .messages
            .front()
            .cloned()
    }

    /// A copy of the current messages (for the plan editor).
    pub fn snapshot_messages(&self) -> Vec<String> {
        self.state
            .lock()
            .expect("queue lock poisoned")
            .messages
            .iter()
            .cloned()
            .collect()
    }
}

fn queue_path(session_id: &str) -> PathBuf {
    goose::config::paths::Paths::in_data_dir("queues").join(format!("{session_id}.json"))
}

/// Whether a typed line is a queue-management command that the reader
/// must intercept mid-turn instead of queueing as content.
pub fn is_queue_command(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed == "/queue" || trimmed.starts_with("/queue ")
}

/// What the reader should do with an intercepted `/queue` line.
pub enum QueueCommandOutcome {
    Notice(String),
    /// Start browsing the message at 1-based `position`
    /// (non-destructive).
    Browse {
        position: usize,
    },
}

/// The notice text of an outcome known to be a notice.
pub fn notice_text(outcome: &QueueCommandOutcome) -> String {
    match outcome {
        QueueCommandOutcome::Notice(notice) => notice.clone(),
        QueueCommandOutcome::Browse { .. } => String::new(),
    }
}

#[cfg(test)]
impl QueueCommandOutcome {
    fn as_notice(&self) -> &str {
        match self {
            QueueCommandOutcome::Notice(notice) => notice,
            QueueCommandOutcome::Browse { .. } => "",
        }
    }
}

/// Interpret an intercepted `/queue` line. `list` and errors are
/// answered immediately, `resume` releases a held queue, a number
/// recalls that message for editing, and anything else requests the
/// plan editor at the turn boundary.
pub fn interpret_queue_command(queue: &SharedQueue, line: &str) -> QueueCommandOutcome {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("/queue ") {
        let rest = rest.trim();
        if let Some(number) = rest
            .strip_prefix("drop ")
            .and_then(|n| n.trim().parse::<usize>().ok())
        {
            if number > 0 {
                return match queue.remove_at(number) {
                    Some(_) => QueueCommandOutcome::Notice(format!(
                        "dropped message {number} ({} remain)",
                        queue.len()
                    )),
                    None => QueueCommandOutcome::Notice(format!(
                        "no queued message {number} ({} pending)",
                        queue.len()
                    )),
                };
            }
        }
        if let Ok(number) = rest.parse::<usize>() {
            if number > 0 {
                return if queue.message_at(number).is_some() {
                    QueueCommandOutcome::Browse { position: number }
                } else {
                    QueueCommandOutcome::Notice(format!(
                        "no queued message {number} ({} pending)",
                        queue.len()
                    ))
                };
            }
        }
    }
    if trimmed == "/queue" || trimmed == "/queue list" {
        let state = queue.state.lock().expect("queue lock poisoned");
        if state.messages.is_empty() {
            return QueueCommandOutcome::Notice("queue is empty".to_string());
        }
        let mut out = String::from("queued messages:");
        for (idx, message) in state.messages.iter().enumerate() {
            let preview = preview_line(message, 72);
            out.push_str(&format!("\n  {:>2}. {}", idx + 1, preview));
        }
        out.push_str(
            "\n  (/queue <n> browses message n; up/down arrows walk the queue; /queue drop <n> removes one)",
        );
        if state.held {
            out.push_str("\n  (held — /queue resume to continue auto-sending)");
        }
        QueueCommandOutcome::Notice(out)
    } else if trimmed == "/queue resume" {
        queue.unhold();
        QueueCommandOutcome::Notice(
            "queue resumed — messages will auto-send after the current turn".to_string(),
        )
    } else {
        queue.request_edit();
        QueueCommandOutcome::Notice("queue editor will open when the turn completes".to_string())
    }
}

pub fn preview_line(message: &str, max_chars: usize) -> String {
    let first_line = message.lines().next().unwrap_or("");
    if first_line.chars().count() <= max_chars {
        first_line.to_string()
    } else {
        let mut preview: String = first_line.chars().take(max_chars).collect();
        preview.push('…');
        preview
    }
}

/// Render the hold-window line shown before a queued message
/// auto-sends.
pub fn hold_window_line(index: usize, total: usize, message: &str) -> String {
    format!(
        "sending queued message {}/{} — {:?}  (any key to hold)",
        index,
        total,
        preview_line(message, 60)
    )
}

// ── terminal plumbing ─────────────────────────────────────────────────

/// Terminal state guard: switches stdin to a byte-at-a-time mode (echo
/// and ISIG preserved) and restores the original state on drop. No file
/// status flags are changed, so nothing leaks to other users of stdin.
struct CbreakGuard {
    original: libc::termios,
    active: bool,
}

impl CbreakGuard {
    /// Reader mode: non-canonical, ISIG on, kernel echo off. The kernel
    /// cannot be trusted to echo usefully here: with ECHOCTL on, control
    /// bytes render as caret notation (`^?`); with it off, terminals
    /// render the raw DEL byte unpredictably. The reader echoes accepted
    /// bytes itself, except for type-ahead already echoed by the kernel
    /// before the reader armed (see `stdin_pending`).
    fn new() -> io::Result<Self> {
        Self::configure(true, true, false)
    }

    /// Hold-window mode: Ctrl+C reads as a plain byte (interrupting the
    /// process between turns would kill the session) and the hold key is
    /// not echoed.
    fn hold_mode() -> io::Result<Self> {
        Self::configure(false, true, false)
    }

    fn configure(isig: bool, no_echo: bool, no_echoctl: bool) -> io::Result<Self> {
        use std::io::IsTerminal;
        if !io::stdin().is_terminal() {
            return Ok(Self {
                original: unsafe { std::mem::zeroed() },
                active: false,
            });
        }
        let fd = io::stdin().as_raw_fd();
        // SAFETY: plain termios query on stdin.
        let mut original = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(fd, &mut original) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let mut raw = original;
        raw.c_lflag &= !libc::ICANON;
        if !isig {
            raw.c_lflag &= !libc::ISIG;
        }
        if no_echo {
            raw.c_lflag &= !libc::ECHO;
        }
        if no_echoctl {
            raw.c_lflag &= !libc::ECHOCTL;
        }
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            original,
            active: true,
        })
    }
}

impl Drop for CbreakGuard {
    fn drop(&mut self) {
        if self.active {
            let fd = io::stdin().as_raw_fd();
            // SAFETY: restoring saved terminal state.
            unsafe { libc::tcsetattr(fd, libc::TCSANOW, &self.original) };
        }
    }
}

/// Read one byte with an optional timeout. `Ok(None)` on timeout or
/// spurious readiness. Relies on `poll` readiness rather than
/// `O_NONBLOCK` so file status flags are never modified.
fn read_byte_timeout(timeout_ms: i32) -> io::Result<Option<u8>> {
    let fd = io::stdin().as_raw_fd();
    let mut descriptor = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: poll blocks only this thread for at most `timeout_ms`.
    let ready = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
    if ready < 0 {
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            return Ok(None);
        }
        return Err(error);
    }
    if ready == 0 {
        return Ok(None);
    }
    let mut byte = [0u8; 1];
    // Raw read on the fd itself: std's stdin is a BufReader that would
    // slurp a whole burst (an escape sequence arrives as one write)
    // into a buffer poll cannot see, splitting the sequence across
    // reads and stranding the tail. Poll reported readiness, so this
    // read returns promptly.
    // SAFETY: plain single-byte read on stdin.
    let count = unsafe { libc::read(fd, byte.as_mut_ptr() as *mut libc::c_void, 1) };
    if count < 0 {
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::WouldBlock || error.kind() == io::ErrorKind::Interrupted {
            return Ok(None);
        }
        return Err(error);
    }
    if count == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "stdin closed"));
    }
    Ok(Some(byte[0]))
}

/// Non-consuming check for input already waiting on stdin. Bytes
/// waiting when the reader arms were typed while the terminal was still
/// in cooked mode, so the kernel has already echoed them.
fn stdin_pending() -> bool {
    let fd = io::stdin().as_raw_fd();
    let mut descriptor = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: poll with zero timeout only queries readiness.
    (unsafe { libc::poll(&mut descriptor, 1, 0) }) > 0
}

/// The hold window: show the announcement, wait briefly, and report
/// whether the user asked to hold (any key) or the window expired
/// (auto-send).
pub fn hold_window(index: usize, total: usize, message: &str) -> bool {
    println!(
        "{}",
        console::style(hold_window_line(index, total, message)).dim()
    );
    let _ = io::stdout().flush();
    let Ok(_guard) = CbreakGuard::hold_mode() else {
        // No terminal to hold with; auto-send after a beat so the user
        // can still read the announcement.
        std::thread::sleep(Duration::from_millis(HOLD_WINDOW_MS as u64));
        return false;
    };
    if matches!(read_byte_timeout(HOLD_WINDOW_MS), Ok(Some(_))) {
        // Drain any remaining bytes of the key that held (escape
        // sequences, wide characters) so they cannot leak into the next
        // prompt as garbage.
        while matches!(read_byte_timeout(10), Ok(Some(_))) {}
        true
    } else {
        false
    }
}

/// Handle to a running type-ahead reader. Dropping it stops the reader
/// and restores the terminal.
pub struct TypeaheadReader {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for TypeaheadReader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Why [`reader_loop`] returned.
enum ReaderExit {
    Stop,
    /// stdin was yielded to another prompt; the caller must park.
    Yielded,
    /// stdin failed (closed terminal); back off before re-arming.
    StdinError,
}

/// Spawn the mid-turn type-ahead reader.
///
/// Runs on a plain thread so it never occupies a tokio worker. It owns
/// stdin in cbreak mode while the turn runs; lines entered are queued,
/// `/queue` commands are intercepted, and a lone ESC works through the
/// layered actions (clear line → pop last queued message → interrupt
/// the turn via `cancel_token`). While another component holds the
/// [`StdinYield`] guard (elicitation, tool confirmation), the reader
/// parks with the terminal restored.
pub fn spawn_typeahead_reader(
    queue: SharedQueue,
    cancel_token: CancellationToken,
) -> TypeaheadReader {
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let handle = std::thread::Builder::new()
        .name("typeahead-reader".into())
        .spawn(move || {
            READER_ACTIVE.store(true, Ordering::Release);
            loop {
                if thread_stop.load(Ordering::Acquire) {
                    READER_ACTIVE.store(false, Ordering::Release);
                    return;
                }
                let guard = match CbreakGuard::new() {
                    Ok(guard) => guard,
                    Err(_) => {
                        READER_ACTIVE.store(false, Ordering::Release);
                        return;
                    }
                };
                let exit = reader_loop(&thread_stop, &queue, &cancel_token, stdin_pending());
                // Restore the terminal BEFORE announcing the park, so a
                // yielding prompt never has its terminal modes clobbered
                // by this restore.
                drop(guard);
                match exit {
                    ReaderExit::Stop => {
                        READER_ACTIVE.store(false, Ordering::Release);
                        return;
                    }
                    ReaderExit::StdinError => {
                        // The terminal may be gone (disconnect); retry
                        // slowly rather than spinning.
                        std::thread::sleep(Duration::from_millis(500));
                    }
                    ReaderExit::Yielded => {
                        READER_PARKED.store(true, Ordering::Release);
                        while STDIN_YIELDED.load(Ordering::Acquire) {
                            if thread_stop.load(Ordering::Acquire) || cancel_token.is_cancelled() {
                                READER_PARKED.store(false, Ordering::Release);
                                READER_ACTIVE.store(false, Ordering::Release);
                                return;
                            }
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        READER_PARKED.store(false, Ordering::Release);
                    }
                }
            }
        })
        .ok();
    if handle.is_none() {
        eprintln!("warning: could not start the type-ahead reader");
    }
    TypeaheadReader { stop, handle }
}

fn dim_note(text: &str) {
    println!("{}", console::style(text).dim());
    let _ = io::stdout().flush();
}

/// Announce a browse and place the message text on the live line, as if
/// typed, so the cursor sits after it and edits are visible.
fn show_browsed_line(position: usize, total: usize, message: &str) {
    dim_note(&format!("(browsing message {position}/{total})"));
    let _ = io::stdout().write_all(message.as_bytes());
    let _ = io::stdout().flush();
}

/// The in-progress line being typed, as raw bytes so multi-byte UTF-8
/// sequences survive; decoded on Enter. `cursor` is a byte index into
/// `bytes`, always on a character boundary after a completed action.
#[derive(Default)]
struct LineBuffer {
    bytes: Vec<u8>,
    cursor: usize,
}

/// Display columns a byte slice occupies (one per character; wide
/// glyphs are the accepted wart).
fn char_columns(bytes: &[u8]) -> usize {
    bytes.iter().filter(|byte| *byte & 0xC0 != 0x80).count()
}

impl LineBuffer {
    /// Insert a byte at the cursor, advancing it.
    fn insert(&mut self, byte: u8) {
        self.bytes.insert(self.cursor, byte);
        self.cursor += 1;
    }

    fn push_str(&mut self, text: &str) {
        self.bytes.extend_from_slice(text.as_bytes());
        self.cursor = self.bytes.len();
    }

    /// The not-yet-echoed tail after the cursor.
    fn tail(&self) -> &[u8] {
        &self.bytes[self.cursor..]
    }

    /// Remove the character before the cursor.
    fn backspace(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        let mut start = self.cursor - 1;
        while start > 0 && self.bytes[start] & 0xC0 == 0x80 {
            start -= 1;
        }
        self.bytes.drain(start..self.cursor);
        self.cursor = start;
        true
    }

    /// Move left one character; returns whether the cursor moved.
    fn left(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        let mut start = self.cursor - 1;
        while start > 0 && self.bytes[start] & 0xC0 == 0x80 {
            start -= 1;
        }
        self.cursor = start;
        true
    }

    /// Move right one character; returns whether the cursor moved.
    fn right(&mut self) -> bool {
        if self.cursor >= self.bytes.len() {
            return false;
        }
        self.cursor += 1;
        while self.cursor < self.bytes.len() && self.bytes[self.cursor] & 0xC0 == 0x80 {
            self.cursor += 1;
        }
        true
    }

    /// Move to the start; returns the columns travelled.
    fn home(&mut self) -> usize {
        let moved = char_columns(&self.bytes[..self.cursor]);
        self.cursor = 0;
        moved
    }

    /// Move to the end; returns the columns travelled.
    fn end(&mut self) -> usize {
        let moved = char_columns(&self.bytes[self.cursor..]);
        self.cursor = self.bytes.len();
        moved
    }

    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The full line contents, for whole-line redraws.
    fn contents(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode the line; invalid UTF-8 (half-typed sequences) is an
    /// error rather than silently corrupted text.
    fn finish(&mut self) -> Result<String, std::string::FromUtf8Error> {
        let bytes = std::mem::take(&mut self.bytes);
        self.cursor = 0;
        String::from_utf8(bytes)
    }
}

/// Preserve work-in-progress when the reader stops mid-edit: a recalled
/// queued message must return to the queue (it was already popped and
/// persisted as such) rather than vanish; a freshly typed partial line
/// becomes the next prompt's draft when the reader is stopping for good,
/// and is dropped when it yields or fails (a yielding prompt leaves the
/// echoed text on screen, so a saved draft would be duplicated).
fn salvage_partial_line(
    queue: &SharedQueue,
    buffer: &mut LineBuffer,
    browsing: Option<usize>,
    persist_fresh: bool,
    line_visible: bool,
) {
    if let Some(position) = browsing {
        // Browsing is non-destructive, so the original is intact; save
        // any edits the user had made when the turn ended.
        if let Ok(line) = buffer.finish() {
            if !line.trim().is_empty() {
                queue.replace_at(position, line.trim_end().to_string());
                dim_note(&format!("(edits returned to message {position})"));
                return;
            }
        }
        return;
    }
    if buffer.is_empty() {
        return;
    }
    // A fresh partial line: when the reader is stopping for good, hand it
    // to the next prompt (clearing the echoed copy so it is not shown
    // twice — the prompt re-renders it as pre-typed text). A yielding or
    // failing reader re-arms and keeps its own screen state, so the line
    // is dropped as before.
    let line = buffer.finish().ok().filter(|line| !line.trim().is_empty());
    match (persist_fresh, line) {
        (true, Some(line)) => {
            queue.set_pending_draft(line.trim_end().to_string());
            if line_visible {
                clear_echoed_line();
            }
            dim_note("(draft saved — it resumes at the prompt)");
        }
        _ => dim_note("(type-ahead in progress was discarded)"),
    }
}

/// Erase the partial line the reader echoed on the current terminal row
/// (carriage return + clear-to-end-of-line) so the next prompt can
/// re-render the draft as its own pre-typed input without duplication.
fn clear_echoed_line() {
    print!("\r\x1b[K");
    let _ = io::stdout().flush();
}

/// Take ownership of the current terminal row as the reader's input
/// line: erase whatever frame or output occupies it, redraw the whole
/// in-progress line, and park the cursor back on its edit position.
/// While the line is owned, the progress ticker is muted (see
/// [`crate::session::output::note_reader_keystroke`]), so nothing else
/// redraws over it.
fn redraw_input_line(buffer: &LineBuffer) {
    print!("\r\x1b[K");
    let _ = io::stdout().write_all(buffer.contents());
    let tail_columns = char_columns(buffer.tail());
    for _ in 0..tail_columns {
        print!("\u{8}");
    }
    let _ = io::stdout().flush();
}

/// Hand the input line's row back to progress output: erase the echoed
/// line (the draft stays in the buffer and reappears on the next
/// keystroke) so a resuming spinner starts from a clean row.
fn release_input_line() {
    print!("\r\x1b[K");
    let _ = io::stdout().flush();
}

fn reader_loop(
    stop: &AtomicBool,
    queue: &SharedQueue,
    cancel_token: &CancellationToken,
    mut echo_suppressed: bool,
) -> ReaderExit {
    let mut buffer = LineBuffer::default();
    // 1-based queue position being browsed (non-destructive editing).
    let mut browsing: Option<usize> = None;
    // Whether the reader currently owns the terminal's input row (the
    // echoed line is on screen). Cleared after typing goes idle, when
    // the row is released back to progress output.
    let mut line_visible = false;
    loop {
        if stop.load(Ordering::Acquire) || cancel_token.is_cancelled() {
            salvage_partial_line(queue, &mut buffer, browsing, true, line_visible);
            return ReaderExit::Stop;
        }
        if STDIN_YIELDED.load(Ordering::Acquire) {
            salvage_partial_line(queue, &mut buffer, browsing, false, line_visible);
            return ReaderExit::Yielded;
        }
        let byte = match read_byte_timeout(50) {
            Ok(Some(byte)) => byte,
            Ok(None) => {
                // Typing has been idle long enough: release the input
                // row so the (about-to-resume) progress spinner starts
                // from a clean line. The draft reappears on the next
                // keystroke.
                if line_visible && super::output::reader_should_release_line() {
                    release_input_line();
                    line_visible = false;
                }
                continue;
            }
            Err(_) => {
                salvage_partial_line(queue, &mut buffer, browsing, false, line_visible);
                return ReaderExit::StdinError;
            }
        };
        // Every byte is typing activity: it (re)opens the mute window
        // that freezes animated progress output, and the first byte
        // after an idle period retakes the input row.
        let was_visible = line_visible;
        super::output::note_reader_keystroke();
        if !line_visible {
            redraw_input_line(&buffer);
            line_visible = true;
            // The redraw erased any kernel-echoed bytes with the row,
            // so the reader echoes everything from here on itself.
            echo_suppressed = false;
        }
        match byte {
            b'\r' | b'\n' => {
                // A draft hidden by the idle release must not be
                // submitted blindly: the Enter that reveals it is
                // swallowed, the next one submits.
                if !was_visible && !buffer.is_empty() {
                    continue;
                }
                println!();
                let line = match buffer.finish() {
                    Ok(line) => line,
                    Err(_) => {
                        dim_note("(discarded line with invalid UTF-8)");
                        browsing = None;
                        continue;
                    }
                };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    if let Some(position) = browsing.take() {
                        dim_note(&format!(
                            "(message {position} unchanged — empty edit ignored)"
                        ));
                    }
                    continue;
                }
                // Past this line, every byte is echoed by the reader.
                echo_suppressed = false;
                if let Some(position) = browsing.take() {
                    // Browsing: write the edited line back in place.
                    queue.replace_at(position, line.trim_end().to_string());
                    dim_note(&format!("(updated message {position})"));
                    continue;
                }
                if is_queue_command(trimmed) {
                    match interpret_queue_command(queue, trimmed) {
                        QueueCommandOutcome::Notice(notice) => dim_note(&notice),
                        QueueCommandOutcome::Browse { position } => {
                            if let Some(message) = queue.message_at(position) {
                                buffer = LineBuffer::default();
                                buffer.push_str(&message);
                                browsing = Some(position);
                                show_browsed_line(position, queue.len(), &message);
                            }
                        }
                    }
                } else {
                    let count = queue.enqueue(line.trim_end().to_string());
                    dim_note(&format!("queued ({count} pending)"));
                }
            }
            0x7f | 0x08 => {
                if buffer.backspace() {
                    // Kernel echo is off, so nothing was echoed for the
                    // DEL. Move left over the deleted character, redraw
                    // the tail, blank the vacated column, and return to
                    // the cursor position.
                    let tail: Vec<u8> = buffer.tail().to_vec();
                    print!("\u{8}");
                    if !tail.is_empty() {
                        let _ = io::stdout().write_all(&tail);
                    }
                    print!(" ");
                    for _ in 0..char_columns(&tail) + 1 {
                        print!("\u{8}");
                    }
                    let _ = io::stdout().flush();
                }
            }
            0x1b => {
                // Coalesce consecutive ESC bytes: pressing ESC and then
                // an arrow key quickly delivers ESC ESC [ A — the second
                // ESC introduces the sequence and must not be discarded
                // as an Alt modifier, or the [A tail leaks into the
                // buffer as text.
                let mut follow = read_byte_timeout(10).ok().flatten();
                while follow == Some(0x1b) {
                    follow = read_byte_timeout(10).ok().flatten();
                }
                if let Some(follow) = follow {
                    // Escape sequence: read to the final byte
                    // (0x40..=0x7E). Left/right/home/end move the edit
                    // cursor; everything else (other navigation keys,
                    // bracketed-paste markers) is ignored. Paste payloads
                    // between markers arrive as plain bytes and queue
                    // line-by-line, like typed input.
                    if follow == b'[' || follow == b'O' {
                        let mut final_byte = None;
                        for _ in 0..16 {
                            match read_byte_timeout(10) {
                                Ok(Some(byte)) if (0x40..=0x7e).contains(&byte) => {
                                    final_byte = Some(byte);
                                    break;
                                }
                                Ok(Some(_)) => continue,
                                _ => break,
                            }
                        }
                        // Up/down arrows walk the queue like shell
                        // history: browsing is non-destructive, Enter
                        // writes edits back in place, and walking past
                        // the end (down) returns to a fresh line.
                        if matches!(final_byte, Some(b'A') | Some(b'B')) {
                            let len = queue.len();
                            if len > 0 {
                                let next = if final_byte == Some(b'A') {
                                    // Up: start at the end, walk back,
                                    // clamped at the first message.
                                    browsing.map_or(len, |b| b.saturating_sub(1).max(1))
                                } else if let Some(current) = browsing {
                                    // Down: walk forward; past the end
                                    // exits browsing to a new line.
                                    if current >= len {
                                        browsing = None;
                                        buffer = LineBuffer::default();
                                        dim_note("(new line)");
                                        continue;
                                    }
                                    current + 1
                                } else {
                                    // Down with nothing browsed: no-op.
                                    continue;
                                };
                                if let Some(message) = queue.message_at(next) {
                                    buffer = LineBuffer::default();
                                    buffer.push_str(&message);
                                    browsing = Some(next);
                                    show_browsed_line(next, len, &message);
                                }
                            }
                            continue;
                        }
                        match final_byte {
                            Some(b'C') => {
                                if buffer.right() {
                                    print!("\u{1b}[C");
                                    let _ = io::stdout().flush();
                                }
                            }
                            Some(b'D') => {
                                if buffer.left() {
                                    print!("\u{8}");
                                    let _ = io::stdout().flush();
                                }
                            }
                            Some(b'H') => {
                                for _ in 0..buffer.home() {
                                    print!("\u{8}");
                                }
                                let _ = io::stdout().flush();
                            }
                            Some(b'F') => {
                                for _ in 0..buffer.end() {
                                    print!("\u{1b}[C");
                                }
                                let _ = io::stdout().flush();
                            }
                            _ => {}
                        }
                    }
                    continue;
                }
                // Lone ESC: layered action. Alt-modified keys (ESC
                // followed by a letter) lose the letter; they have no
                // meaning mid-turn.
                if let Some(position) = browsing.take() {
                    // Leave browsing; the message is untouched.
                    buffer = LineBuffer::default();
                    dim_note(&format!("(left message {position} unchanged)"));
                } else if !buffer.is_empty() {
                    buffer = LineBuffer::default();
                    dim_note("(line discarded)");
                } else {
                    dim_note("interrupting…");
                    cancel_token.cancel();
                    return ReaderExit::Stop;
                }
            }
            0x04 => { /* Ctrl+D has no EOF meaning mid-turn; ignore */ }
            0x15 => {
                // Ctrl+U drops the browsed message, or clears the line
                // being typed (readline's kill-line).
                if let Some(position) = browsing.take() {
                    queue.remove_at(position);
                    buffer = LineBuffer::default();
                    dim_note(&format!(
                        "(dropped message {position}, {} remain)",
                        queue.len()
                    ));
                } else if !buffer.is_empty() {
                    buffer = LineBuffer::default();
                    dim_note("(line cleared)");
                }
            }
            byte if byte >= 0x20 && byte != 0x7f => {
                // Printable ASCII and UTF-8 lead/continuation bytes.
                buffer.insert(byte);
                if !echo_suppressed {
                    // Echo the byte, then redraw the tail it displaced
                    // and park the cursor back on the insertion point.
                    let tail: Vec<u8> = buffer.tail().to_vec();
                    let _ = io::stdout().write_all(&[byte]);
                    if !tail.is_empty() {
                        let _ = io::stdout().write_all(&tail);
                        for _ in 0..char_columns(&tail) {
                            print!("\u{8}");
                        }
                    }
                    let _ = io::stdout().flush();
                }
            }
            _ => { /* other control bytes are ignored */ }
        }
    }
}

// ── /queue edit: rebase-style plan file ────────────────────────────────

const PLAN_HEADER: &str = "\
# Queued messages, in send order. Each message is one line starting
# with '- '. Edit freely: reorder, retext, or delete lines; empty the
# list to clear the queue. Lines starting with '#' are comments; any
# other non-blank line is an error and aborts the apply. Save and quit
# to apply; quit without saving to abort.";

/// Open the queue in the user's editor as a plan file. Returns `None`
/// when the editor exited unsuccessfully (abort), `Some(Ok(messages))`
/// with the (possibly empty) new queue on a clean parse, and
/// `Some(Err(..))` when the edited plan is malformed (the queue is left
/// untouched).
pub fn edit_queue_plan(messages: &[String]) -> anyhow::Result<Option<anyhow::Result<Vec<String>>>> {
    // Same resolution order as /edit: configured prompt editor first,
    // then $VISUAL / $EDITOR, then vim.
    let editor_cmd = goose::config::Config::global()
        .get_goose_prompt_editor()
        .ok()
        .flatten()
        .filter(|cmd| !cmd.is_empty())
        .or_else(|| std::env::var("VISUAL").ok())
        .or_else(|| std::env::var("EDITOR").ok())
        .unwrap_or_else(|| "vim".to_string());
    let parts = shlex::split(&editor_cmd)
        .ok_or_else(|| anyhow::anyhow!("could not parse editor command {editor_cmd:?}"))?;
    let Some(program) = parts.first() else {
        anyhow::bail!("editor command {editor_cmd:?} is empty");
    };

    let mut file = tempfile::Builder::new()
        .prefix("goose-queue-")
        .suffix(".txt")
        .tempfile()?;
    writeln!(file, "{PLAN_HEADER}")?;
    for message in messages {
        // Single-line plan entries: newlines inside a message become
        // spaces so the file stays one-message-per-line.
        writeln!(file, "- {}", message.replace('\n', " "))?;
    }
    file.flush()?;
    let path = file.path().to_path_buf();

    let mut command = std::process::Command::new(program);
    if parts.len() > 1 {
        command.args(&parts[1..]);
    }
    let status = command
        .arg(&path)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()?;

    if !status.success() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)?;
    Ok(Some(parse_plan(&content)))
}

/// Parse a plan file into messages: each `- ` line is one message, `#`
/// lines are comments, blank lines are ignored, anything else is an
/// error. Pure so it can be unit tested.
pub fn parse_plan(content: &str) -> anyhow::Result<Vec<String>> {
    let mut messages = Vec::new();
    for (number, line) in content.lines().enumerate() {
        let line = line.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let Some(message) = line.strip_prefix("- ") else {
            return Err(anyhow::anyhow!(
                "line {}: expected '- ' prefix or '#': {:?}",
                number + 1,
                preview_line(line, 40)
            ));
        };
        messages.push(message.trim().to_string());
    }
    Ok(messages)
}

/// Serve a queued line through the same interpretation as a typed
/// prompt line, so queued slash commands and bare exit/quit behave
/// exactly as if typed.
pub fn interpret_queued_line(line: &str) -> InputResult {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return InputResult::Retry;
    }
    if trimmed.starts_with('/') {
        if let Some(result) = crate::session::input::handle_slash_command(trimmed) {
            return result;
        }
    }
    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
        return InputResult::Exit;
    }
    InputResult::Message(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plan_ignores_comments_and_blanks() {
        let plan = "# header\n- first\n\n  # indented comment\n-  second  \n\n";
        assert_eq!(parse_plan(plan).unwrap(), vec!["first", "second"]);
    }

    #[test]
    fn parse_plan_preserves_hash_prefixed_messages() {
        let plan = "- # not a comment\n- plain";
        assert_eq!(parse_plan(plan).unwrap(), vec!["# not a comment", "plain"]);
    }

    #[test]
    fn parse_plan_empty_list_clears_queue() {
        assert!(parse_plan("# only comments\n\n").unwrap().is_empty());
    }

    #[test]
    fn parse_plan_rejects_unprefixed_lines() {
        let plan = "- good\nbroken line\n";
        let error = parse_plan(plan).unwrap_err().to_string();
        assert!(error.contains("line 2"), "unexpected error: {error}");
    }

    #[test]
    fn queue_command_detection() {
        assert!(is_queue_command("/queue"));
        assert!(is_queue_command(" /queue edit "));
        assert!(is_queue_command("/queue resume"));
        assert!(!is_queue_command("/queueanything"));
        assert!(!is_queue_command("/quiet"));
        assert!(!is_queue_command("queue edit"));
    }

    #[test]
    fn detached_queue_roundtrip() {
        let queue = SharedQueue::detached();
        assert!(queue.is_empty());
        queue.enqueue("one".into());
        queue.enqueue("two".into());
        assert_eq!(queue.len(), 2);
        assert!(queue.ready_to_send());
        queue.hold();
        assert!(!queue.ready_to_send());
        assert!(queue.is_held());
        queue.unhold();
        assert_eq!(queue.message_at(2).as_deref(), Some("two"));
        queue.replace_at(2, "two-edited".into());
        assert!(queue.ready_to_send());
        assert_eq!(queue.pop_head().as_deref(), Some("one"));
        assert_eq!(queue.pop_head().as_deref(), Some("two-edited"));
        assert!(queue.is_empty());
    }

    #[test]
    fn held_queue_released_by_replace_all() {
        let queue = SharedQueue::detached();
        queue.enqueue("one".into());
        queue.hold();
        queue.replace_all(vec!["a".into(), "b".into()]);
        assert!(!queue.is_held());
        assert!(queue.ready_to_send());
    }

    #[test]
    fn pending_edit_gates_auto_send() {
        let queue = SharedQueue::detached();
        queue.enqueue("one".into());
        assert!(queue.ready_to_send());
        queue.request_edit();
        assert!(!queue.ready_to_send());
        assert!(queue.take_pending_edit());
        assert!(queue.ready_to_send());
    }

    #[test]
    fn replace_all_resets_queue() {
        let queue = SharedQueue::detached();
        queue.enqueue("old".into());
        queue.replace_all(vec!["a".into(), "b".into()]);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.head().as_deref(), Some("a"));
    }

    #[test]
    fn preview_line_truncates_long_messages() {
        assert_eq!(preview_line("short", 72), "short");
        let long = "x".repeat(80);
        let preview = preview_line(&long, 72);
        assert!(preview.ends_with('…'));
        assert_eq!(preview.chars().count(), 73);
    }

    #[test]
    fn line_buffer_handles_utf8() {
        let mut buffer = LineBuffer::default();
        for byte in "café".bytes() {
            buffer.insert(byte);
        }
        assert_eq!(buffer.finish().unwrap(), "café");
        for byte in "日".bytes() {
            buffer.insert(byte);
        }
        assert!(buffer.backspace());
        assert!(buffer.is_empty());
        assert_eq!(buffer.finish().unwrap(), "");
    }

    #[test]
    fn line_buffer_cursor_editing() {
        let mut buffer = LineBuffer::default();
        buffer.push_str("abcd");
        assert!(!buffer.right());
        assert!(buffer.left());
        assert!(buffer.left());
        // cursor between 'b' and 'c': "ab|cd"
        buffer.insert(b'X');
        assert_eq!(buffer.finish().unwrap(), "abXcd");
    }

    #[test]
    fn line_buffer_backspace_at_cursor() {
        let mut buffer = LineBuffer::default();
        buffer.push_str("abcd");
        assert!(buffer.left());
        assert!(buffer.left());
        // "ab|cd": backspace removes 'b'
        assert!(buffer.backspace());
        assert_eq!(buffer.finish().unwrap(), "acd");
    }

    #[test]
    fn line_buffer_home_end() {
        let mut buffer = LineBuffer::default();
        buffer.push_str("abcd");
        assert_eq!(buffer.home(), 4);
        assert!(buffer.right());
        assert_eq!(buffer.end(), 3);
        assert_eq!(buffer.finish().unwrap(), "abcd");
    }

    #[test]
    fn line_buffer_utf8_cursor_movement() {
        let mut buffer = LineBuffer::default();
        buffer.push_str("aéz");
        // Move left twice: lands before 'é', then before 'a' — each move
        // crosses one multi-byte character cleanly.
        assert!(buffer.left());
        assert!(buffer.left());
        buffer.insert(b'X');
        assert_eq!(buffer.finish().unwrap(), "aXéz");
    }

    #[test]
    fn line_buffer_rejects_half_typed_sequence() {
        let mut buffer = LineBuffer::default();
        buffer.insert(0x61); // 'a'
        buffer.insert(0xE2); // UTF-8 lead byte, incomplete
        assert!(buffer.finish().is_err());
    }

    #[test]
    fn line_buffer_contents_expose_full_line_for_redraw() {
        let mut buffer = LineBuffer::default();
        buffer.push_str("draft");
        assert_eq!(buffer.contents(), b"draft");
        buffer.left();
        assert_eq!(buffer.contents(), b"draft", "contents ignore cursor");
    }

    #[test]
    fn queue_numeric_browse_is_non_destructive() {
        let queue = SharedQueue::detached();
        queue.enqueue("one".into());
        queue.enqueue("two".into());
        queue.enqueue("three".into());
        match interpret_queue_command(&queue, "/queue 2") {
            QueueCommandOutcome::Browse { position } => assert_eq!(position, 2),
            other => panic!("expected browse, got: {}", other.as_notice()),
        }
        // Browsing does not remove anything.
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.message_at(2).as_deref(), Some("two"));
    }

    #[test]
    fn queue_drop_removes_position() {
        let queue = SharedQueue::detached();
        queue.enqueue("one".into());
        queue.enqueue("two".into());
        queue.enqueue("three".into());
        match interpret_queue_command(&queue, "/queue drop 2") {
            QueueCommandOutcome::Notice(notice) => {
                assert!(notice.contains("dropped message 2"), "got: {notice}")
            }
            QueueCommandOutcome::Browse { .. } => panic!("expected a notice"),
        }
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.message_at(2).as_deref(), Some("three"));
    }

    #[test]
    fn queue_numeric_recall_out_of_range() {
        let queue = SharedQueue::detached();
        queue.enqueue("one".into());
        match interpret_queue_command(&queue, "/queue 5") {
            QueueCommandOutcome::Notice(notice) => {
                assert!(notice.contains("no queued message 5"), "got: {notice}")
            }
            QueueCommandOutcome::Browse { .. } => panic!("expected a notice"),
        }
    }

    #[test]
    fn interpret_queued_line_routes_commands_and_text() {
        assert!(matches!(interpret_queued_line("/exit"), InputResult::Exit));
        assert!(matches!(interpret_queued_line("exit"), InputResult::Exit));
        assert!(matches!(interpret_queued_line("QUIT"), InputResult::Exit));
        assert!(matches!(
            interpret_queued_line("  hello there "),
            InputResult::Message(text) if text == "hello there"
        ));
        assert!(matches!(interpret_queued_line("   "), InputResult::Retry));
    }

    #[test]
    fn pending_draft_roundtrip_and_idle_sidecar_semantics() {
        let queue = SharedQueue::detached();
        assert!(queue.take_pending_draft().is_none());
        queue.set_pending_draft("half-typed".into());
        // A pending draft does not gate auto-send of queued messages.
        queue.enqueue("one".into());
        assert!(queue.ready_to_send());
        assert_eq!(queue.take_pending_draft().as_deref(), Some("half-typed"));
        assert!(queue.take_pending_draft().is_none());
        assert_eq!(queue.pop_head().as_deref(), Some("one"));
    }

    #[test]
    fn salvage_persists_fresh_draft_only_on_final_stop() {
        let queue = SharedQueue::detached();

        // Final stop: fresh partial line is saved, not discarded.
        let mut buffer = LineBuffer::default();
        buffer.push_str("draft in progress");
        salvage_partial_line(&queue, &mut buffer, None, true, true);
        assert_eq!(
            queue.take_pending_draft().as_deref(),
            Some("draft in progress")
        );

        // Yield/stdin error: fresh partial line is dropped, an already
        // saved draft is left untouched.
        queue.set_pending_draft("saved earlier".into());
        let mut buffer = LineBuffer::default();
        buffer.push_str("typed during yield");
        salvage_partial_line(&queue, &mut buffer, None, false, true);
        assert_eq!(queue.take_pending_draft().as_deref(), Some("saved earlier"));

        // Empty and whitespace-only buffers never produce a draft.
        let mut buffer = LineBuffer::default();
        salvage_partial_line(&queue, &mut buffer, None, true, true);
        assert!(queue.take_pending_draft().is_none());
        let mut buffer = LineBuffer::default();
        buffer.push_str("   ");
        salvage_partial_line(&queue, &mut buffer, None, true, true);
        assert!(queue.take_pending_draft().is_none());
    }

    #[test]
    fn salvage_keeps_browsing_edits_over_fresh_draft() {
        let queue = SharedQueue::detached();
        queue.enqueue("original".into());
        let mut buffer = LineBuffer::default();
        buffer.push_str("edited");
        salvage_partial_line(&queue, &mut buffer, Some(1), true, true);
        assert!(queue.take_pending_draft().is_none());
        assert_eq!(queue.message_at(1).as_deref(), Some("edited"));
    }

    #[test]
    fn queue_state_deserializes_sidecar_without_pending_draft() {
        // Sidecars written before the draft field existed must load.
        let legacy = r#"{"messages":["one"],"held":false,"pending_edit":false}"#;
        let state: QueueState = serde_json::from_str(legacy).unwrap();
        assert_eq!(state.messages.len(), 1);
        assert!(state.pending_draft.is_none());
    }
}
