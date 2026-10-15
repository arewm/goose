//! Foreground-terminal helpers for the interactive prompt.
//!
//! A process that is not the terminal's foreground process group is stopped
//! with `SIGTTOU` the moment it tries to control the terminal (rustyline's
//! raw-mode `tcsetattr`, cliclack prompts) and with `SIGTTIN` when it reads
//! input. Before goose reads a prompt it should therefore make sure it is
//! in the foreground, printing a notice instead of silently hanging
//! suspended.
//!
//! This is UX hardening for job-control interaction, not a security
//! boundary.

/// Block until this process's process group is the terminal's foreground
/// process group.
///
/// Returns `true` when it is safe to proceed (foreground, or no meaningful
/// foreground to wait for) and `false` when the cancel token fired while
/// waiting.
///
/// Returns immediately when stdin is not a terminal (tests, headless runs)
/// or when the controlling terminal cannot be determined. When nobody can
/// ever bring us to the foreground — orphaned process group, a foreground
/// group that no longer exists, or a terminal belonging to another session
/// — we also proceed: the subsequent read fails with `EIO`, which beats an
/// unbounded silent poll.
///
/// Residual race: the foreground can be lost again between this check and
/// the caller's raw-mode `tcsetattr`, which would stop us with SIGTTOU.
/// That stop is recoverable — resuming with `fg` surfaces as `EINTR`, which
/// the prompt-read path retries — so the race is benign.
#[cfg(unix)]
pub fn wait_for_foreground(cancel_token: Option<&tokio_util::sync::CancellationToken>) -> bool {
    use std::io::IsTerminal;

    if !std::io::stdin().is_terminal() {
        return true;
    }

    let mut notified = false;
    loop {
        if let Some(token) = cancel_token {
            if token.is_cancelled() {
                return false;
            }
        }
        match owning_pgrp() {
            None => return true,
            Some(foreground) if foreground == our_pgrp() => return true,
            Some(foreground) => {
                if nobody_can_foreground_us(foreground) {
                    return true;
                }
                if !notified {
                    notified = true;
                    print_background_notice();
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
}

/// Whether the caller's process group currently owns the terminal that
/// stdin is attached to. `None` when the answer cannot be determined (no
/// controlling terminal, closed terminal); callers treat that as "nothing
/// to wait for".
#[cfg(unix)]
fn owning_pgrp() -> Option<libc::pid_t> {
    use std::os::fd::AsRawFd;
    // SAFETY: tcgetpgrp only queries terminal state; it has no side effects.
    let foreground = unsafe { libc::tcgetpgrp(std::io::stdin().as_raw_fd()) };
    if foreground < 0 {
        return None;
    }
    Some(foreground)
}

#[cfg(unix)]
fn our_pgrp() -> libc::pid_t {
    // SAFETY: getpgrp never fails and has no side effects.
    unsafe { libc::getpgrp() }
}

/// Detect the situations in which waiting for the foreground would hang
/// forever because no one can ever grant it to us.
#[cfg(unix)]
fn nobody_can_foreground_us(foreground: libc::pid_t) -> bool {
    use std::os::fd::AsRawFd;
    // Reparented to init while backgrounded: the shell that tracked this
    // job is gone and nothing will foreground us. Checked on every
    // iteration so a parent that exits mid-wait is caught too. (A subreaper
    // parent is not detected by this check; the dead-foreground and
    // foreign-session checks below cover the common subreaper outcomes.)
    // SAFETY: getppid never fails and has no side effects.
    let ppid = unsafe { libc::getppid() };
    // The foreground process group no longer exists: the terminal's owner
    // died without handing the foreground over. Signal the whole group
    // (negative pid): testing only the leader misclassifies a group whose
    // leader exited while other members live.
    // SAFETY: kill with signal 0 only checks for existence.
    let foreground_alive = unsafe { libc::kill(-foreground, 0) } == 0
        || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH);
    // stdin is a terminal that belongs to a different session (e.g. an
    // explicit `goose < /dev/pts/N`): our process group can never become
    // that terminal's foreground group.
    // SAFETY: tcgetsid only queries terminal state.
    let terminal_session = unsafe { libc::tcgetsid(std::io::stdin().as_raw_fd()) };
    // SAFETY: getsid never fails for our own process.
    let own_session = unsafe { libc::getsid(0) };
    should_proceed_despite_background(ppid, foreground_alive, terminal_session, own_session)
}

/// Pure decision core of [`nobody_can_foreground_us`], unit-testable.
/// `terminal_session` is negative when it could not be determined.
#[cfg(unix)]
fn should_proceed_despite_background(
    ppid: libc::pid_t,
    foreground_alive: bool,
    terminal_session: libc::pid_t,
    own_session: libc::pid_t,
) -> bool {
    if ppid == 1 {
        return true;
    }
    if !foreground_alive {
        return true;
    }
    terminal_session >= 0 && terminal_session != own_session
}

/// Tell the user we are parked, on whichever standard stream is a terminal.
#[cfg(unix)]
fn print_background_notice() {
    use std::io::IsTerminal;
    let notice = "goose is in the background; bring it to the foreground with `fg`";
    if std::io::stderr().is_terminal() {
        eprintln!("{notice}");
    } else if std::io::stdout().is_terminal() {
        println!("{notice}");
    }
}

#[cfg(not(unix))]
pub fn wait_for_foreground(_cancel_token: Option<&tokio_util::sync::CancellationToken>) -> bool {
    true
}

/// Decide, from the owning process group and our own, whether we are in the
/// foreground. Pure so it can be unit tested.
#[cfg(all(test, unix))]
fn in_foreground(foreground: Option<libc::pid_t>, own: libc::pid_t) -> bool {
    foreground.map_or(true, |fg| fg == own)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn foreground_when_owning_or_unknown() {
        assert!(in_foreground(Some(42), 42));
        assert!(in_foreground(None, 42));
    }

    #[test]
    fn background_when_other_group_owns_terminal() {
        assert!(!in_foreground(Some(7), 42));
    }

    #[test]
    fn proceeds_when_parent_is_init() {
        assert!(should_proceed_despite_background(1, true, 100, 100));
    }

    #[test]
    fn proceeds_when_foreground_group_is_dead() {
        assert!(should_proceed_despite_background(500, false, 100, 100));
    }

    #[test]
    fn proceeds_when_terminal_belongs_to_foreign_session() {
        assert!(should_proceed_despite_background(500, true, 100, 200));
    }

    #[test]
    fn waits_in_ordinary_background_case() {
        assert!(!should_proceed_despite_background(500, true, 100, 100));
    }
}
