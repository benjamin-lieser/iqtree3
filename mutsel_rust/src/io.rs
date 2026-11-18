use std::io::Write;


const STDOUT_FD: libc::c_int = 1;
const STDERR_FD: libc::c_int = 2;

unsafe fn os_dup(fd: libc::c_int) -> libc::c_int {
    unsafe { libc::dup(fd) }
}

unsafe fn os_dup2(src: libc::c_int, dst: libc::c_int) -> libc::c_int {
    unsafe { libc::dup2(src, dst) }
}


unsafe fn os_close(fd: libc::c_int) -> libc::c_int {
    unsafe { libc::close(fd) }
}

#[cfg(unix)]
unsafe fn os_read(fd: libc::c_int, buf: &mut [u8]) -> isize {
    unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) as isize }
}

#[cfg(windows)]
unsafe fn os_read(fd: libc::c_int, buf: &mut [u8]) -> isize {
    unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as libc::c_uint) as isize }
}

#[cfg(unix)]
unsafe fn os_write(fd: libc::c_int, buf: &[u8]) -> isize {
    unsafe { libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len()) as isize }
}

#[cfg(windows)]
unsafe fn os_write(fd: libc::c_int, buf: &[u8]) -> isize {
    unsafe { libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len() as libc::c_uint) as isize }
}

#[cfg(unix)]
fn create_pipe() -> std::io::Result<[libc::c_int; 2]> {
    let mut fds = [0; 2];
    let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if ret != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(fds)
    }
}

#[cfg(windows)]
fn create_pipe() -> std::io::Result<[libc::c_int; 2]> {
    let mut fds = [0; 2];
    // libc::pipe on Windows maps to the CRT signature (fds, psize, textmode).
    let ret = unsafe { libc::pipe(fds.as_mut_ptr(), 4096, 0) };
    if ret != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(fds)
    }
}

fn write_all_fd(fd: libc::c_int, mut buf: &[u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        let n = unsafe { os_write(fd, buf) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        let n = n as usize;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "write to redirected stdout returned 0",
            ));
        }
        buf = &buf[n..];
    }
    Ok(())
}

pub fn tee_to_file(path: &str) -> (i32, i32, std::thread::JoinHandle<()>) {
    let fds = create_pipe().unwrap_or_else(|e| panic!("tee_to_file: pipe() failed: {}", e));

    // Save original stdout
    let saved_stdout = unsafe { os_dup(STDOUT_FD) };
    if saved_stdout < 0 {
        panic!(
            "tee_to_file: dup(STDOUT_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    let saved_stderr = unsafe { os_dup(STDERR_FD) };
    if saved_stderr < 0 {
        panic!(
            "tee_to_file: dup(STDERR_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Dedicated copy for tee thread so saved_stdout can be restored/closed independently.
    let tee_stdout = unsafe { os_dup(saved_stdout) };
    if tee_stdout < 0 {
        panic!(
            "tee_to_file: dup(saved_stdout) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Redirect stdout to pipe write end
    let ret = unsafe { os_dup2(fds[1], STDOUT_FD) };
    if ret < 0 {
        panic!(
            "tee_to_file: dup2(STDOUT_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    let ret = unsafe { os_dup2(fds[1], STDERR_FD) };
    if ret < 0 {
        panic!(
            "tee_to_file: dup2(STDERR_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Close the original write end — stdout/stderr now hold the only references
    let ret = unsafe { os_close(fds[1]) };
    if ret != 0 {
        panic!(
            "tee_to_file: close(pipe write end) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Open log file
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("tee_to_file: failed to open '{}': {}", path, e));

    // Spawn thread to read from pipe and write to both
    let handle = std::thread::spawn(move || {
        let mut buffer = [0u8; 1024];

        loop {
            let n = unsafe { os_read(fds[0], &mut buffer) };
            if n == 0 {
                break; // EOF
            }
            if n < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                panic!("tee_to_file: read from pipe failed: {}", err);
            }

            let n = n as usize;
            // Write to original stdout
            write_all_fd(tee_stdout, &buffer[..n])
                .unwrap_or_else(|e| panic!("tee_to_file: write to stdout failed: {}", e));
            // Write to file
            file.write_all(&buffer[..n])
                .unwrap_or_else(|e| panic!("tee_to_file: write to log file failed: {}", e));
        }

        let ret = unsafe { os_close(fds[0]) };
        if ret != 0 {
            panic!(
                "tee_to_file: close(pipe read end) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        let ret = unsafe { os_close(tee_stdout) };
        if ret != 0 {
            panic!(
                "tee_to_file: close(tee stdout) failed: {}",
                std::io::Error::last_os_error()
            );
        }

        file.flush()
            .unwrap_or_else(|e| panic!("tee_to_file: flush log file failed: {}", e));
    });

    (saved_stdout, saved_stderr, handle)
}

pub fn restore_stdout_stderr(
    saved_stdout: i32,
    saved_stderr: i32,
    handle: std::thread::JoinHandle<()>,
) {
    unsafe {
        // Restoring stdout/stderr drops the last write-end references to the pipe,
        // causing EOF on the read end so the tee thread can exit cleanly.
        os_dup2(saved_stdout, STDOUT_FD);
        os_dup2(saved_stderr, STDERR_FD);
    }
    // Wait for the tee thread to drain and flush all remaining output.
    handle
        .join()
        .unwrap_or_else(|_| panic!("restore_stdout_stderr: tee thread panicked"));
    unsafe {
        os_close(saved_stdout);
        os_close(saved_stderr);
    }
}