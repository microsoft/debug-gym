## Bound command output to prevent resource exhaustion

### Problem

`DockerTerminal.exec_run()` buffers the entire stdout/stderr with no size limit. A single runaway command (e.g. `yes ...`) can allocate gigabytes in memory, produce 77GB log files, and crash the host.

### Fix

**Raise `UnrecoverableTerminalError` when output exceeds `max_output_bytes` (default 100MB).**

Instead of silently truncating, we terminate the agent episode — there's no point continuing with a corrupt/oversized output that would either OOM or flood the LLM context.

#### Docker
- Switched from `container.exec_run()` to `api.exec_create` + `api.exec_start(stream=True)` — reads output in chunks, never buffers the full blob in memory.
- Uses `_safe_closing()` to cleanly handle the case where a container is killed externally (watchdog, `env.close()`), suppressing `ValueError`/`OSError` from dead sockets.

#### Kubernetes
- Early exit from the streaming loop when limit is exceeded, then raises `UnrecoverableTerminalError`.

#### Local
- Checks output size after `process.communicate()` in both normal and timeout paths.

#### Base class (`Terminal`)
- `max_output_bytes` parameter (default `100MB`, configurable via YAML).
- `_raise_output_limit_exceeded()` helper — includes a 2KB preview in the error message for debugging.
- `_truncate_output()` retained for any future use but no longer called in the main paths.

### Config

```yaml
terminal:
  type: docker
  max_output_bytes: 100000000  # 100MB, default
```

Set to `0` to disable the limit.
