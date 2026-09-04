# NeverGPU

A sandboxed virtual GPU and portable compute device.

## Goal

NeverGPU provides a small, explicit virtual GPU interface that can execute parallel compute workloads without exposing applications directly to host GPU drivers.

The same virtual device is intended to support multiple execution backends:

- CPU software backend (reference implementation)
- Vulkan backend (future hardware acceleration)
- WebGPU backend (future browser/mobile path)
- Remote backend (future)

## Architecture

```text
Application
    |
    v
NeverGPU API
    |
    v
Command validation + resource limits
    |
    v
Virtual device
    |
    +--> CPU backend
    +--> Vulkan backend
    +--> WebGPU backend
    +--> Remote backend
```

## v0.1 target

The first milestone is deliberately small:

1. Virtual device discovery
2. Device-local buffers
3. Upload/download operations
4. Parallel kernel dispatch
5. Explicit synchronization
6. CPU reference backend
7. Deterministic tests
8. Resource and execution limits

The CPU backend is not intended to compete with a physical GPU. It is the correctness reference and establishes the virtual hardware contract before accelerated backends are added.

## Design principles

- Backend-independent API
- Sandboxed execution
- Explicit memory ownership
- Deterministic behavior where practical
- No dependency on a physical GPU for development or testing
- Hardware acceleration must remain an implementation detail

## Status

**Phase 0 — repository initialized.**

Next: implement the minimal virtual device, buffer model, command protocol, and CPU compute backend.
