//! The ONNX models' threads under an affinity limit (issue #14). Left to
//! choose, ONNX Runtime starts one thread per core of the machine and pins
//! each to its core, outside any `taskset` or cpuset the process runs
//! under. Loaded with `threads: None`, Kokoro and the speaker model must
//! start no more threads than the CPUs they may use, and none elsewhere.
//!
//! One test in its own binary, so every thread that appears while a model
//! loads is that model's.
#![cfg(target_os = "linux")]

use loqui_kokoro::{Device, KokoroModel};
use loqui_speaker::SpeakerEncoder;
use rustix::thread::{CpuSet, Pid, sched_getaffinity, sched_setaffinity};

#[test]
fn onnx_models_keep_their_threads_on_the_cpus_they_may_use() {
    let allowed = cpus(&sched_getaffinity(None).expect("this thread's affinity"));
    // A limit of two CPUs, so a model has a pool to start. An affinity mask
    // is only where threads begin: ONNX Runtime's own pins reach the rest of
    // the machine even when this process may use no more than these two.
    let [a, b, ..] = allowed[..] else {
        eprintln!("skipped: needs 2 CPUs, and this process may use {}", allowed.len());
        return;
    };
    let path = std::env::temp_dir().join(format!("loqui-threads-{}.onnx", std::process::id()));
    std::fs::write(&path, tiny_model()).unwrap();
    // On a thread of its own, so the harness keeps its CPUs; threads the
    // models start inherit this one's.
    let limited = std::thread::spawn({
        let path = path.clone();
        move || {
            let mut limit = CpuSet::new();
            limit.set(a);
            limit.set(b);
            sched_setaffinity(None, &limit).expect("limit this thread to two CPUs");
            assert_threads_stay_on(&[a, b], "Kokoro", || KokoroModel::load(&path, Device::Cpu, None));
            assert_threads_stay_on(&[a, b], "the speaker model", || SpeakerEncoder::load(&path, None));
        }
    })
    .join();
    let _ = std::fs::remove_file(&path);
    if let Err(panic) = limited {
        std::panic::resume_unwind(panic);
    }
}

/// Loads a model and checks the threads that appeared meanwhile: at least
/// one, so the check is not vacuous, no more than `limit` has CPUs, and none
/// that may run outside it.
fn assert_threads_stay_on<M, E: std::fmt::Debug>(limit: &[usize], name: &str, load: impl FnOnce() -> Result<M, E>) {
    let before = threads();
    let model = load().unwrap_or_else(|e| panic!("{name} did not load: {e:?}"));
    let started: Vec<i32> = threads().into_iter().filter(|t| !before.contains(t)).collect();
    // A thread that has already exited has no affinity to read.
    let escaped: Vec<(i32, Vec<usize>)> = started
        .iter()
        .filter_map(|&tid| Some((tid, cpus(&sched_getaffinity(Some(Pid::from_raw(tid)?)).ok()?))))
        .filter(|(_, cpus)| !cpus.iter().all(|c| limit.contains(c)))
        .collect();
    assert!(escaped.is_empty(), "{name}: threads that may run outside CPUs {limit:?} (thread, CPUs): {escaped:?}");
    assert!(!started.is_empty(), "{name} started no threads, so nothing was checked");
    assert!(started.len() <= limit.len(), "{name} started {} threads for {} CPUs", started.len(), limit.len());
    drop(model);
}

/// The ids of this process's threads.
fn threads() -> Vec<i32> {
    std::fs::read_dir("/proc/self/task")
        .expect("/proc/self/task")
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok())
        .collect()
}

fn cpus(set: &CpuSet) -> Vec<usize> {
    (0..CpuSet::MAX_CPU).filter(|&cpu| set.is_set(cpu)).collect()
}

/// An ONNX model with one `Identity` node, float[1] in and out, and the
/// metadata [`SpeakerEncoder::load`] checks, so both crates load it. Written
/// in protobuf's wire format; the field numbers are onnx.proto's.
fn tiny_model() -> Vec<u8> {
    // TypeProto { tensor_type: { elem_type: FLOAT, shape: { dim: [{ dim_value: 1 }] } } }
    let float1 = message(1, &[int(1, 1), message(2, &message(1, &int(1, 1)))].concat());
    // ValueInfoProto { name, type }
    let value = |name: &str| [message(1, name.as_bytes()), message(2, &float1)].concat();
    // NodeProto { input, output, op_type }
    let node = [message(1, b"x"), message(2, b"y"), message(4, b"Identity")].concat();
    // GraphProto { node, name, input, output }
    let graph = [message(1, &node), message(2, b"identity"), message(11, &value("x")), message(12, &value("y"))].concat();
    // ModelProto.metadata_props: StringStringEntryProto { key, value }
    let meta = |key: &str, value: &str| message(14, &[message(1, key.as_bytes()), message(2, value.as_bytes())].concat());
    // ModelProto { ir_version: 8, opset_import: [{ version: 13 }], graph, metadata_props }
    [
        int(1, 8),
        message(8, &int(2, 13)),
        message(7, &graph),
        meta("sample_rate", "16000"),
        meta("normalize_samples", "0"),
        meta("output_dim", "1"),
    ]
    .concat()
}

/// A varint field.
fn int(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}

/// A length-delimited field: a string, bytes or an embedded message.
fn message(field: u64, value: &[u8]) -> Vec<u8> {
    [varint(field << 3 | 2), varint(value.len() as u64), value.to_vec()].concat()
}

fn varint(mut n: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while n >= 0x80 {
        out.push(n as u8 | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
    out
}
