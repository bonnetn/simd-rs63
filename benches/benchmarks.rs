use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simd_rs63::{encode, recover, K, M, N, BLOCK_ALIGNMENT};

const SIZES_KIB: &[usize] = &[64, 256, 1024];

fn make_blocks(block_size: usize) -> Vec<Vec<u8>> {
    let mut data: Vec<Vec<u8>> = (0..K).map(|i| vec![i as u8; block_size]).collect();
    let mut parity: [Vec<u8>; M] = std::array::from_fn(|_| vec![0u8; block_size]);
    let [p0, p1, p2] = &mut parity;
    encode(
        std::array::from_fn(|i| data[i].as_slice()),
        [p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice()],
    ).unwrap();
    data.extend(parity);
    data
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    for &kib in SIZES_KIB {
        let bs = kib * 1024;
        assert!(bs % BLOCK_ALIGNMENT == 0);
        group.throughput(Throughput::Bytes((N * bs) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{kib}KiB")), &bs, |b, &bs| {
            let mut data: Vec<Vec<u8>> = (0..K).map(|i| vec![i as u8; bs]).collect();
            let mut parity: [Vec<u8>; M] = std::array::from_fn(|_| vec![0u8; bs]);
            b.iter(|| {
                let [p0, p1, p2] = &mut parity;
                encode(
                    std::array::from_fn(|i| data[i].as_slice()),
                    [p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice()],
                ).unwrap();
                let _ = &mut data;
            });
        });
    }
    group.finish();
}

fn bench_recover_1(c: &mut Criterion) {
    let mut group = c.benchmark_group("recover_1");
    for &kib in SIZES_KIB {
        let bs = kib * 1024;
        group.throughput(Throughput::Bytes((N * bs) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{kib}KiB")), &bs, |b, &bs| {
            let blocks = make_blocks(bs);
            let mut out = vec![0u8; bs];
            b.iter(|| {
                recover(
                    [
                        (0, blocks[0].as_slice()),
                        (1, blocks[1].as_slice()),
                        (2, blocks[2].as_slice()),
                        (3, blocks[3].as_slice()),
                        (4, blocks[4].as_slice()),
                        (6, blocks[6].as_slice()),
                    ],
                    [(5, &mut out)],
                ).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_recover_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("recover_2");
    for &kib in SIZES_KIB {
        let bs = kib * 1024;
        group.throughput(Throughput::Bytes((N * bs) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{kib}KiB")), &bs, |b, &bs| {
            let blocks = make_blocks(bs);
            let mut out0 = vec![0u8; bs];
            let mut out1 = vec![0u8; bs];
            b.iter(|| {
                recover(
                    [
                        (0, blocks[0].as_slice()),
                        (1, blocks[1].as_slice()),
                        (2, blocks[2].as_slice()),
                        (3, blocks[3].as_slice()),
                        (6, blocks[6].as_slice()),
                        (7, blocks[7].as_slice()),
                    ],
                    [(4, &mut out0), (5, &mut out1)],
                ).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_recover_3(c: &mut Criterion) {
    let mut group = c.benchmark_group("recover_3");
    for &kib in SIZES_KIB {
        let bs = kib * 1024;
        group.throughput(Throughput::Bytes((N * bs) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{kib}KiB")), &bs, |b, &bs| {
            let blocks = make_blocks(bs);
            let mut out0 = vec![0u8; bs];
            let mut out1 = vec![0u8; bs];
            let mut out2 = vec![0u8; bs];
            b.iter(|| {
                recover(
                    [
                        (0, blocks[0].as_slice()),
                        (1, blocks[1].as_slice()),
                        (2, blocks[2].as_slice()),
                        (6, blocks[6].as_slice()),
                        (7, blocks[7].as_slice()),
                        (8, blocks[8].as_slice()),
                    ],
                    [(3, &mut out0), (4, &mut out1), (5, &mut out2)],
                ).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_recover_1, bench_recover_2, bench_recover_3);
criterion_main!(benches);
