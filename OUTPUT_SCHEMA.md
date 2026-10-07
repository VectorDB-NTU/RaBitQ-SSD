# Output CSV schema

The search and build stages write CSVs with a fixed column set. This document
defines them, so the numbers can be read — or compared against another
system — without reverse-engineering the code.

A column the run cannot fill is left **blank** rather than renamed or
reordered. Method-specific diagnostics go in a separate `method_ext.csv`, which
is not part of this schema and carries no stability guarantee.

## 1. Query CSV — one row per operating point

32 columns, in this order:

```
run_id,system,dataset,metric_space,topk,threads,io_backend,cache_state,
rounds,nq,op_param_name,op_param,recall_at_topk,qps,
lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,lat_p999_us,
io_reqs_mean,io_pages_mean,io_bytes_mean,io_pages_p99,io_pages_max,io_bytes_p99,io_bytes_max,
eff_bw_mbps,eff_iops,cpu_util_pct,peak_rss_mb,resident_index_mb,index_ssd_bytes
```

| column | unit | filled by | definition |
|---|---|---|---|
| run_id | str | driver | unique id; the merge key for the driver-filled columns |
| system | str | driver | system label (e.g. `RaBitQ-SSD`) |
| dataset | str | driver | dataset name |
| metric_space | l2\|ip\|cosine | driver | |
| topk | int | driver | neighbors retrieved |
| threads | int | driver | query threads |
| io_backend | str | binary | `libaio` |
| cache_state | str | const | `odirect`; all SSD reads bypass the page cache |
| rounds | int | binary | timing rounds |
| nq | int | binary | number of queries |
| op_param_name | str | binary | name of the swept knob (`nprobe` here) |
| op_param | num | binary | its value |
| recall_at_topk | [0,1] | binary (offline) | result ids vs ground truth, computed outside the timed segment |
| qps | q/s | derived | nq / timed wall-seconds |
| lat_mean/p50/p95/p99/p999_us | µs | binary (offline) | percentiles over per-query latency |
| io_reqs_mean | count | binary | mean read operations submitted per query |
| io_pages_mean | count(4 KiB) | binary | mean 4 KiB pages read per query (= io_bytes / 4096) |
| io_bytes_mean | bytes | binary | mean bytes read per query (aligned/actual) |
| io_pages_p99, io_pages_max | count | binary (offline) | tail of per-query pages |
| io_bytes_p99, io_bytes_max | bytes | binary (offline) | tail of per-query bytes |
| eff_bw_mbps | MB/s (1e6) | derived | total bytes / wall seconds |
| eff_iops | reads/s | derived | total read operations / wall seconds |
| cpu_util_pct | % | driver | `/usr/bin/time -v` "Percent of CPU" (>100% when multithreaded) |
| peak_rss_mb | MiB | driver | `/usr/bin/time -v` "Maximum resident set size" |
| resident_index_mb | MiB | driver | in-RAM index footprint |
| index_ssd_bytes | bytes | driver | size of the on-SSD index file(s) |

`p999` needs at least 1000 queries to mean anything.

## 2. Build manifest CSV — one row per index

```
index_id,system,dataset,build_params,build_s,build_peak_rss_mb,index_mem_bytes,index_ssd_bytes
```

| column | unit | definition |
|---|---|---|
| index_id | str | unique index identifier (the index directory name) |
| system, dataset | str | as above |
| build_params | str | `;`-separated knobs, no commas (e.g. `C=...;B=9;memdim=...;metric=l2`, or `store=raw` in place of `B=` for a raw-store index) |
| build_s | s | build wall time (note in build_params when clustering is shared across variants) |
| build_peak_rss_mb | MiB | peak RSS during the build |
| index_mem_bytes | bytes | in-RAM portion of the index |
| index_ssd_bytes | bytes | on-SSD portion |

## 3. Who fills what

- **The search binary** writes every query-CSV column except the four driver
  ones, which it leaves blank.
- **The driver** (`scripts/main_exp/`) fills `cpu_util_pct` and `peak_rss_mb`
  from `/usr/bin/time -v`, and `resident_index_mb` and `index_ssd_bytes` from
  `stat`, matching rows by `run_id`.
- **Offline** means computed after timing stops: recall, and the latency and IO
  percentiles, all from samples the engine already had.

## 4. How the timing stays honest

Two rules keep the measured QPS from being an artifact of the measurement:

- **No clock is read inside any loop.** The timed segment contains three
  timers per query — one around the whole query, one around the coarse-quantizer
  step, one around the drain phase — and none of them is inside the cluster
  loop, the batch loop or the drain loop. The per-query counters are
  ones the engine maintains anyway (an increment of an already-loaded value);
  they are pooled after timing ends.
- Recall is computed outside the timed segment.

This matters more than it looks. The engine prunes pending SSD reads against a
threshold derived from whatever has already been re-ranked, so how much compute
sits between a submit and the next poll decides which pages are read at all.
Instrumentation inside that window does not just add overhead — it changes the
result. Clock reads per probed batch would therefore move recall, not only
throughput.

The schema records no device-capability constants, so it needs no per-device
calibration and moves between hosts unchanged. IO efficiency is reported
absolutely (`eff_bw_mbps`, `eff_iops`); normalising against a device ceiling is
left to whoever wants it.
