#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Guards the pure decisions in verify-gpu-health-monitors.sh.
# Hermetic: sources the script and calls its pure functions, never a cluster.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the subject SCRIPT_DIR-relative so this exercises the file in THIS
# worktree, never a deployed copy.
# shellcheck source=./verify-gpu-health-monitors.sh
source "${SCRIPT_DIR}/verify-gpu-health-monitors.sh"

fail=0
check() {
    local desc="$1" want="$2" got="$3"
    if [[ "$want" != "$got" ]]; then
        echo "FAIL: ${desc}: want '${want}', got '${got}'" >&2
        fail=1
    else
        echo "ok: ${desc}"
    fi
}

# --- deriving the label from the pin ---------------------------------------
#
# The lane derives the expected dcgm.version from its own image reference using
# the labeler's rule, so a pin the labeler cannot parse fails here rather than
# as an absent DaemonSet an hour later.
check "a tag+digest reference yields the major" "4.x" \
    "$(expected_dcgm_major 'nvcr.io/nvidia/cloud-native/dcgm:4.6.0-1-ubuntu24.04@sha256:aaaa')"
check "a plain tag yields the major" "3.x" \
    "$(expected_dcgm_major 'nvcr.io/nvidia/cloud-native/dcgm:3.3.9-1-ubuntu22.04')"

# THE REGRESSION THIS EXISTS FOR. Pinning by bare digest, the way every other
# image in this lane is pinned, removes the tag the labeler parses. Nothing
# errors: the label is never written and the monitor never schedules.
out="$(expected_dcgm_major 'nvcr.io/nvidia/cloud-native/dcgm@sha256:aaaa' 2>/dev/null)"; rc=$?
check "a bare digest reference returns empty stdout" "" "${out}"
check "a bare digest reference fails closed" "1" "${rc}"

out="$(expected_dcgm_major 'nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3' 2>/dev/null)"; rc=$?
check "the dcgm-exporter image is not mistaken for a host engine" "1" "${rc}"

out="$(expected_dcgm_major 'nvcr.io/nvidia/cloud-native/dcgm:9.0.0' 2>/dev/null)"; rc=$?
check "an unrenderable major fails closed" "1" "${rc}"

# --- the mutually exclusive pair -------------------------------------------
check "4.x selects the 4.x DaemonSet" "gpu-health-monitor-dcgm-4.x" "$(monitor_daemonset 4.x)"
check "3.x selects the 3.x DaemonSet" "gpu-health-monitor-dcgm-3.x" "$(monitor_daemonset 3.x)"
check "4.x's sibling is 3.x" "gpu-health-monitor-dcgm-3.x" "$(monitor_sibling 4.x)"
check "3.x's sibling is 4.x" "gpu-health-monitor-dcgm-4.x" "$(monitor_sibling 3.x)"
monitor_daemonset 5.x >/dev/null 2>&1; check "an unknown version fails closed" "1" "$?"

# --- readiness, and the vacuous pass it must reject ------------------------
#
# desired=0/ready=0 satisfies "ready equals desired" while proving the monitor
# never ran. That is the exact state this lane exists to catch, so it must read
# as NOT ready.
daemonset_is_ready 1 1; check "one of one is ready" "0" "$?"
daemonset_is_ready 4 4; check "four of four is ready" "0" "$?"
daemonset_is_ready 0 0; check "zero of zero is NOT ready (the vacuous pass)" "1" "$?"
daemonset_is_ready 4 3; check "a partial rollout is not ready" "1" "$?"
daemonset_is_ready 0 1; check "ready exceeding desired is not ready" "1" "$?"
daemonset_is_ready "" ""; check "empty fields are not ready" "1" "$?"
daemonset_is_ready notanumber 1; check "a non-numeric field fails closed" "1" "$?"

# --- connection evidence, and the Ready monitor that never connected -------
#
# At NVSentinel v1.25.0 the monitor's readiness probe is GET /metrics, which
# answers as soon as its HTTP server starts, and the watch loop marks itself
# alive before it tries DCGM and retries on failure. So a monitor that never
# reaches a host engine is still Ready. What only a connected monitor emits is
# the "dcgm gpu_id are [...]" line (dcgm_watcher/dcgm.py:1098), logged after it
# has connected, built a group of every supported GPU DCGM discovered, and set
# its health watches. Fixture lines follow the monitor's structlog JSON shape.
log_line() { printf '{"event": "%s", "level": "%s", "module": "gpu-health-monitor"}\n' "$2" "$1"; }
CONNECTED_LOG="$(
    log_line info "Successfully created DCGM handle to nvidia-dcgm.gpu-operator.svc:5555"
    log_line info "supported gpus are [0, 1, 2, 3, 4, 5, 6, 7]"
    log_line info "dcgm gpu_id are [0, 1, 2, 3, 4, 5, 6, 7]"
)"
UNCONNECTED_LOG="$(
    log_line info "DCGM probe watchdog enabled with a 30.0s deadline"
    log_line error "Error creating DCGM handle: Unable to connect to any DCGM address"
)"
check "a connected monitor reports the GPUs it watches" "8" "$(monitor_gpu_count "${CONNECTED_LOG}")"
out="$(monitor_gpu_count "${UNCONNECTED_LOG}")"; rc=$?
check "a monitor that never connected reports no count" "" "${out}"
check "and fails closed" "1" "${rc}"
check "an empty GPU list counts zero" "0" \
    "$(monitor_gpu_count "$(log_line info "dcgm gpu_id are []")")"
# Discovery alone is not the evidence: the group and its health watches are
# set up between "supported gpus are" and "dcgm gpu_id are", and either can
# still fail and roll back.
out="$(monitor_gpu_count "$(log_line info "supported gpus are [0, 1, 2, 3, 4, 5, 6, 7]")")"; rc=$?
check "discovery without the monitoring group is not evidence" "1" "${rc}"
# The monitor re-initialises after a connectivity failure, so the latest
# initialisation is the one that describes it now.
check "the latest initialisation wins" "0" \
    "$(monitor_gpu_count "${CONNECTED_LOG}
$(log_line info "dcgm gpu_id are []")")"

# --- the live verdict: driving main ----------------------------------------
#
# The cases above call pure helpers. The reads happen in main, and a read is
# where a Kubernetes API error can be mistaken for a verdict, so these cases
# drive main ITSELF with only kubectl replaced, one layer deep at the API
# boundary, as topology-golden_test.sh drives verify-topology.sh. sleep is
# replaced too, advancing SECONDS instead of the wall clock, so a polling case
# costs no real time.
#
# Stub state is named stub_* because bash scopes dynamically: a stub closing
# over a name main declares local (desired, ready, ...) would read main's value
# instead of the fixture.
#
# The monitor pods and their logs are global stub_* settings rather than more
# positional arguments, because every earlier case wants the same healthy
# default: one connected monitor per worker. A case changes one and resets it.
CLUSTER="testcluster"
MONITOR="gpu-health-monitor-dcgm-4.x"
SIBLING="gpu-health-monitor-dcgm-3.x"
ALL_WORKERS="node/testcluster-worker
node/testcluster-worker2
node/testcluster-worker3
node/testcluster-worker4"
ALL_MONITOR_PODS="testcluster-worker ${MONITOR}-a
testcluster-worker2 ${MONITOR}-b
testcluster-worker3 ${MONITOR}-c
testcluster-worker4 ${MONITOR}-d"
stub_defaults() {
    stub_monitor_pods="${ALL_MONITOR_PODS}"
    stub_monitor_log="${CONNECTED_LOG}"
    stub_odd_pod="" stub_odd_log=""
    stub_logs_rc=0
    stub_unconnected_log_reads=0
}
stub_defaults

# main_with <labelled-nodes> <desired> <ready> <sibling-desired> [<sibling-rc>]
#           [<unlabelled-reads>]
#
# Prints main's stdout and stderr, then a final "label reads: N" line; the exit
# code is main's. The first <unlabelled-reads> node reads return no node, which
# is a labeler that has not reconciled yet. An unstubbed read fails with 97 so
# a case can never pass on a call this harness did not expect.
main_with() {
    local stub_labelled="$1" stub_desired="$2" stub_ready="$3"
    local stub_sibling="$4" stub_sibling_rc="${5:-0}" stub_unlabelled_reads="${6:-0}"
    local stub_reads_log stub_log_reads_log stub_rc
    stub_reads_log="$(mktemp)"
    stub_log_reads_log="$(mktemp)"
    (
        kubectl() {
            case "$*" in
                *" get nodes "*)
                    # A file, not a variable: each read runs in its own
                    # command substitution, so a counter would not persist.
                    echo read >>"${stub_reads_log}"
                    if (($(wc -l <"${stub_reads_log}") > stub_unlabelled_reads)); then
                        [[ -z "${stub_labelled}" ]] || printf '%s\n' "${stub_labelled}"
                    fi
                    ;;
                *" get daemonset ${MONITOR} "*desiredNumberScheduled*) printf '%s' "${stub_desired}" ;;
                *" get daemonset ${MONITOR} "*numberReady*) printf '%s' "${stub_ready}" ;;
                *" get daemonset ${SIBLING} "*desiredNumberScheduled*)
                    printf '%s' "${stub_sibling}"
                    return "${stub_sibling_rc}"
                    ;;
                *" get pods "*"${MONITOR}"*) printf '%s\n' "${stub_monitor_pods}" ;;
                *" logs "*" -c gpu-health-monitor"*)
                    # The first <stub_unconnected_log_reads> reads see a
                    # monitor that has not connected yet.
                    echo read >>"${stub_log_reads_log}"
                    if (($(wc -l <"${stub_log_reads_log}") <= stub_unconnected_log_reads)); then
                        printf '%s\n' "${UNCONNECTED_LOG}"
                    elif [[ -n "${stub_odd_pod}" && "$*" == *" ${stub_odd_pod} "* ]]; then
                        printf '%s\n' "${stub_odd_log}"
                    else
                        printf '%s\n' "${stub_monitor_log}"
                    fi
                    return "${stub_logs_rc}"
                    ;;
                *)
                    echo "unstubbed kubectl call: $*" >&2
                    return 97
                    ;;
            esac
        }
        sleep() { SECONDS=$((SECONDS + ${1%s})); }
        MONITOR_TIMEOUT=30 main "${CLUSTER}" 2>&1
    )
    stub_rc=$?
    echo "label reads: $(wc -l <"${stub_reads_log}" | tr -d ' ')"
    rm -f "${stub_reads_log}" "${stub_log_reads_log}"
    return "${stub_rc}"
}

# label_reads <main_with-output>
label_reads() {
    sed -n 's/^label reads: //p' <<<"$1"
}

# A FAILED SIBLING LOOKUP IS NOT A ZERO. The sibling is asserted at 0, and an
# empty read used to default to 0, so an API error, a missing RBAC verb or a
# renamed DaemonSet all printed "correctly inactive" and passed the gate.
out="$(main_with "${ALL_WORKERS}" 4 4 "" 1)"; rc=$?
check "a failed sibling lookup fails the gate" "1" "${rc}"
check "and says the lookup failed rather than calling it inactive" "1" \
    "$(grep -cxF "error: could not read ${SIBLING}; a failed lookup does not prove it inactive" <<<"${out}")"
out="$(main_with "${ALL_WORKERS}" 4 4 "" 0)"; rc=$?
check "an empty sibling read fails the gate" "1" "${rc}"
check "and names the value it could not count" "1" \
    "$(grep -cxF "error: ${SIBLING} reported desiredNumberScheduled '', not a count" <<<"${out}")"
# The two controls: a successful read of 0 is the inactive state, and one
# above 0 is the mutual-exclusion failure.
out="$(main_with "${ALL_WORKERS}" 4 4 0)"; rc=$?
check "a sibling read of 0 passes" "0" "${rc}"
check "and reports the sibling inactive" "1" "$(grep -cxF "${SIBLING} is correctly inactive" <<<"${out}")"
out="$(main_with "${ALL_WORKERS}" 4 4 2)"; rc=$?
check "a scheduled sibling fails the gate" "1" "${rc}"

# THE LABEL IS WRITTEN ASYNCHRONOUSLY. The labeler stamps dcgm.version from its
# own event handlers once it runs and its caches sync, and nothing before this
# step waits for that, so the first read can legitimately find no node. The
# gate polls within MONITOR_TIMEOUT instead of failing on that read.
out="$(main_with "${ALL_WORKERS}" 4 4 0 0 2)"; rc=$?
check "labels that appear on the third read pass" "0" "${rc}"
check "and polling stops once they do" "3" "$(label_reads "${out}")"
# The label wait and the DaemonSet wait share one budget. Labels that arrive at
# the deadline must still get the DaemonSet read once, not be failed because
# the first wait spent the budget.
out="$(main_with "${ALL_WORKERS}" 4 4 0 0 3)"; rc=$?
check "labels arriving at the deadline still reach the DaemonSet check" "0" "${rc}"
# A label that never appears still fails, at the deadline and not before it.
out="$(main_with "" 4 4 0)"; rc=$?
check "a label that never appears fails the gate" "1" "${rc}"
check "and names the label, the budget and every worker" "1" \
    "$(grep -cxF "error: 4 of 4 worker(s) lack nvsentinel.dgxc.nvidia.com/dcgm.version=4.x after 30s: testcluster-worker testcluster-worker2 testcluster-worker3 testcluster-worker4" <<<"${out}")"
reads="$(label_reads "${out}")"
check "after polling, not on the first read" "polled" \
    "$( ((reads > 1)) && echo polled || echo "read ${reads} time(s)")"

# EVERY WORKER, NOT ANY NODE. The lane runs a host engine on all four workers,
# so one labelled node and a monitor DaemonSet at 1/1 used to pass while three
# workers had no monitor at all.
out="$(main_with "node/testcluster-worker" 1 1 0)"; rc=$?
check "one labelled worker of four fails the gate" "1" "${rc}"
check "and names the three it is missing" "1" \
    "$(grep -cxF "error: 3 of 4 worker(s) lack nvsentinel.dgxc.nvidia.com/dcgm.version=4.x after 30s: testcluster-worker2 testcluster-worker3 testcluster-worker4" <<<"${out}")"
# By name, not by count: four labelled nodes are not four labelled workers when
# one of them is the control plane, which must stay GPU-free.
out="$(main_with "node/testcluster-control-plane
node/testcluster-worker
node/testcluster-worker2
node/testcluster-worker3" 4 4 0)"; rc=$?
check "a labelled control plane does not stand in for a missing worker" "1" "${rc}"
check "and the missing worker is the one named" "1" \
    "$(grep -cxF "error: 1 of 4 worker(s) lack nvsentinel.dgxc.nvidia.com/dcgm.version=4.x after 30s: testcluster-worker4" <<<"${out}")"
# And the DaemonSet must cover them: every worker labelled with the monitor
# scheduled on one of them is still three workers without one.
out="$(main_with "${ALL_WORKERS}" 1 1 0)"; rc=$?
check "a monitor on one of four labelled workers fails the gate" "1" "${rc}"
check "and reports the coverage it found" "1" \
    "$(grep -cxF "error: ${MONITOR} is not ready on all 4 worker(s) after 30s (desired=1 ready=1)" <<<"${out}")"

# READY IS NOT CONNECTED. Every monitor pod Ready on every worker, and none of
# them ever reached a host engine: `-b ALL` dropped, a Service mismatch, or
# nv-hostengine failing on the mocked NVML all look like this. The gate must
# fail here, by worker, rather than pass on numberReady.
stub_monitor_log="${UNCONNECTED_LOG}"
out="$(main_with "${ALL_WORKERS}" 4 4 0)"; rc=$?
stub_defaults
check "a Ready monitor that never connected fails the gate" "1" "${rc}"
check "and names every worker without connection evidence" "1" \
    "$(grep -cxF "error: 4 of 4 monitor(s) show no DCGM connection with 8 GPU(s) after 30s: testcluster-worker (no 'dcgm gpu_id are' line) testcluster-worker2 (no 'dcgm gpu_id are' line) testcluster-worker3 (no 'dcgm gpu_id are' line) testcluster-worker4 (no 'dcgm gpu_id are' line)" <<<"${out}")"
# One monitor connected to an engine that discovered fewer GPUs than the
# worker advertises: the host engine and the device plugin disagree.
stub_odd_pod="${MONITOR}-c" stub_odd_log="$(log_line info "dcgm gpu_id are [0, 1, 2]")"
out="$(main_with "${ALL_WORKERS}" 4 4 0)"; rc=$?
stub_defaults
check "a monitor watching the wrong number of GPUs fails the gate" "1" "${rc}"
check "and reports the count it saw on that worker" "1" \
    "$(grep -cxF "error: 1 of 4 monitor(s) show no DCGM connection with 8 GPU(s) after 30s: testcluster-worker3 (3 GPU(s))" <<<"${out}")"
# A worker whose monitor pod is missing from the listing has no evidence,
# whatever the DaemonSet counts say.
stub_monitor_pods="$(grep -v '^testcluster-worker4 ' <<<"${ALL_MONITOR_PODS}")"
out="$(main_with "${ALL_WORKERS}" 4 4 0)"; rc=$?
stub_defaults
check "a worker with no monitor pod fails the gate" "1" "${rc}"
check "and is named as having no pod" "1" \
    "$(grep -cxF "error: 1 of 4 monitor(s) show no DCGM connection with 8 GPU(s) after 30s: testcluster-worker4 (no monitor pod)" <<<"${out}")"
# A failed log read is not evidence either way, so it cannot pass.
stub_logs_rc=1
out="$(main_with "${ALL_WORKERS}" 4 4 0)"; rc=$?
stub_defaults
check "failed log reads fail the gate" "1" "${rc}"
check "and say the logs could not be read" "1" \
    "$(grep -cF "testcluster-worker (logs unreadable)" <<<"${out}")"
# The monitor waits one poll interval before its first connect, so the line
# can trail readiness. The gate polls for it inside the same budget.
stub_unconnected_log_reads=4
out="$(main_with "${ALL_WORKERS}" 4 4 0)"; rc=$?
stub_defaults
check "connection evidence that appears on the next poll passes" "0" "${rc}"
check "and is reported per worker" "1" \
    "$(grep -cxF "all 4 monitor(s) connected to DCGM and watch 8 GPU(s) each" <<<"${out}")"

# --- the lane must actually run this, and gate on it -----------------------
#
# A guard nobody runs is not a guard. The same reasoning topology-golden_test.sh
# applies to verify-topology.sh applies here: a step whose failure nothing
# depends on lets the run publish signed evidence claiming GPU health coverage
# the lane never demonstrated.
WORKFLOW="${SCRIPT_DIR}/../../../.github/workflows/uat-kind-sim.yaml"
check "the CI lane calls verify-gpu-health-monitors.sh" "1" \
    "$(grep -c 'tests/uat/kind/verify-gpu-health-monitors\.sh' "${WORKFLOW}" | tr -d ' ')"
check "the step has an id the conformance job can depend on" "1" \
    "$(grep -cE '^        id: gpu_health$' "${WORKFLOW}" | tr -d ' ')"
check "conformance is gated on this step" "1" \
    "$(grep -c "steps.gpu_health.outcome == 'success'" "${WORKFLOW}" | tr -d ' ')"
# And it must not have displaced the gate that was already there.
check "conformance is still gated on the topology step" "1" \
    "$(grep -c "steps.topology.outcome == 'success' && steps.gpu_health.outcome == 'success'" "${WORKFLOW}" | tr -d ' ')"

# THE STEP MUST OUTLIVE THE SCRIPT'S OWN DEADLINE. A step timeout that fires
# first kills the verifier before it prints which wait failed, and the run
# shows a cancelled step instead of a cause. Past MONITOR_TIMEOUT the script
# can still sleep one interval and finish the label read it then starts, make
# the DaemonSet's two reads that a late label still earns, list the monitor
# pods, read one log per worker, and make the sibling read: five kubectl calls
# plus one per worker, each of up to KUBECTL_TIMEOUT.
step_timeout_minutes() {
    awk -v id="$1" '
        /^      - name:/ { in_step = 0 }
        $0 == "        id: " id { in_step = 1 }
        in_step && /^        timeout-minutes:/ { print $2; exit }
    ' "${WORKFLOW}"
}
worst_case=$((MONITOR_TIMEOUT + MONITOR_INTERVAL + (5 + $(worker_indices | wc -l)) * ${KUBECTL_TIMEOUT%s}))
step_minutes="$(step_timeout_minutes gpu_health)"
check "the step timeout outlives the verifier's worst case of ${worst_case}s" "outlives" \
    "$([[ "${step_minutes}" =~ ^[0-9]+$ ]] && ((step_minutes * 60 > worst_case)) &&
        echo outlives || echo "timeout-minutes=${step_minutes:-unset}")"

exit "${fail}"
