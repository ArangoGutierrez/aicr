// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const testGPUUUID = "GPU-0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0"

// slurmGPUProbeStdout renders output in the shape slinkySlurmGPUProbeShell
// prints. errnos[i] is the open errno reported for /dev/nvidia<i>.
func slurmGPUProbeStdout(node string, nvsmiExit int, nvsmiLines []string, errnos []int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "NODE=%s\nCUDA_VISIBLE_DEVICES=0\nNVSMI_RC=%d\n", node, nvsmiExit)
	for _, line := range nvsmiLines {
		fmt.Fprintf(&b, "NVSMI_LINE=%s\n", line)
	}
	fmt.Fprintf(&b, "DEVICES_LISTED=%d\n", len(errnos))
	for i, errno := range errnos {
		fmt.Fprintf(&b, "DEV=/dev/nvidia%d %d\n", i, errno)
	}
	b.WriteString("DONE=1\n")
	return b.String()
}

// epermExcept returns the open errnos of an 8-GPU node: all EPERM (1) except
// index openIdx, which is 0 (opened). openIdx -1 leaves every device denied.
func epermExcept(openIdx int) []int {
	errnos := make([]int, 8)
	for i := range errnos {
		errnos[i] = 1
	}
	if openIdx >= 0 {
		errnos[openIdx] = 0
	}
	return errnos
}

func TestParseSlurmGPUProbe(t *testing.T) {
	twoDevices := []slurmGPUDeviceOpen{{path: "/dev/nvidia0", errno: 0}, {path: "/dev/nvidia1", errno: 1}}
	tests := []struct {
		name    string
		stdout  string
		want    slurmGPUProbe
		wantErr string
	}{
		{
			name:   "allocated job on a two-GPU node",
			stdout: slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, []int{0, 1}),
			want: slurmGPUProbe{node: "slinky-0", cudaVisibleDevices: "0", nvsmiSeen: true,
				gpuUUIDs: []string{testGPUUUID}, devicesListed: 2, listedSeen: true, devices: twoDevices, done: true},
		},
		{
			name:   "wc pads the device count",
			stdout: strings.Replace(slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, []int{0, 1}), "DEVICES_LISTED=2", "DEVICES_LISTED=       2", 1),
			want: slurmGPUProbe{node: "slinky-0", cudaVisibleDevices: "0", nvsmiSeen: true,
				gpuUUIDs: []string{testGPUUUID}, devicesListed: 2, listedSeen: true, devices: twoDevices, done: true},
		},
		{
			name: "slurm and prolog noise is ignored",
			stdout: "srun: job 42 queued and waiting for resources\nexport FOO=bar\n" +
				slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, []int{0, 1}),
			want: slurmGPUProbe{node: "slinky-0", cudaVisibleDevices: "0", nvsmiSeen: true,
				gpuUUIDs: []string{testGPUUUID}, devicesListed: 2, listedSeen: true, devices: twoDevices, done: true},
		},
		{
			name: "nvidia-smi warning line is not a GPU",
			stdout: slurmGPUProbeStdout("slinky-0", 0,
				[]string{"WARNING: infoROM is corrupted at gpu 0000:3B:00.0", testGPUUUID}, []int{0, 1}),
			want: slurmGPUProbe{node: "slinky-0", cudaVisibleDevices: "0", nvsmiSeen: true,
				gpuUUIDs: []string{testGPUUUID}, devicesListed: 2, listedSeen: true, devices: twoDevices, done: true},
		},
		{
			name:   "MIG lines are counted apart from GPUs",
			stdout: slurmGPUProbeStdout("slinky-0", 0, []string{"MIG-11111111-2222-3333-4444-555555555555"}, []int{0, 1}),
			want: slurmGPUProbe{node: "slinky-0", cudaVisibleDevices: "0", nvsmiSeen: true,
				migLines: 1, devicesListed: 2, listedSeen: true, devices: twoDevices, done: true},
		},
		{
			name:   "CRLF line endings",
			stdout: strings.ReplaceAll(slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, []int{0, 1}), "\n", "\r\n"),
			want: slurmGPUProbe{node: "slinky-0", cudaVisibleDevices: "0", nvsmiSeen: true,
				gpuUUIDs: []string{testGPUUUID}, devicesListed: 2, listedSeen: true, devices: twoDevices, done: true},
		},
		{
			name:    "repeated NODE means more than one task ran",
			stdout:  "NODE=slinky-0\nNODE=slinky-1\n",
			wantErr: "GPU probe output repeats NODE",
		},
		{
			name:    "non-integer device count",
			stdout:  "DEVICES_LISTED=two\n",
			wantErr: `GPU probe DEVICES_LISTED "two" is not a non-negative integer`,
		},
		{
			name:    "DEV line without errno",
			stdout:  "DEV=/dev/nvidia0\n",
			wantErr: `malformed GPU probe line "DEV=/dev/nvidia0"`,
		},
		{
			name:    "negative errno",
			stdout:  "DEV=/dev/nvidia0 -1\n",
			wantErr: `malformed GPU probe line "DEV=/dev/nvidia0 -1"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseSlurmGPUProbe(tt.stdout)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("error = %v, want nil", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parsed = %+v\nwant     %+v", got, tt.want)
			}
		})
	}
}

func TestEvaluateAllocatedGPUProbe(t *testing.T) {
	ok := slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, epermExcept(0))
	tests := []struct {
		name    string
		stdout  string
		wantErr string
	}{
		{name: "one GPU listed and exactly one device opens", stdout: ok},
		{
			name: "nvidia-smi warning plus one GPU",
			stdout: slurmGPUProbeStdout("slinky-0", 0,
				[]string{"WARNING: infoROM is corrupted at gpu 0000:3B:00.0", testGPUUUID}, epermExcept(0)),
		},
		{
			name:    "probe did not finish",
			stdout:  strings.Replace(ok, "DONE=1\n", "", 1),
			wantErr: "allocated job (--gpus=1): GPU probe output has no DONE=1 marker, so the probe did not finish",
		},
		{
			name:    "probe aborted",
			stdout:  ok + "PROBE_ERROR=perl not found\n",
			wantErr: "allocated job (--gpus=1): GPU probe aborted: perl not found",
		},
		{
			name:    "SLURMD_NODENAME unset",
			stdout:  slurmGPUProbeStdout("", 0, []string{testGPUUUID}, epermExcept(0)),
			wantErr: `allocated job (--gpus=1): GPU probe reported Slurm node "", want a single node name`,
		},
		{
			name:    "hostlist expression as node",
			stdout:  slurmGPUProbeStdout("slinky-[0-1]", 0, []string{testGPUUUID}, epermExcept(0)),
			wantErr: `allocated job (--gpus=1): GPU probe reported Slurm node "slinky-[0-1]", want a single node name`,
		},
		{
			name:    "perl dropped a device",
			stdout:  strings.Replace(ok, "DEV=/dev/nvidia7 1\n", "", 1),
			wantErr: "allocated job (--gpus=1): GPU probe reported 7 of 8 listed GPU device nodes",
		},
		{
			name:    "nvidia-smi not found",
			stdout:  slurmGPUProbeStdout("slinky-0", 127, []string{"/bin/sh: 5: nvidia-smi: not found"}, epermExcept(0)),
			wantErr: "allocated job (--gpus=1): nvidia-smi exited 127, so the job cannot use its allocated GPU",
		},
		{
			name:    "no GPU listed",
			stdout:  slurmGPUProbeStdout("slinky-0", 0, nil, epermExcept(0)),
			wantErr: "allocated job (--gpus=1): nvidia-smi listed 0 GPUs, want exactly 1",
		},
		{
			name:    "two GPUs listed",
			stdout:  slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID, "GPU-ffffffff-0000-1111-2222-333333333333"}, epermExcept(0)),
			wantErr: "allocated job (--gpus=1): nvidia-smi listed 2 GPUs, want exactly 1",
		},
		{
			name:    "MIG device listed",
			stdout:  slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID, "MIG-11111111-2222-3333-4444-555555555555"}, epermExcept(0)),
			wantErr: "allocated job (--gpus=1): nvidia-smi listed 1 MIG device(s); this check expects whole GPUs",
		},
		{
			name:    "no confinement: every device opens",
			stdout:  slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, []int{0, 0, 0, 0, 0, 0, 0, 0}),
			wantErr: "allocated job (--gpus=1): opened 8 of 8 GPU device nodes, want exactly 1",
		},
		{
			name:    "allocation reached no device",
			stdout:  slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, epermExcept(-1)),
			wantErr: "allocated job (--gpus=1): opened 0 of 8 GPU device nodes, want exactly 1",
		},
		{
			name:    "ENXIO on another minor",
			stdout:  slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, []int{0, 6, 1, 1, 1, 1, 1, 1}),
			wantErr: "allocated job (--gpus=1): unexpected errno opening GPU device nodes (/dev/nvidia1 errno 6); only EPERM counts as denied",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe, err := parseSlurmGPUProbe(tt.stdout)
			if err != nil {
				t.Fatalf("parse error = %v", err)
			}
			err = evaluateAllocatedGPUProbe(probe)
			if tt.wantErr == "" && err != nil {
				t.Fatalf("error = %v, want nil", err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestEvaluateUnallocatedGPUProbe(t *testing.T) {
	tests := []struct {
		name    string
		stdout  string
		wantErr string
	}{
		{
			name:   "every device listed and refused with EPERM",
			stdout: slurmGPUProbeStdout("slinky-0", 6, []string{"No devices were found"}, epermExcept(-1)),
		},
		{
			name:   "no device nodes listed at all",
			stdout: slurmGPUProbeStdout("slinky-0", 127, []string{"/bin/sh: 5: nvidia-smi: not found"}, nil),
		},
		{
			name:    "ran on another node",
			stdout:  slurmGPUProbeStdout("slinky-1", 6, nil, epermExcept(-1)),
			wantErr: `unallocated job (no GPU request) ran on Slurm node "slinky-1", want "slinky-0" (the allocated job's node)`,
		},
		{
			name:    "isolation broken: a device opens",
			stdout:  slurmGPUProbeStdout("slinky-0", 6, nil, epermExcept(2)),
			wantErr: "unallocated job (no GPU request) opened 1 GPU device node(s) without a GPU allocation: Slurm GPU isolation is broken",
		},
		{
			name:    "ENXIO is inconclusive",
			stdout:  slurmGPUProbeStdout("slinky-0", 6, nil, []int{1, 6, 1, 1}),
			wantErr: "unallocated job (no GPU request): unexpected errno opening GPU device nodes (/dev/nvidia1 errno 6); only EPERM counts as denied, so isolation is unproven",
		},
		{
			name:    "EACCES is inconclusive until a live run says otherwise",
			stdout:  slurmGPUProbeStdout("slinky-0", 6, nil, []int{13, 13}),
			wantErr: "(/dev/nvidia0 errno 13, /dev/nvidia1 errno 13)",
		},
		{
			name:    "NVML lists a GPU without an allocation",
			stdout:  slurmGPUProbeStdout("slinky-0", 0, []string{testGPUUUID}, epermExcept(-1)),
			wantErr: "unallocated job (no GPU request): nvidia-smi listed 1 GPU(s) without a GPU allocation: Slurm GPU isolation is broken",
		},
		{
			name:    "probe did not finish",
			stdout:  strings.Replace(slurmGPUProbeStdout("slinky-0", 6, nil, epermExcept(-1)), "DONE=1\n", "", 1),
			wantErr: "unallocated job (no GPU request): GPU probe output has no DONE=1 marker",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe, err := parseSlurmGPUProbe(tt.stdout)
			if err != nil {
				t.Fatalf("parse error = %v", err)
			}
			err = evaluateUnallocatedGPUProbe(probe, "slinky-0")
			if tt.wantErr == "" && err != nil {
				t.Fatalf("error = %v, want nil", err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

// The shipped probe must be byte-identical to the copy the PR's offline
// Docker smoke runs (Task 6); editing one without the other fails here.
func TestSlinkySlurmGPUProbeShellMatchesOfflineVerifiedCopy(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "slinky_slurm_gpu_probe.sh"))
	if err != nil {
		t.Fatalf("read testdata copy: %v", err)
	}
	if string(want) != slinkySlurmGPUProbeShell {
		t.Fatalf("slinkySlurmGPUProbeShell differs from testdata/slinky_slurm_gpu_probe.sh;\nconst:\n%s\ntestdata:\n%s",
			slinkySlurmGPUProbeShell, want)
	}
}
