export PATH="${PATH:+${PATH}:}/usr/local/nvidia/bin"; export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/nvidia/lib64"; 
printf 'NODE=%s\n' "${SLURMD_NODENAME:-}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
if out="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>&1)"; then rc=0; else rc=$?; fi
printf 'NVSMI_RC=%s\n' "$rc"
printf '%s\n' "$out" | sed 's/^/NVSMI_LINE=/'
command -v perl >/dev/null 2>&1 || { printf 'PROBE_ERROR=perl not found\n'; exit 1; }
devs="$(find /dev -maxdepth 1 -type c -name 'nvidia[0-9]*' | sort)" || { printf 'PROBE_ERROR=find failed\n'; exit 1; }
if [ -n "$devs" ]; then n="$(printf '%s\n' "$devs" | wc -l)"; else n=0; fi
printf 'DEVICES_LISTED=%s\n' "$n"
if [ -n "$devs" ]; then
  printf '%s\n' "$devs" | perl -ne 'chomp; if (sysopen(my $fh, $_, 0)) { close($fh); print "DEV=$_ 0\n" } else { print "DEV=$_ ", $!+0, "\n" }' || { printf 'PROBE_ERROR=perl failed\n'; exit 1; }
fi
printf 'DONE=1\n'
