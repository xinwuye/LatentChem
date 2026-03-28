#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

slurm_bin="/opt/gridview/slurm/bin"
remote_repo="/public/home/xinwuye/Bio-LatentCOT"
log_dir="${remote_repo}/slurm_logs"
state_file="${log_dir}/pudong_watch_active_14b_jobs.state"
pid_file="${log_dir}/pudong_watch_active_14b_jobs.pid"
poll_seconds="${POLL_SECONDS:-120}"

mkdir -p "${log_dir}"

if [[ -f "${pid_file}" ]]; then
  existing_pid="$(cat "${pid_file}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Watchdog already running with pid ${existing_pid}" >&2
    exit 1
  fi
fi

echo "$$" > "${pid_file}"
trap 'rm -f "${pid_file}"' EXIT

seed2029_job_id="${SEED2029_JOB_ID:-28295}"
seed2029_final_state="${SEED2029_FINAL_STATE:-}"

latentchem14b_job_id="${LATENTCHEM14B_JOB_ID:-28731}"
latentchem14b_oom_count="${LATENTCHEM14B_OOM_COUNT:-0}"
latentchem14b_final_state="${LATENTCHEM14B_FINAL_STATE:-}"

latentchem14b_stage124_job_id="${LATENTCHEM14B_STAGE124_JOB_ID:-29161}"
latentchem14b_stage124_oom_count="${LATENTCHEM14B_STAGE124_OOM_COUNT:-0}"
latentchem14b_stage124_final_state="${LATENTCHEM14B_STAGE124_FINAL_STATE:-}"

latentchem14b_stage4_dir="${remote_repo}/code_train_sft/outputs/stage4-latentchem14b/stage4"
latentchem14b_stage124_stage4_dir="${remote_repo}/code_train_sft/outputs/stage4-stage124-14b/stage4"

latentchem14b_resume_sbatch="${remote_repo}/scripts/training/pudong_latentchem14b_stage4_resume.sbatch"
latentchem14b_resume_lowmem_sbatch="${remote_repo}/scripts/training/pudong_latentchem14b_stage4_resume_lowmem.sbatch"
latentchem14b_stage124_resume_sbatch="${remote_repo}/scripts/training/pudong_latentchem14b_stage124_stage4_resume.sbatch"
latentchem14b_stage124_resume_lowmem_sbatch="${remote_repo}/scripts/training/pudong_latentchem14b_stage124_stage4_resume_lowmem.sbatch"

for required in \
  "${latentchem14b_resume_sbatch}" \
  "${latentchem14b_resume_lowmem_sbatch}" \
  "${latentchem14b_stage124_resume_sbatch}" \
  "${latentchem14b_stage124_resume_lowmem_sbatch}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required watchdog dependency: ${required}" >&2
    exit 1
  fi
done

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

save_state() {
  cat > "${state_file}" <<EOF
seed2029_job_id=${seed2029_job_id}
seed2029_final_state=${seed2029_final_state}
latentchem14b_job_id=${latentchem14b_job_id}
latentchem14b_oom_count=${latentchem14b_oom_count}
latentchem14b_final_state=${latentchem14b_final_state}
latentchem14b_stage124_job_id=${latentchem14b_stage124_job_id}
latentchem14b_stage124_oom_count=${latentchem14b_stage124_oom_count}
latentchem14b_stage124_final_state=${latentchem14b_stage124_final_state}
EOF
}

load_state() {
  if [[ -f "${state_file}" ]]; then
    # shellcheck disable=SC1090
    source "${state_file}"
  fi
}

job_field() {
  local job_id="$1"
  local field="$2"
  "${slurm_bin}/sacct" -j "${job_id}" --format="JobID,${field}" -P |
    awk -F'|' -v jid="${job_id}" '$1 == jid {print $2; exit}'
}

job_state() {
  job_field "$1" "State"
}

job_name() {
  job_field "$1" "JobName"
}

job_log_path() {
  local job_id="$1"
  local name
  name="$(job_name "${job_id}")"
  if [[ -z "${name}" ]]; then
    return 1
  fi
  printf '%s/%s-%s.err\n' "${log_dir}" "${name}" "${job_id}"
}

has_checkpoint() {
  local stage4_dir="$1"
  compgen -G "${stage4_dir}/checkpoint-*" > /dev/null
}

is_oom_log() {
  local log_path="$1"
  [[ -f "${log_path}" ]] || return 1
  grep -Eq 'OutOfMemoryError|CUDA out of memory|out of memory|Tried to allocate' "${log_path}"
}

submit_resume_job() {
  local sbatch_script="$1"
  local output
  output="$("${slurm_bin}/sbatch" "${sbatch_script}")"
  echo "${output##* }"
}

handle_terminal_non14b() {
  local name="$1"
  local job_id="$2"
  local state="$3"
  log "ALERT: ${name} job ${job_id} entered terminal state ${state}"
}

handle_oom_resubmit() {
  local logical_name="$1"
  local stage4_dir="$2"
  local sbatch_normal="$3"
  local sbatch_lowmem="$4"
  local current_job_id="$5"
  local current_count="$6"
  local log_path="$7"
  local next_count next_script next_job_id

  if ! has_checkpoint "${stage4_dir}"; then
    log "ALERT: ${logical_name} job ${current_job_id} OOMed but no checkpoint exists in ${stage4_dir}; cannot resume with latest"
    echo "NO_RESUBMIT"
    return 0
  fi

  next_count=$((current_count + 1))
  # Assumption: "short-time multiple OOMs" means the second detected OOM for the
  # same logical task while this watchdog is running. From the second OOM onward,
  # switch to the low-memory submission variant.
  if (( next_count >= 2 )); then
    next_script="${sbatch_lowmem}"
  else
    next_script="${sbatch_normal}"
  fi

  next_job_id="$(submit_resume_job "${next_script}")"
  log "ALERT: ${logical_name} job ${current_job_id} OOMed; resubmitted job ${next_job_id} via $(basename "${next_script}") using log ${log_path}"
  printf '%s|%s\n' "${next_job_id}" "${next_count}"
}

check_seed2029() {
  if [[ -n "${seed2029_final_state}" ]]; then
    return 0
  fi
  local state
  state="$(job_state "${seed2029_job_id}")"
  case "${state}" in
    RUNNING|PENDING|CONFIGURING|COMPLETING)
      return 0
      ;;
    COMPLETED)
      seed2029_final_state="${state}"
      log "latentchem_seed2029 job ${seed2029_job_id} completed"
      ;;
    *)
      seed2029_final_state="${state:-UNKNOWN}"
      handle_terminal_non14b "latentchem_seed2029" "${seed2029_job_id}" "${seed2029_final_state}"
      ;;
  esac
}

check_latentchem14b() {
  if [[ -n "${latentchem14b_final_state}" ]]; then
    return 0
  fi
  local state log_path result new_job_id new_count
  state="$(job_state "${latentchem14b_job_id}")"
  case "${state}" in
    RUNNING|PENDING|CONFIGURING|COMPLETING)
      return 0
      ;;
    COMPLETED)
      latentchem14b_final_state="${state}"
      log "latentchem14b job ${latentchem14b_job_id} completed"
      ;;
    *)
      log_path="$(job_log_path "${latentchem14b_job_id}" || true)"
      if [[ -n "${log_path}" ]] && is_oom_log "${log_path}"; then
        result="$(handle_oom_resubmit \
          "latentchem14b" \
          "${latentchem14b_stage4_dir}" \
          "${latentchem14b_resume_sbatch}" \
          "${latentchem14b_resume_lowmem_sbatch}" \
          "${latentchem14b_job_id}" \
          "${latentchem14b_oom_count}" \
          "${log_path}")"
        if [[ "${result}" == "NO_RESUBMIT" ]]; then
          latentchem14b_final_state="${state:-OOM_NO_CHECKPOINT}"
        else
          new_job_id="${result%%|*}"
          new_count="${result##*|}"
          latentchem14b_job_id="${new_job_id}"
          latentchem14b_oom_count="${new_count}"
        fi
      else
        latentchem14b_final_state="${state:-UNKNOWN}"
        log "ALERT: latentchem14b job ${latentchem14b_job_id} entered terminal state ${latentchem14b_final_state} without detected OOM"
      fi
      ;;
  esac
}

check_latentchem14b_stage124() {
  if [[ -n "${latentchem14b_stage124_final_state}" ]]; then
    return 0
  fi
  local state log_path result new_job_id new_count
  state="$(job_state "${latentchem14b_stage124_job_id}")"
  case "${state}" in
    RUNNING|PENDING|CONFIGURING|COMPLETING)
      return 0
      ;;
    COMPLETED)
      latentchem14b_stage124_final_state="${state}"
      log "latentchem14b_stage124 job ${latentchem14b_stage124_job_id} completed"
      ;;
    *)
      log_path="$(job_log_path "${latentchem14b_stage124_job_id}" || true)"
      if [[ -n "${log_path}" ]] && is_oom_log "${log_path}"; then
        result="$(handle_oom_resubmit \
          "latentchem14b_stage124" \
          "${latentchem14b_stage124_stage4_dir}" \
          "${latentchem14b_stage124_resume_sbatch}" \
          "${latentchem14b_stage124_resume_lowmem_sbatch}" \
          "${latentchem14b_stage124_job_id}" \
          "${latentchem14b_stage124_oom_count}" \
          "${log_path}")"
        if [[ "${result}" == "NO_RESUBMIT" ]]; then
          latentchem14b_stage124_final_state="${state:-OOM_NO_CHECKPOINT}"
        else
          new_job_id="${result%%|*}"
          new_count="${result##*|}"
          latentchem14b_stage124_job_id="${new_job_id}"
          latentchem14b_stage124_oom_count="${new_count}"
        fi
      else
        latentchem14b_stage124_final_state="${state:-UNKNOWN}"
        log "ALERT: latentchem14b_stage124 job ${latentchem14b_stage124_job_id} entered terminal state ${latentchem14b_stage124_final_state} without detected OOM"
      fi
      ;;
  esac
}

load_state
save_state
log "Started watchdog: seed2029=${seed2029_job_id}, latentchem14b=${latentchem14b_job_id}, latentchem14b_stage124=${latentchem14b_stage124_job_id}, poll_seconds=${poll_seconds}"

while true; do
  check_seed2029
  check_latentchem14b
  check_latentchem14b_stage124
  save_state
  sleep "${poll_seconds}"
done
