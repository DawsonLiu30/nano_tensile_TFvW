#!/usr/bin/env bash
set -u

RUNROOT="${HOME}/qe_vaclm_local_20260630"
cd "$RUNROOT" || {
  echo "pristine_done=0"
  echo "vacancy_done=0"
  echo "error=missing_runroot:$RUNROOT"
  exit 0
}

if [ -f pristine_vcrelax/pw.out ]; then
  pristine_done="$(grep -c "JOB DONE" pristine_vcrelax/pw.out 2>/dev/null)"
else
  pristine_done=0
fi

if [ -f vacancy_vcrelax/pw.out ]; then
  vacancy_done="$(grep -c "JOB DONE" vacancy_vcrelax/pw.out 2>/dev/null)"
else
  vacancy_done=0
fi

echo "pristine_done=${pristine_done:-0}"
echo "vacancy_done=${vacancy_done:-0}"
