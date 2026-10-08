#!/usr/bin/env python3
"""Execute one recorded private manuscript build; no model inference or release."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest['phase'] not in ('pilot','full') or manifest['publication_allowed'] is not False:
        raise ValueError('Only private review builds are allowed')
    plan=Path(manifest['full_plan'])
    if digest(plan)!=manifest['full_plan_sha256']:
        raise ValueError('Recorded frozen inference plan changed')
    output=Path(manifest['output']).resolve()
    if (ROOT/'outputs/proceedings_2026').resolve() not in output.parents or output.exists():
        raise ValueError('Review output must be a new private directory')
    # Snapshot the actual publication sources in the receipt; inference stays frozen.
    paths=[*sorted((ROOT/'scripts').glob('proceedings_*.py')),ROOT/'scripts/build_proceedings.py',Path(__file__),
           *[p for p in (ROOT/'manuscript/proceedings_2026').rglob('*') if p.is_file()]]
    files=[{'path':str(p.resolve()),'sha256':digest(p)} for p in paths]
    receipt=args.manifest.with_name(args.manifest.stem+'-execution.json')
    cmd=[sys.executable,str(ROOT/'scripts/build_proceedings.py'),'--phase',manifest['phase'],
         '--full-plan',str(plan),'--output',str(output)]
    record={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'command':cmd,'configuration_sha256':digest(args.manifest),
        'source_hashes_at_start':files,'source_policy':'Publication source selected at execution; unchanged inference plan required.',
        'new_inference_queries':0,'publication_allowed':False}
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    try:
        result=subprocess.run(cmd,cwd=ROOT,check=True)
        changed=[row['path'] for row in files if digest(row['path'])!=row['sha256']]
        if changed:raise RuntimeError('Publication source changed during build: '+', '.join(changed))
        record.update(status='passed',build_manifest_sha256=digest(output/'build_manifest.json'))
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:
        record['finished_at']=datetime.now(timezone.utc).isoformat()
        receipt.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
