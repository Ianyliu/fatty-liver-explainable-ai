#!/usr/bin/env python3
"""Run the frozen in-place updater with narrow layout repairs to current edits.

Restore the original Figure 2 include only if its referenced block is missing.
Keep the user's words and use local paragraph wrapping for a confirmed overflow.
Both adapters run within the updater's staged copy and concurrent-edit guard.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

import full_secondary_study as secondary
import integrate_full_secondary_recovery as integration
import patient_study as study

ORIGINAL_REPLACE=integration.replace_once
ORIGINAL_RUN=integration.run


def repair_body(body):
    if r'\label{fig:influence}' not in body:
        anchor=re.findall(r'(?m)^[^\n]*\\ref\{fig:influence\}[^\n]*$',body)
        secondary.require(len(anchor)==1,'Missing/ambiguous Figure 2 reference; preserve user text for review')
        block=(r'\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/figure2_influence.pdf}'+'\n'
               +r'\caption{\InfluenceCaption}\label{fig:influence}\end{figure}')
        body=body.replace(anchor[0],anchor[0]+'\n\n'+block,1)
    paragraphs=re.findall(r'(?m)^The supplied model inherits[^\n]*$',body)
    secondary.require(len(paragraphs)==1,'Ambiguous grouped-model paragraph; preserve user text for review')
    # This only changes TeX line-breaking flexibility; every original word stays.
    body=body.replace(paragraphs[0],r'{\emergencystretch=2em'+'\n'+paragraphs[0]+'\n'+r'\par}',1)
    return body


def replace_adapter(text,old,new,label):
    result=ORIGINAL_REPLACE(text,old,new,label)
    if label=='new full secondary results placement':result=repair_body(result)
    return result


def run_adapter(command,cwd):
    result=ORIGINAL_RUN(command,cwd)
    if command[0]=='pdftotext':
        log=(Path(cwd)/'main.log').read_text(errors='replace')
        secondary.require(not re.search(r'Overfull \\hbox|Float too large|There were undefined references|Missing character:',log),
                          'Staged PDF formatting/reference QA failed; existing manuscript preserved')
    return result


def main(args):
    integration.replace_once=replace_adapter;integration.run=run_adapter
    integration.integrate(args)
    target=args.manuscript.resolve();receipt_path=target/'full_secondary_integration.json'
    receipt=json.loads(receipt_path.read_text());receipt['layout_repairs']={
        'source':secondary.hashed(Path(__file__)),'original_figure2_block_restored_if_missing':True,
        'grouped_model_paragraph_wording_unchanged':True,'local_paragraph_emergency_stretch':'2em',
        'references_and_actual_overfull_hbox_check':'passed before installation'}
    study.write_json(receipt_path,receipt)
    snapshot=target/'source_snapshot/scripts'/Path(__file__).name;shutil.copy2(__file__,snapshot)
    ledger_path=target/'evidence_ledger.json';ledger=json.loads(ledger_path.read_text())
    ledger['sources'].append(secondary.hashed(Path(__file__)));study.write_json(ledger_path,ledger)
    manifest_path=target/'build_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['integrations'][-1]=receipt
    manifest['artifacts']=[secondary.hashed(row['path']) for row in manifest['artifacts']]
    manifest['artifacts'].append(secondary.hashed(snapshot));study.write_json(manifest_path,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--manuscript',type=Path,required=True)
    main(parser.parse_args())
