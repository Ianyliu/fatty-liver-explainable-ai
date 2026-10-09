#!/usr/bin/env python3
"""Build the reviewed editable package directly, without invoking legacy exporters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ALLOWED = {'figure1_workflow', 'figure2_influence', 'figure3_sampling',
           'figure4_fidelity', 'figure5_deletion', 'secondary_fidelity', 'loo_behavior'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--font-dir', type=Path, required=True,
                        help='Locally licensed Times New Roman directory; never packaged')
    parser.add_argument('--latexmk', default=os.environ.get('LATEXMK', 'latexmk'))
    args = parser.parse_args()
    package, fonts = args.package.resolve(), args.font_dir.resolve()
    latex = package / 'latex'
    text = '\n'.join((latex / n).read_text() for n in ('main.tex', 'body.tex', 'appendix.tex'))
    used = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\.pdf\}', text)
    if set(used) != ALLOWED or len(used) != len(ALLOWED):
        raise ValueError('Exactly the seven reviewed figures must be referenced once each')
    if {f.stem for f in (latex / 'figures').glob('*.pdf')} != ALLOWED:
        raise ValueError('Unexpected or missing figure PDFs')
    original = json.loads((package / 'revision_qa.json').read_text())
    if sha(latex / 'figures/figure2_influence.pdf') != original['original_figure2_sha256']:
        raise ValueError('The reviewed original Figure 2 changed')
    source = json.loads((package / 'original_evidence/font_source.json').read_text())
    for name, record in source['files'].items():
        path = fonts / name
        if sha(path) != record['sha256']:
            raise ValueError('Font hash mismatch: ' + name)
        family = subprocess.check_output(['fc-scan', '--format', '%{family}', str(path)], text=True)
        if family != 'Times New Roman':
            raise ValueError('Incorrect font family: ' + name)
    (latex / 'generated/font_setup_server.tex').write_text(
        r'\setmainfont{Times}[Path=' + str(fonts) +
        r'/,Extension=.TTF,UprightFont=*,BoldFont=*bd,ItalicFont=*i,BoldItalicFont=*bi]' + '\n')
    if r'\input{generated/font_setup_server.tex}' not in (latex / 'main.tex').read_text():
        raise ValueError('Reviewed main.tex must input generated/font_setup_server.tex')
    subprocess.run([sys.executable, str(package / 'scripts/generate_manuscript_summaries.py')], check=True)
    command = [args.latexmk, '-xelatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']
    with (package / 'server_build_console.txt').open('w') as handle:
        subprocess.run(command, cwd=latex, stdout=handle, stderr=subprocess.STDOUT, check=True)
    log = (latex / 'main.log').read_text(errors='replace')
    critical = [line for line in log.splitlines() if re.search(
        r'Overfull|Missing character|undefined|multiply defined|Float too large|LaTeX Error|fontspec error', line, re.I)]
    if critical:
        raise ValueError('Build QA failed: ' + '\n'.join(critical))
    info = subprocess.check_output(['pdfinfo', str(latex / 'main.pdf')], text=True)
    font_listing = subprocess.check_output(['pdffonts', str(latex / 'main.pdf')], text=True)
    if '612 x 792 pts' not in info or not re.search(r'Encrypted:\s+no', info):
        raise ValueError('Expected nonsecured US-letter PDF')
    lines = font_listing.splitlines()[2:]
    if not any('TimesNewRomanPSMT' in line for line in lines):
        raise ValueError('Times New Roman body font absent')
    if any(not re.search(r'\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$', line) for line in lines):
        raise ValueError('Unembedded PDF font detected')
    (package / 'server_evidence/final_pdffonts.txt').write_text(font_listing)
    (package / 'server_evidence/final_pdfinfo.txt').write_text(info)
    shutil.copy2(latex / 'main.pdf', package / 'JSM2026_server_revised_manuscript.pdf')
    record = {'status': 'passed', 'command': command, 'pdf_sha256': sha(latex / 'main.pdf'),
              'critical_warnings': critical, 'underfull_boxes': log.count('Underfull'),
              'figures': {name: sha(latex / 'figures' / (name + '.pdf')) for name in sorted(ALLOWED)},
              'fonts': 'Times New Roman body; all PDF fonts embedded',
              'publication_allowed': False, 'visual_review': 'Requires inspection of this PDF hash'}
    (package / 'server_build_checks.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Built', package / 'JSM2026_server_revised_manuscript.pdf')

if __name__ == '__main__':
    main()
