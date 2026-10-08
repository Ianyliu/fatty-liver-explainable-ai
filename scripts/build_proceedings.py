#!/usr/bin/env python3
"""Build a private, evidence-backed JSM review package without GNN inference.

Explicit pilot/full modes; full mode refuses partial aggregates. Publication is
never performed. Submission PDF generation requires the author's release gate.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript/proceedings_2026"


def command(args, cwd=None):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError('Command failed: '+args[0]+'\n'+result.stdout[-3500:]+'\n'+result.stderr[-1000:])
    return result.stdout


def check_references():
    bib=(SOURCE/'references.bib').read_text()
    entries=re.findall(r'^@\w+\{([^,]+),',bib,re.M)
    if len(entries)!=len(set(entries)):
        raise ValueError('Duplicate bibliography keys')
    lines=bib.splitlines()
    for index,line in enumerate(lines):
        if line.startswith('@') and (index==0 or not re.match(r'% Original source: https?://\S+',lines[index-1])):
            raise ValueError('Every bibliography entry requires an immediately preceding original-source URL')
    text='\n'.join((SOURCE/name).read_text() for name in ('body.tex','appendix.tex'))
    citations={key.strip() for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',text) for key in group.split(',')}
    manifest=json.loads((SOURCE/'reference_verification.json').read_text())
    verified={row['key'] for row in manifest['references'] if row['status']=='verified'}
    if citations != set(entries) or not set(entries)<=verified:
        raise ValueError('Unresolved/unused/unverified bibliography entry: '+str((citations^set(entries))|(set(entries)-verified)))
    dois=re.findall(r'doi\s*=\s*\{([^}]+)\}',bib)
    if len(dois)!=len(set(dois)):
        raise ValueError('Duplicate DOI')
    return manifest


def release_gate():
    confirmations=json.loads((SOURCE/'release_confirmations.json').read_text())
    missing=[name for name,confirmed in confirmations['confirmations'].items() if confirmed is not True]
    if missing:
        raise ValueError('Submission PDF blocked by required author confirmations: '+', '.join(missing))
    text='\n'.join((SOURCE/name).read_text() for name in ('body.tex','appendix.tex'))
    if r'\pending{' in text or 'AUTHOR CONFIRMATION' in text:
        raise ValueError('Submission PDF blocked: unresolved declaration placeholders')
    main=(SOURCE/'main.tex').read_text()
    if 'Internal review' in main or 'require author confirmation' in main or 'require confirmation' in main:
        raise ValueError('Submission PDF blocked: source still contains internal-review author/title text')
    family=command(['fc-match','--format','%{family}','Times New Roman']).strip()
    if family!='Times New Roman' and not confirmations.get('asa_font_exception_evidence'):
        raise ValueError('Submission PDF blocked: Times New Roman absent and no documented ASA font exception')
    if confirmations.get('release_status')!='approved_for_submission_pdf':
        raise ValueError('Submission PDF requires explicit final review approval recorded by the authors')
    return confirmations


def contactsheet(figures):
    import matplotlib.pyplot as plt
    from proceedings_style import configure, save
    configure();fig,axes=plt.subplots(3,2,figsize=(11,16))
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.93,wspace=.04,hspace=.10)
    files=sorted(figures.glob('figure[1-5]_*.png'))
    for ax,path in zip(axes.flat,files):
        ax.imshow(plt.imread(path));ax.axis('off');ax.set_title(path.stem.replace('_',' '),fontsize=10)
    for ax in list(axes.flat)[len(files):]:ax.axis('off')
    fig.suptitle('JSM 2026 proceedings · figure consistency review',fontsize=14,y=.99)
    save(fig,figures,'contact_sheet')


def historical_workflow(output):
    import matplotlib.pyplot as plt
    from proceedings_style import save
    original=SOURCE/'assets/original_workflow.png'
    fig,ax=plt.subplots(figsize=(11,8.5));fig.subplots_adjust(left=.025,right=.975,bottom=.08,top=.92)
    ax.imshow(plt.imread(original));ax.axis('off')
    fig.suptitle('Preserved historical workflow — supplementary source, not current experimental evidence',fontsize=10)
    fig.text(.04,.035,'Historical classifier / 10-fold CV and physician-evaluation labels are not completed analyses in this manuscript.',fontsize=9)
    save(fig,output,'original_workflow_historical')
    shutil.copy2(original,output/original.name)
    for name in ('original_workflow.drawio','original_workflow.json'):
        shutil.copy2(SOURCE/'assets'/name,output/name)


def build(args):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/proceedings_2026/.matplotlib'))
    from threadpoolctl import threadpool_limits
    from proceedings_data import FULL, hash_entry, load_full, load_pilot
    from proceedings_tables import export_tables, tex
    from proceedings_workflow import workflow
    from proceedings_sampling_figure import sampling_figure
    from proceedings_fidelity_figure import fidelity_figure
    from proceedings_deletion_figure import deletion_figure
    from proceedings_diagnostics_figure import diagnostics_figure
    output=args.output.resolve()
    allowed=(ROOT/'outputs/proceedings_2026').resolve()
    if allowed not in output.parents or output.exists():
        raise ValueError('Use a NEW directory below ignored outputs/proceedings_2026/')
    references=check_references()
    if args.submission_pdf:release_gate()
    # Crucially, validate full completeness before creating a directory or building any pilot assets.
    with threadpool_limits(limits=1):
        main=load_full(args.full_plan) if args.phase=='full' else None
        pilot=load_pilot()
        if main is None:main=pilot
    output.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
    os.environ['SOURCE_DATE_EPOCH']='1791417600'
    work=output/'latex';shutil.copytree(SOURCE,work)
    generated=work/'generated';figures=work/'figures'
    generated.mkdir();figures.mkdir()
    # Explicit OpenType files avoid duplicate Fontconfig Type-1 family matches.
    font=command(['fc-match','--format','%{family}','Times New Roman']).strip()
    if font=='Times New Roman':
        font_setup=r'\setmainfont{Times New Roman}'
    else:
        path=Path(command(['fc-match','--format','%{file}','Nimbus Roman']).strip())
        if path.suffix!='.otf':
            raise ValueError('Review build needs the installed Nimbus Roman OpenType font')
        font_setup=('\\setmainfont{NimbusRoman}[Path='+str(path.parent)+'/,Extension=.otf,'
                    'UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic]')
    (generated/'font_setup.tex').write_text(font_setup+'\n')
    ledger,metrics,pilot_metrics,secondary=export_tables(main,pilot,generated)
    workflow(figures)
    captions={
        'SamplingCaption':sampling_figure(main,figures),
        'FidelityCaption':fidelity_figure(main,pilot,figures),
        'DeletionCaption':deletion_figure(main,figures),
        'DiagnosticsCaption':diagnostics_figure(pilot,figures)}
    status=('All 135 eligible patients and 405 patient–seed runs have passed raw-evidence validation. '
            'This expansion followed inspection of the pilot, reuses its ten patients, and remains exploratory.'
            if args.phase=='full' else
            'Only the ten-patient pilot is reported in this review build. The submitted 135-patient expansion is pending complete validation; no expansion result is presented.')
    with (generated/'results.tex').open('a') as handle:
        framing={'PrimaryStatusText':status,
            'PrimaryResultsHeading':'Expanded eligible-cohort analysis' if args.phase=='full' else 'Pilot fidelity and availability',
            'ExpansionAbstractText':('Expansion to 135 eligible positive patients followed inspection of the pilot.' if args.phase=='full'
                                     else 'Expansion to 135 eligible positive patients was submitted after inspecting the pilot and remains pending validation.'),
            'AnalysisAbstractText':'In the expanded analysis' if args.phase=='full' else 'In the ten-patient pilot'}
        for name,text in {**framing,**captions}.items():
            handle.write('\\newcommand{\\'+name+'}{'+tex(text)+'}\n')
    for name,caption in captions.items():
        ledger['entries'].append({'id':'figure:'+name,'definition':caption,'population':'pilot' if name=='DiagnosticsCaption' else args.phase,
            'source_set':'pilot' if name=='DiagnosticsCaption' else 'primary','validation':'passed'})
    ledger['entries'].append({'id':'figure:workflow','definition':'Explicit adaptation of original workflow topology with historical/current differences.',
        'source':hash_entry(SOURCE/'assets/original_workflow.png'),'editable_original':hash_entry(SOURCE/'assets/original_workflow.drawio'),
        'provenance':hash_entry(SOURCE/'assets/original_workflow.json')})
    ledger['entries'].append({'id':'methods:protocol','definition':'Fixed scientific settings and source-publication comparison',
        'values':{'seeds':[0,1,2],'training_rows_per_arm':1000,'shared_evaluation_rows':200,'subset_minimum':3,
                  'ridge_alpha':1,'adaptive_class1_target':.5,'pool_bias': [.85,.15],
                  'elastic_net_cv_folds':5,'elastic_net_alphas':25,'elastic_net_alpha_range':[1e-4,1],
                  'elastic_net_l1_ratios':[.1,.5,.9,1],'actual_correlation_threshold':.95,'source_publication_threshold':.995},
        'source_set':'primary and pilot frozen plans; source publication'})
    contactsheet(figures);historical_workflow(output/'supplementary')
    (output/'figure_captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    (output/'evidence_ledger.json').write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
    shutil.copy2(SOURCE/'reference_verification.json',output/'reference_verification.json')
    shutil.copy2(SOURCE/'release_confirmations.json',output/'release_confirmations.json')
    status_record={
        'phase':args.phase,'new_inference_queries':0,'publication_allowed':False,
        'completed':[
            'uv environment and CPU/GPU smoke validation; recovered-image checksums and eligibility coverage',
            'ten-patient three-seed P0 comparison with raw-query validation',
            'ten-patient Elastic Net, Pearson, seed stability and leave-one-image-out analyses',
            'independent review including 120 surrogate refits',
            'CPU-only saved-artifact design diagnostics and ranking/LOO agreement',
            'five multi-panel figures, five numbered tables, verified references and manuscript review draft'],
        'full_cohort':('135 patients and 405 runs validated, including the ten reused pilot patients'
                       if args.phase=='full' else 'pending GPU array 9558368 and validation summary 9558371; no partial results reported'),
        'pending':['Ian manuscript/figure review; Prof. Yen authorship/content approval opportunity',
                   'affiliations, correspondence, contribution assignments, funding, interests, secondary-use and access confirmations',
                   'Times New Roman installation or documented ASA font exception',
                   'final manual PDF/figure QA and explicit release approval'],
        'deferred':['matched total-query budgets','sampling-ratio and subset-size sensitivity',
                    'repeated random deletion orders','deletion using additional explanation rankings',
                    'full-cohort Elastic Net, Pearson, stability and leave-one-image-out',
                    'bootstrap inference, significance testing, physician validation and clustering ablations']}
    (output/'analysis_status.json').write_text(json.dumps(status_record,indent=2)+'\n')
    publication_sources=[*sorted((ROOT/'scripts').glob('proceedings_*.py')),Path(__file__).resolve()]
    for path in publication_sources:
        destination=output/'source_snapshot/scripts'/path.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    commands=[['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],
              ['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
    for index,cmd in enumerate(commands):
        (output/f'build-{index+1}.log').write_text(command(cmd,work))
    target=output/('manuscript_submission.pdf' if args.submission_pdf else 'manuscript_review.pdf')
    shutil.copy2(work/'main.pdf',target)
    pdfinfo=command(['pdfinfo',str(target)]);pdffonts=command(['pdffonts',str(target)])
    text=command(['pdftotext',str(target),'-'])
    (output/'manuscript_text.txt').write_text(text)
    (output/'pdfinfo.txt').write_text(pdfinfo);(output/'pdffonts.txt').write_text(pdffonts)
    log=(work/'main.log').read_text(errors='replace')
    if re.search(r'Citation .* undefined|There were undefined references|Missing character:',log):
        raise ValueError('PDF QA failed: unresolved citation/reference or missing font glyph')
    if not re.search(r'Encrypted:\s+no\b',pdfinfo):
        raise ValueError('PDF QA failed: nonsecured PDF required')
    # Check the rendered PDF for patient identifiers, not only the authored text.
    private_ids=[]
    for path in (ROOT/'outputs/complete_patient/cohort-20261007-v1/cohort_plan.json',args.full_plan):
        plan=json.loads(Path(path).read_text());private_ids.extend(str(item['patient']) for item in plan['patients'])
    if any(re.search(r'(?<!\w)'+re.escape(identifier)+r'(?!\w)',text) for identifier in set(private_ids)):
        raise ValueError('PDF QA failed: private patient identifier detected')
    if re.search(r'IMG\d{4}|MI_ID|(?i:access_token=|authorization: bearer)',text):
        raise ValueError('PDF QA failed: private field/token pattern detected')
    font=command(['fc-match','--format','%{family}','Times New Roman']).strip()
    qa={'citations_resolved':True,'nonsecured_pdf':True,'no_patient_identifiers_in_extracted_text':True,
        'letter_page':bool(re.search(r'Page size:\s+612 x 792 pts',pdfinfo)),
        'font_family_for_times_request':font,'font_substitution':font!='Times New Roman',
        'overfull_boxes':re.findall(r'Overfull \\hbox .*',log),
        'visual_review':'pending: inspect rendered pages and contact sheet at actual manuscript size',
        'submission_ready':bool(args.submission_pdf),
        'release_status':'approved_for_submission_pdf' if args.submission_pdf else 'internal_review_only'}
    (output/'pdf_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    source_files=[p for p in SOURCE.rglob('*') if p.is_file()]+publication_sources
    manifest={'schema':1,'phase':args.phase,'created_at':datetime.now(timezone.utc).isoformat(),
        'status':'review_package_built','analysis_patients':main['n'],'validated_runs':main['n']*3,
        'model_queries':main['queries'],'new_inference_queries':0,'git_commit':command(['git','rev-parse','HEAD'],ROOT).strip(),
        'sources':[hash_entry(p) for p in source_files],
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','scikit-learn','matplotlib','Pillow')},
        'reference_count':len(references['references']),'commands':commands,'release_gate':'No public release without Ian review and Prof. Yen approval opportunity.'}
    manifest['artifacts']=[hash_entry(p) for p in [target,output/'evidence_ledger.json',*sorted(generated.glob('*.csv')),*sorted(figures.glob('*.pdf'))]]
    (output/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'git_commits.txt').write_text(command(['git','log','--format=%h %s','f59092f^..HEAD','--','manuscript/proceedings_2026','scripts/proceedings_*.py','scripts/build_proceedings.py','tests/test_proceedings.py'],ROOT))
    instructions=f'''# Private review package — {args.phase} mode

Read `manuscript_review.pdf` (or the explicitly gated submission PDF), `latex/figures/contact_sheet.pdf`, generated CSV/TeX tables, `evidence_ledger.json`, `analysis_status.json`, and `pdf_qa.json`. Author confirmations are in `release_confirmations.json`. The complete source original, recovered editable draw.io file and provenance are in `supplementary/`; the editable adaptation is generated by `source_snapshot/scripts/proceedings_workflow.py`.

Regenerate with the existing environment, without syncing or relinking it:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/build_proceedings.py --phase {args.phase} --output outputs/proceedings_2026/NEW_DIRECTORY
```

System tools: XeLaTeX, BibTeX, fontspec, geometry, natbib, booktabs, tabularx, graphicx, pdftotext, pdfinfo, pdffonts, Fontconfig. Python versions are pinned in `build_manifest.json` and the project environment. R is not required. No network access, package installation, model loading or GNN inference is performed by the build. Full mode requires the complete validated expansion summary and all preserved runs.

This package is PRIVATE: the evidence ledger contains restricted source paths. Do not publish the whole package. Public release of the manuscript itself requires Ian's final review, Prof. Yen's approval opportunity, resolved declarations and final font QA. No submission was performed.
'''
    (output/'README.md').write_text(instructions)
    print('Built private review package:',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('pilot','full'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--full-plan',type=Path,default=ROOT/'outputs/complete_patient/full-135-20261007-v2/full_cohort_plan.json')
    parser.add_argument('--submission-pdf',action='store_true',help='Require explicit author confirmations, no placeholders and final typography; does not publish')
    args=parser.parse_args()
    build(args)


if __name__=='__main__':
    main()
