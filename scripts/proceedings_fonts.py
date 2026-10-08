"""Resolve verified local typesetting fonts without installing system packages."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def resolve_font():
    directory=Path(os.environ.get('XAI_TIMES_FONT_DIR',str(ROOT/'outputs/proceedings_2026/fonts/times-new-roman')))
    source=ROOT/'manuscript/proceedings_2026/font_source.json'
    if source.exists() and all((directory/name).is_file() for name in ('Times.TTF','Timesbd.TTF','Timesi.TTF','Timesbi.TTF')):
        manifest=json.loads(source.read_text());files=[]
        for name,entry in manifest['files'].items():
            path=directory/name
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=entry['sha256']:
                raise ValueError('Typesetting font hash changed: '+str(path))
            family=subprocess.check_output(['fc-scan','--format','%{family}',str(path)],text=True)
            if family!='Times New Roman':raise ValueError('Incorrect typesetting font family')
            files.append({'path':str(path.resolve()),'sha256':digest})
        setup=('\\setmainfont{Times}[Path='+str(directory.resolve())+'/,Extension=.TTF,'
               'UprightFont=*,BoldFont=*bd,ItalicFont=*i,BoldItalicFont=*bi]')
        return {'family':'Times New Roman','setup':setup,'files':files,'substitution':False}
    family=subprocess.check_output(['fc-match','--format','%{family}','Times New Roman'],text=True).strip()
    if family=='Times New Roman':
        return {'family':family,'setup':r'\setmainfont{Times New Roman}','files':[],'substitution':False}
    path=Path(subprocess.check_output(['fc-match','--format','%{file}','Nimbus Roman'],text=True).strip())
    if path.suffix!='.otf':raise ValueError('No Times New Roman or Nimbus Roman OpenType font available')
    setup=('\\setmainfont{NimbusRoman}[Path='+str(path.parent)+'/,Extension=.otf,'
           'UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic]')
    return {'family':'Nimbus Roman','setup':setup,'files':[],'substitution':True}
