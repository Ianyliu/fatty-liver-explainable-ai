"""Regress the nested recovery/original-plan figure lookup and profile reuse."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finalize_full_secondary_report import primary_sources,check_profile_csv,STEMS
import full_secondary_study as secondary

class FinalizeTests(unittest.TestCase):
    def test_resolves_primary_exports_from_original_not_retry_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);figures=root/'manuscript/latex/figures';figures.mkdir(parents=True)
            exports=[]
            for stem in STEMS:
                for suffix in ('pdf','svg','png'):
                    path=figures/(stem+'.'+suffix);path.write_text(stem);exports.append(secondary.hashed(path))
            original=root/'original.json';original.write_text(json.dumps({'files':exports}))
            retry={'recovery_base_plan':str(original),'primary_figure_package':str(root/'manuscript'),'files':[secondary.hashed(original)]}
            found=primary_sources(retry);self.assertEqual(len(found),15)
            (figures/'figure2_influence.png').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):primary_sources(retry)

    def test_missing_original_figure_rejects_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'original.json';path.write_text('{"files":[]}')
            with self.assertRaisesRegex(ValueError,'all 15'):
                primary_sources({'recovery_base_plan':str(path),'primary_figure_package':directory})

    def test_profile_reuse_preserves_undefined_values_and_detects_changed_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.csv';expected=pd.DataFrame({'ridge':[.01,.02],'marginal_correlation':[np.nan,.2],'image_label':['I01','I02']})
            expected.columns.name='method'
            expected.to_csv(path,index=False);check_profile_csv(path,expected)
            changed=expected.copy();changed.loc[0,'ridge']=.1;changed.to_csv(path,index=False)
            with self.assertRaises(AssertionError):check_profile_csv(path,expected)

if __name__=='__main__':unittest.main()
