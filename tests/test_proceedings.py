"""Scientific reporting gates: patient weighting, availability and release safety."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from proceedings_data import finite_correlation, full_gate, patient_average, strict_mean
from proceedings_tables import stability_patient
from build_proceedings import check_references, release_gate


class ProceedingsReportingTests(unittest.TestCase):
    def test_seed_means_then_equal_patient_weight(self):
        frame=pd.DataFrame([{'patient_index':p,'arm':'random','seed':s,'mae':v,'mask_count':100 if p else 1}
            for p,values in [(0,[.1,.2,.3]),(1,[10.,20.,30.])] for s,v in enumerate(values)])
        means=patient_average(frame,['mae'])
        self.assertAlmostEqual(strict_mean(means.mae),10.1)

    def test_missing_seed_cannot_be_averaged_away(self):
        frame=pd.DataFrame([{'patient_index':0,'arm':'random','seed':s,'mae':.1} for s in (0,1)])
        with self.assertRaisesRegex(ValueError,'Missing/duplicate seed'):patient_average(frame,['mae'])

    def test_unavailable_seed_blocks_complete_patient_metric(self):
        frame=pd.DataFrame([{'patient_index':0,'arm':'random','seed':s,'mae':v} for s,v in enumerate([.1,np.nan,.3])])
        means=patient_average(frame,['mae'])
        self.assertTrue(np.isnan(means.mae.iloc[0]))
        self.assertIsNone(strict_mean(means.mae))

    def test_constant_ranking_is_not_stability(self):
        frame=pd.DataFrame([{'patient_index':0,'arm':'random','method':'elastic_net','defined':False,
            'spearman':np.nan,'top_five_jaccard':1.,'sign_agreement':1.} for _ in range(3)])
        result=stability_patient(frame)
        self.assertTrue(result[['spearman','top_five_jaccard','sign_agreement']].isna().all().all())
        self.assertTrue(np.isnan(finite_correlation([0,0,0],[1,2,3])))

    def test_full_gate_rejects_absent_and_partial_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'full_cohort_plan.json'
            with self.assertRaisesRegex(ValueError,'absent'):full_gate(p)
            (p.parent/'summary').mkdir()
            (p.parent/'summary/analysis.json').write_text(json.dumps({'status':'passed','validated_patient_seed_runs':404}))
            with patch('proceedings_data.full_cohort_study.verified',return_value={}):
                with self.assertRaisesRegex(ValueError,'405'):full_gate(p)

    def test_reference_sources_and_citations_resolve(self):
        manifest=check_references()
        self.assertGreaterEqual(len(manifest['references']),6)

    def test_submission_output_requires_actual_author_confirmations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'release_confirmations.json').write_text(json.dumps({'confirmations':{'ian_final_review':False}}))
            with patch('build_proceedings.SOURCE',root):
                with self.assertRaisesRegex(ValueError,'author confirmations'):release_gate()

    def test_checked_boxes_cannot_replace_missing_author_statements(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'release_confirmations.json').write_text(json.dumps({'confirmations':{'ian_final_review':True}}))
            (root/'author_declarations.json').write_text(json.dumps({'funding':None,'competing_interests':''}))
            with patch('build_proceedings.SOURCE',root):
                with self.assertRaisesRegex(ValueError,'missing declaration text'):release_gate()


if __name__=='__main__':unittest.main()
