"""Full secondary coverage, reuse, refits and protected manuscript updates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import full_secondary_study as secondary
from integrate_full_secondary import replace_once
from report_full_secondary import patient_means, stats
import test_parallel_experiments as pilot_tests


class FullSecondaryTests(unittest.TestCase):
    def test_reuse_noncontiguous_pilot_by_identity_not_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            patients=[{'patient_index':i,'patient':f'p{i}','plan':f'/synthetic/patient-{i}/plan.json'} for i in range(135)]
            chosen=[2,7,12,18,20,45,65,90,110,134]
            pilot={'patients':[dict(patients[i],patient_index=j) for j,i in enumerate(chosen)]}
            path=root/'pilot.json';path.write_text(json.dumps(pilot))
            records=secondary.mappings(patients,pilot,path,root/'full')
            self.assertEqual([len(x) for x in records],[405,135,375,125])
            self.assertEqual({r['patient_index'] for r in records[0] if r['reused']},set(chosen))
            self.assertEqual(next(r for r in records[0] if r['patient_index']==134 and r['seed']==2)['task'],29)
            plan={'patients':patients,'pilot_plan':str(path),'root':str(root/'full'),
                  **dict(zip(('cpu_records','loo_records','cpu_tasks','loo_tasks'),records))}
            secondary.check_mapping(plan)
            plan['cpu_tasks'][0]['seed']=2
            with self.assertRaisesRegex(ValueError,'mapping'):secondary.check_mapping(plan)

    def test_reused_record_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=root/'plan.json';plan.write_text('{}')
            values=root/'values.csv';values.write_text('value\n1\n')
            record={'directory':str(root),'task':4,'source_patient_index':1,'patient_index':77,'seed':1,'source_plan':str(plan)}
            report={'status':'passed','command':'cpu','task':4,'patient_index':1,'seed':1,
                    'plan_sha256':secondary.study.sha256(plan),'output_hashes':[secondary.hashed(values)]}
            (root/'report.json').write_text(json.dumps(report))
            secondary.check_record(record,'cpu')
            values.write_text('value\n2\n')
            with self.assertRaisesRegex(ValueError,'changed'):secondary.check_record(record,'cpu')

    def test_loo_requires_complete_images_and_correct_class_sign(self):
        child={'images':['a','b','c','d']};report={'original_prediction':{'p_class1':.3,'yhat':0},'model_queries':5}
        rows=[{'omitted_image':image,'retained_count':3,'p_class1':.4,'delta_class1':-.1,'delta_original_class':.1} for image in child['images']]
        secondary.validate_loo(child,report,rows)
        rows[0]['delta_original_class']=-.1
        with self.assertRaisesRegex(ValueError,'sign'):secondary.validate_loo(child,report,rows)
        with self.assertRaisesRegex(ValueError,'coverage'):secondary.validate_loo(child,report,rows[:-1])

    def test_missing_values_cannot_create_complete_patient_or_cohort_mean(self):
        frame=pd.DataFrame([{'patient_index':0,'seed':s,'value':v} for s,v in enumerate([.1,np.nan,.3])])
        means=patient_means(frame,['value'],['patient_index'])
        self.assertTrue(np.isnan(means.value.iloc[0]))
        result=stats([.2,np.nan],2)
        self.assertIsNone(result['complete_cohort_mean']);self.assertEqual(result['defined'],1)
        self.assertAlmostEqual(result['available_patient_mean'],.2)
        self.assertIsNone(secondary.complete_mean([.1,None,.3],3))

    def test_patient_means_reject_duplicate_seeds(self):
        frame=pd.DataFrame([{'patient_index':0,'seed':s,'value':.1} for s in [0,0,2]])
        with self.assertRaisesRegex(ValueError,'Duplicate/missing'):patient_means(frame,['value'],['patient_index'])

    def test_independent_refits_detect_changed_coefficients(self):
        helper=pilot_tests.ParallelTests();helper.setUp();self.addCleanup(helper.doCleanups)
        output=helper.synthetic_run('refit',False);child=json.loads((output.parent/'plan.json').read_text())
        settings={'alphas':[.001,.01,.1],'l1_ratios':[.5,.9],'folds':3,'tol':1e-5,'max_iter':20000}
        self.assertEqual(secondary.independent_cpu_check(child,output.parent/'run',output,0,settings),4)
        rows=pd.read_csv(output/'rankings.csv');rows.loc[rows.method=='elastic_net','value']+=.01
        rows.to_csv(output/'rankings.csv',index=False)
        with self.assertRaisesRegex(ValueError,'Elastic Net refit'):secondary.independent_cpu_check(child,output.parent/'run',output,0,settings)

    def test_minimal_text_patch_preserves_user_writing_and_rejects_conflict(self):
        user='My revised introduction. OLD SCOPE My revised conclusion.'
        self.assertEqual(replace_once(user,'OLD SCOPE','NEW SCOPE','scope'),
                         'My revised introduction. NEW SCOPE My revised conclusion.')
        with self.assertRaisesRegex(ValueError,'Text conflict'):replace_once(user,'missing','new','scope')
        with self.assertRaisesRegex(ValueError,'Text conflict'):replace_once('OLD OLD','OLD','NEW','scope')


if __name__=='__main__':unittest.main()
