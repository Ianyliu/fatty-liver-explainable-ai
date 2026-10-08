"""Additional experiment invariants using synthetic predictions and designs."""
from contextlib import redirect_stdout
import io
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import parallel_experiments as extra


class ParallelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_design_audit_detects_duplicates_and_singular_columns(self):
        x = np.array([[1,0,1],[1,0,1],[0,1,1],[0,1,1]])
        summary, images = extra.design_diagnostics(x,np.array([1,1,0,0]),['a','b','c'])
        self.assertEqual(summary['duplicate_rows'],2)
        self.assertEqual(summary['centered_design_rank'],1)
        self.assertIsNone(summary['centered_design_condition_number'])
        self.assertEqual(images[2]['excluded_rows'],0)

    def test_stage_audit_verifies_pool_deficits_and_rejects_wrong_composition(self):
        images = ['a','b','c','d']
        ledger = [dict(stage='adaptive_singletons',yhat=int(i==0),**{k:int(i==j) for j,k in enumerate(images)}) for i in range(4)]
        training = [dict(yhat=0,a=1,b=1,c=1,d=1) for _ in range(4)]
        summary = dict(requested_class_counts={'0':2,'1':2},stages={'random':2,'biased':2},one_pool_fallback=False)
        report, rows = extra.stage_two_audit(training,ledger,images,summary)
        self.assertEqual(report['biased_rows_with_pool_deficit'],2)
        self.assertEqual(rows[0]['requested_positive_count'],3)
        self.assertEqual(rows[0]['realized_positive_count'],1)
        training[-1]['a']=0
        with self.assertRaisesRegex(ValueError,'composition'):
            extra.stage_two_audit(training,ledger,images,summary)

    def test_constant_marginal_correlations_and_rankings_are_unavailable(self):
        self.assertEqual(extra.marginal_correlations(np.ones((5,3)),np.ones(5)),[None]*3)
        rows = [dict(seed=s,arm=a,method=m,image_id=f'image{i}',value=0.)
                for s in (0,1,2) for a in extra.ARMS for m in extra.METHODS for i in range(6)]
        pairs, frequency = extra.stability_tables(rows,0,[0,1,2],1e-10)
        self.assertTrue(all(row['top_five_jaccard'] is None for row in pairs))
        self.assertTrue(all(row['top_five_seeds'] is None for row in frequency))

    def test_stability_jaccard_signs_and_frequency(self):
        rows = [dict(seed=s,arm=a,method=m,image_id=f'image{i}',value=float(6-i))
                for s in (0,1,2) for a in extra.ARMS for m in extra.METHODS for i in range(6)]
        pairs, frequency = extra.stability_tables(rows,0,[0,1,2],1e-10)
        self.assertTrue(all(row['top_five_jaccard']==1 and row['sign_agreement']==1 for row in pairs))
        self.assertTrue(all(row['top_five_seeds']==(0 if row['image_id']=='image5' else 3) for row in frequency))

    def test_loo_removes_exactly_one_node_and_tracks_original_negative_class(self):
        images=['a','b','c','d']; calls=[]
        def predict(selected):
            calls.append(list(selected)); p=.3 if len(selected)==4 else .4
            return dict(p_class1=p,yhat=0,logit0=0.,logit1=-1.,edges=0)
        result=extra.loo_analysis({'images':images},self.root,predict)
        self.assertEqual(result['model_queries'],5)
        self.assertEqual(calls,[images]+[[j for j in images if j!=i] for i in images])
        for row in extra.read_rows(self.root/'leave_one_out.csv'):
            self.assertAlmostEqual(float(row['delta_class1']),-.1)
            self.assertAlmostEqual(float(row['delta_original_class']),.1)

    def synthetic_run(self, name, shifted_evaluation):
        root=self.root/name; root.mkdir()
        images=[f'image{i:02}' for i in range(20)]
        for image in images:(root/f'synthetic_{image}.jpg').touch()
        plan=dict(patient='synthetic',images=images,paths={'images':str(root)},
            train_samples=40,evaluation_samples=20,minimum_images=3,ridge_alpha=1.,
            adaptive_target_class1=.5,deletion_fractions=[0,.1,.2,.3,.4,.5],
            streams={'0':extra.study.stream_seeds(0)})
        plan_path=root/'plan.json'; plan_path.write_text(json.dumps(plan))
        directory=root/'run'; directory.mkdir(); count=0
        def predict(selected):
            nonlocal count
            count+=1
            p=.15+.55*('image00' in selected)+.003*len(selected)
            if shifted_evaluation and 103<=count<=122:p+=.05
            return dict(p_class1=p,yhat=int(p>=.5),logit0=0.,logit1=float(np.log(p/(1-p))),edges=0)
        with redirect_stdout(io.StringIO()):result=extra.study.experiment(plan,0,directory,predict)
        result.update(status='passed',seed=0,patient='synthetic',plan_sha256=extra.study.sha256(plan_path))
        (directory/'report.json').write_text(json.dumps(result))
        output=root/'cpu'; output.mkdir()
        settings=dict(alphas=[.001,.01,.1],l1_ratios=[.5,.9],folds=3,tol=1e-5,max_iter=20000)
        extra.cpu_analysis({'elastic_net':settings},{'plan':str(plan_path),'seed':0},plan,directory,output)
        return output

    def test_cpu_analysis_is_unchanged_by_evaluation_response_changes(self):
        first=self.synthetic_run('first',False); second=self.synthetic_run('second',True)
        self.assertEqual((first/'rankings.csv').read_text(),(second/'rankings.csv').read_text())
        for arm in extra.ARMS:
            a=json.loads((first/f'{arm}_elastic_net.json').read_text())
            b=json.loads((second/f'{arm}_elastic_net.json').read_text())
            self.assertEqual(a,b)
            self.assertEqual(a['evaluation_rows_used_for_fit'],0)
        self.assertNotEqual((first/'fidelity.csv').read_text(),(second/'fidelity.csv').read_text())

    def test_summary_requires_all_tasks_and_rejects_changed_evidence(self):
        source=self.synthetic_run('source',False)
        root=self.root/'parallel'; root.mkdir()
        images=[f'image{i:02}' for i in range(20)]
        plan={'scope':'synthetic','patients':[{'plan':'synthetic'} for _ in range(10)],
              'stability':{'sign_zero_tolerance':1e-10},'loo':{'planned_model_queries':210}}
        path=root/'plan.json'; path.write_text(json.dumps(plan))
        for task in range(30):
            directory=root/'cpu'/f'task-{task:02}'; shutil.copytree(source,directory)
            report={'status':'passed','task':task,'command':'cpu','patient_index':task//3,'seed':task%3,
                'plan_sha256':extra.study.sha256(path),'input_hashes':[],'model_queries':0,
                'design':{a:extra.design_diagnostics(np.ones((3,20)),np.ones(3),images)[0] for a in extra.ARMS},
                'output_hashes':[{'path':str(p),'sha256':extra.study.sha256(p)} for p in directory.iterdir()]}
            (directory/'report.json').write_text(json.dumps(report))
        for task in range(10):
            directory=root/'loo'/f'task-{task:02}'; directory.mkdir(parents=True)
            result=extra.loo_analysis({'images':images},directory,
                lambda selected:dict(p_class1=.8,yhat=1,logit0=0.,logit1=1.,edges=0))
            report={'status':'passed','task':task,'command':'loo','patient_index':task,'seed':0,
                'plan_sha256':extra.study.sha256(path),**result,
                'output_hashes':[{'path':str(p),'sha256':extra.study.sha256(p)} for p in directory.iterdir()]}
            (directory/'report.json').write_text(json.dumps(report))
        with patch.object(extra,'verified',return_value=plan), \
             patch.object(extra.study,'verified_plan',return_value={'images':images}):
            extra.summarize(SimpleNamespace(plan=path))
            analysis=json.loads((root/'summary/analysis.json').read_text())
            self.assertEqual(analysis['validated_cpu_tasks'],30)
            self.assertEqual(analysis['loo_model_queries'],210)
            evidence=root/'cpu/task-29/rankings.csv'
            evidence.write_text(evidence.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                extra.summarize(SimpleNamespace(plan=path))


if __name__=='__main__':unittest.main()
