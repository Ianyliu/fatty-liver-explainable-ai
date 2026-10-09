"""Strict original-hardware LOO recovery, without changing frozen inference."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import full_secondary_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def test_probability_class_and_edges_are_all_required(self):
        reference={'p_class1':.7,'yhat':1,'edges':20}
        self.assertTrue(recovery.match_original(dict(reference),reference))
        self.assertTrue(recovery.match_original(dict(reference,p_class1=.7000001),reference))
        for bad in [dict(reference,p_class1=.7001),dict(reference,yhat=0),dict(reference,edges=22)]:
            self.assertFalse(recovery.match_original(bad,reference))

    def test_only_failed_full_set_comparisons_are_replaced(self):
        reference={'p_class1':.7,'yhat':1,'edges':20}
        records=[{'patient_index':i,'plan':f'/private/{i}/plan.json','directory':f'/old/{i}',
                  'task':i,'source_patient_index':i,'seed':0,'reused':i==0,'source_plan':'/old/plan.json'} for i in range(3)]
        reports=[{'original_prediction':dict(reference)},
                 {'original_prediction':dict(reference,p_class1=.71)},
                 {'original_prediction':dict(reference,edges=22)}]
        primary={'original_prediction':reference,'host':'original-node.example','gpu_name':'NVIDIA RTX A5000'}
        with patch.object(recovery.base,'check_record',side_effect=reports),patch.object(recovery,'primary_report',return_value=primary):
            output,tasks=recovery.mapping({'loo_records':records},Path('/recovery'))
        self.assertEqual(output[0],records[0]);self.assertEqual(len(tasks),2)
        self.assertEqual([t['patient_index'] for t in tasks],[1,2])
        self.assertEqual([t['task'] for t in tasks],[0,1])
        self.assertTrue(all(t['primary_host']=='original-node' and t['required_gpu_name']=='NVIDIA RTX A5000' and 'amp01' in t['allowed_hosts'] for t in tasks))
        self.assertEqual(records[1]['directory'],'/old/1')

    def test_same_node_mismatch_stops_before_image_removals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan_path=root/'plan.json';plan_path.write_text('{}')
            child_path=root/'child/plan.json';child_path.parent.mkdir();child_path.write_text('{}')
            primary_path=child_path.parent/'runs/seed-0/report.json';primary_path.parent.mkdir(parents=True);primary_path.write_text('{}')
            item={'patient_index':1,'plan':str(child_path),'directory':str(root/'loo/task-000'),
                  'original_directory':'/original','primary_host':'original','allowed_hosts':['original','same-model-peer'],'required_gpu_name':'gpu'}
            primary={'original_prediction':{'p_class1':.7,'yhat':1,'edges':20}}
            predictor=Mock(return_value={'p_class1':.71,'yhat':1,'edges':20});predictor.torch.cuda.get_device_name.return_value='gpu'
            with patch.object(recovery,'verified',return_value={'loo_tasks':[item]}),patch.object(recovery,'primary_report',return_value=primary),patch.object(recovery.study,'verified_plan',return_value={'images':['a','b','c','d']}),patch.object(recovery.study,'PatientPredictor',return_value=predictor),patch.object(recovery.socket,'gethostname',return_value='original.example'):
                with self.assertRaisesRegex(ValueError,'Same-model full-set'):
                    recovery.run(SimpleNamespace(plan=plan_path,task=0))
            self.assertEqual(predictor.call_count,1)
            report=json.loads((Path(item['directory'])/'report.json').read_text())
            self.assertEqual(report['status'],'failed');self.assertEqual(report['actual_model_queries'],1)
            self.assertFalse((Path(item['directory'])/'leave_one_out.csv').exists())


if __name__=='__main__':unittest.main()
