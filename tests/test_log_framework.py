import sys,json,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from log_framework import choose,serialize,parse_prediction
class FrameworkTests(unittest.TestCase):
    def test_serialized_budget_and_original_ids(self):
        steps=[dict(step=i*3,content='中'*i*20) for i in range(12)]
        for policy in ('prefix','suffix','spread'):
            for cap in (2,100,500,6000):
                selected=choose(steps,policy,cap)
                self.assertLessEqual(len(serialize(selected)),cap)
                self.assertEqual([s['step'] for s in selected],sorted(s['step'] for s in selected))
                self.assertTrue(all(s in steps for s in selected))
    def test_large_step_empty_is_explicit(self):
        self.assertEqual(choose([dict(step=0,content='x'*1000)],'prefix',100),[])
    def test_full_unmodified(self):
        steps=[dict(step=8,content='x'*200)]
        self.assertEqual(choose(steps,'full',2),steps)
    def test_no_invalid_prediction_coercion(self):
        for text in ('{"step":true,"agent":"a"}','{"step":"1","agent":"a"}','{"step":5,"agent":"a"}','{"step":null,"agent":"a"}'):
            with self.assertRaises(ValueError):parse_prediction(text,5)
        self.assertEqual(parse_prediction('{"step":null,"agent":null}',5)['step'],None)
    def test_split_groups_disjoint(self):
        p=Path(__file__).resolve().parents[1]/'data/processed/log_study_v1/inputs.json'
        if not p.exists(): self.skipTest('Prepare upstream benchmark inputs for this integration check')
        rows=json.loads(p.read_text(encoding='utf-8'))
        a={r['group'] for r in rows if r['split']=='development'};b={r['group'] for r in rows if r['split']=='test'}
        self.assertFalse(a&b);self.assertEqual((len(a),len(b)),(25,112))
if __name__=='__main__':unittest.main()
