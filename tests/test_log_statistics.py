import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from log_statistics import group_scores,paired,holm
class StatisticsTests(unittest.TestCase):
    def test_groups_not_trajectory_weighted(self):
        rows=[dict(group='a',id='x',correct=True)]*3+[dict(group='a',id='y',correct=False)]+[dict(group='b',id='z',correct=False)]
        self.assertEqual(group_scores(rows),{'a':.5,'b':0.})
    def test_identical_pairs(self):
        rows=[dict(group='a',id='x',correct=True),dict(group='b',id='y',correct=False)]
        r=paired(rows,rows,draws=100)
        self.assertEqual(r['difference'],0);self.assertEqual(r['bootstrap95'],[0,0]);self.assertEqual(r['sign_flip_p'],1)
    def test_mismatch_rejected(self):
        with self.assertRaises(ValueError):paired([],[])
    def test_holm_reference(self):
        self.assertEqual(holm([.04,.01,.03]),[.06,.03,.06])
if __name__=='__main__':unittest.main()
