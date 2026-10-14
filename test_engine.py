import unittest
import numpy as np
from engine import Payment,scenario,run,GraphPolicy,features

class Invariants(unittest.TestCase):
    def test_all_policies_conserve_cash_and_arrivals(self):
        m=GraphPolicy()
        for mode in ['normal','outage','delay','burst','ring']:
            for p in ['fifo','edf','smallest','cycle','lookahead','graph']:
                sc=scenario(42,mode=mode)
                _,_,trace=run(sc,p,model=m,trace=True)
                for s in trace['states']:
                    self.assertEqual(sum(s['balances']),800)
                    self.assertGreaterEqual(min(s['balances']),0)
                for ev in trace['events']:
                    q=trace['payments'][ev['payment']]
                    self.assertGreaterEqual(ev['time'],q['arrival'])
                    if mode=='outage' and ev['src']==0:self.assertFalse(12<=ev['time']<30)
    def test_no_credit_for_deadlock(self):
        sc=dict(n=2,balances=[0,0],outage=-1,payments=[Payment(0,0,1,10,0,1)])
        metrics,_,_=run(sc)
        self.assertEqual(metrics['settled'],0);self.assertEqual(metrics['missed'],1)
    def test_features_do_not_read_future(self):
        sc=scenario(4);b=np.array(sc['balances']); c=[Payment(100,0,1,10,0,10)]
        before=features(sc,b,0,c)
        for q in sc['payments']:
            if q.arrival>0:q.amount*=100
        after=features(sc,b,0,c)
        for a,z in zip(before,after):np.testing.assert_array_equal(a,z)
    def test_reproducible(self):
        self.assertEqual(run(scenario(55),'edf')[0],run(scenario(55),'edf')[0])

    def test_cycle_can_release_zero_cash_gridlock(self):
        sc=dict(n=2,balances=[0,0],outage=-1,payments=[Payment(0,0,1,10,0,1),Payment(1,1,0,10,0,1)])
        self.assertEqual(run(sc,'fifo')[0]['settled'],0)
        self.assertEqual(run(sc,'cycle')[0]['settled'],2)

if __name__=='__main__':unittest.main()
