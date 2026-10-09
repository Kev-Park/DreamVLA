import unittest
from pipeline_health import assess


class PipelineHealthTests(unittest.TestCase):
    def test_abandoned_transfer_is_not_healthy(self):
        feeds={'deliveries':{'time':1000,'deliveries':{'pick':'transferring'}}}
        remote={'transfers':{'pick':{'bytes':100,'active':False,'delivered':False}}}
        self.assertTrue(any('no transfer' in a['reason'] for a in assess(1000,feeds,remote,{})))

    def test_incomplete_eval_cannot_be_done(self):
        run={'done':True,'episodes':99,'metric':{'episodes':99,'missing':0,'world':5,'root':4}}
        self.assertTrue(assess(1000,{}, {'runs':{'pick':run}},{}))
        run['episodes']=100;run['metric']['episodes']=100
        self.assertFalse(assess(1000,{}, {'runs':{'pick':run}},{}))
        run['metric']['root']=float('nan')
        self.assertTrue(assess(1000,{}, {'runs':{'pick':run}},{}))

    def test_live_watcher_can_still_have_stuck_training(self):
        history={}
        feed={'time':1000,'pods':[{'id':'p','training_step':90,'current':{'state':'running','run':'pick'}}]}
        self.assertFalse(assess(1000,{'status':feed},{},history))
        feed['time']=1601
        self.assertTrue(any('optimizer-step' in a['reason'] for a in assess(1601,{'status':feed},{},history)))
        feed['pods'][0]['training_step']=91
        self.assertFalse(assess(1602,{'status':feed},{},history))

    def test_missing_watcher_heartbeat_is_visible(self):
        self.assertTrue(any('heartbeat' in a['reason'] for a in assess(1000,{'budget':{}},{},{})))


if __name__=='__main__':
    unittest.main()
