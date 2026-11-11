"""Exercise the decisions that prevent another unbounded FID regression."""
import json
import unittest

from dynamics import evaluation_due,initial_state,learning_rate,observe

POLICY=dict(source_step=42536,first_evaluation_updates=100,evaluation_every_updates=500,
    maximum_relative_regression=.02,minimum_improvement=.005,
    reduce_after_nonimproving_checks=2,stop_after_nonimproving_checks=4,
    lr_reduction_factor=.5)
SCHEDULE=dict(start_step=42536,decay_updates=5000,maximum=1e-5,
              minimum=2e-6,absolute_floor=5e-7)


class DynamicsTests(unittest.TestCase):
    def test_real_observed_regression_stops(self):
        state,decision=observe(POLICY,initial_state(),2.4862945044553726,3.057921744103396)
        self.assertEqual(decision['action'],'stop_regression')
        self.assertAlmostEqual(decision['best_rfid'],2.4862945044553726)

    def test_plateau_reduces_then_stops_and_improvement_resets(self):
        state=initial_state()
        for i in range(2):state,decision=observe(POLICY,state,2.4863,2.50)
        self.assertEqual(decision['action'],'reduce_learning_rate')
        self.assertEqual(state['lr_factor'],.5)
        recovered=json.loads(json.dumps(state))
        self.assertEqual(learning_rate(SCHEDULE,state,43036),learning_rate(SCHEDULE,recovered,43036))
        for i in range(2):state,decision=observe(POLICY,state,2.4863,2.50)
        self.assertEqual(decision['action'],'stop_plateau')
        state,decision=observe(POLICY,recovered,2.4863,2.46)
        self.assertEqual(decision['action'],'continue')
        self.assertEqual(state['nonimproving_checks'],0)

    def test_schedule_and_evaluation_boundaries(self):
        state=initial_state()
        self.assertAlmostEqual(learning_rate(SCHEDULE,state,42536),1e-5)
        self.assertAlmostEqual(learning_rate(SCHEDULE,state,47536),2e-6)
        state['lr_factor']=.01
        self.assertAlmostEqual(learning_rate(SCHEDULE,state,50000),5e-7)
        self.assertEqual([s for s in range(42536,43537) if evaluation_due(POLICY,s)],
                         [42636,43036,43536])


if __name__=='__main__':unittest.main()
