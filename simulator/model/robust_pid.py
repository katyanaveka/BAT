import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import linprog

from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History
from simulator.model.m_pid import MPIDBidder

class RobustBidder(MPIDBidder):
    def __init__(self, params: dict = None):
        super().__init__()
    
    def place_bid(self, bidding_input_params, history: History) -> float:
        # Calculate budget pace from step 0 to current
        budget_pace = self.budget_pace_count(bidding_input_params)
        self.reference[0] = budget_pace
        self.last_click = bidding_input_params['clicks'] - bidding_input_params['prev_clicks']
        self.u = self.reference
        if len(history.rows) == 0:
            # Cold start: return a simple initial bid
            return self.init_bid
        elif len(history.rows) == 1:
            # Chilly start: use LP solver
            p, q = self.calculate_pq(bidding_input_params)
            self.x_0 = np.array([p, q])
        else:
            # Regular bidding process
            spent = bidding_input_params['prev_balance'] - bidding_input_params['balance']
            cur_cpc = ((bidding_input_params['initial_balance'] - bidding_input_params['balance']) /
                    bidding_input_params['clicks']) if bidding_input_params['clicks'] > 0 else 0
            y = np.array([spent, cur_cpc])
            self.click = bidding_input_params['clicks']
            p, q = self.pid_compute(self.reference, y)

        CTR, CVR = bidding_input_params['prev_ctr'], bidding_input_params['prev_cr']
        bid = max(self.bid_compute(p, q, self.C, CTR, CVR), self.min_bid)
        mean_bid = sum(self.prev_bids) / len(self.prev_bids)
        bid = np.clip(bid, self.lower_clip * mean_bid, self.upper_clip * mean_bid)
        self.prev_bids.append(bid)
        return bid
 
    def pid_compute(self, reference, y) -> Tuple[float, float]:
        cur_error = reference - y  # 2d array

        if self.last_click:
            cur_error[1] *= self.last_click
        self.error = np.append(self.error, [cur_error], axis=0)

        k_i = self.k_dict['k_i']
        k_p = self.k_dict['k_p']
        k_d = self.k_dict['k_d']

        u = k_p * self.error[-1] + k_i * sum(self.error) +\
         (k_d * (self.error[-1] - self.error[-2]) if len(self.error) > 1 else 0)
        if self.click:
            u[1] /= self.click
        u = self.apply_alpha(u)
        x = self.x_0 * np.exp(-np.clip(u, -700, 700))
        self.u = u
        return x
    
    def bid_compute(self, gamma: float, u_0: float, C: float, CTR: float, CVR: float, is_active: bool) -> float:
        g_u = max(gamma + u_0, 1e-4)
        bid = (CVR + CTR * C * u_0) / g_u # CTR * CVR -> CVR
        if is_active:
            bid += - self.alpha / g_u * (u_0 / )
        return self.cold_start_coef * bid
