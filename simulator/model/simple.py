import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import linprog
from math import sqrt

from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class SimpleBid(_Bidder):
    default_params = {
        'traffic_path': '../data/traffic_share.csv',
        'eps': 0.01,
        'p': 1,
        'q': 1,
        'LP': False
    }

    def __init__(self, params: dict = None):
        super().__init__()

        params = params or {}
        self.traffic = Traffic(path=params.get("traffic_path", self.default_params['traffic_path']))
        self.LP = params.get('LP', self.default_params['LP'])
        self.p = params.get('p', self.default_params['p'])  # gamma
        self.q = params.get('q', self.default_params['q'])  # u_0
        eps = params.get('eps', self.default_params['eps'])
        self.alpha = sqrt(2 * eps)
        self.C = 100

    def place_bid(self, bidding_input_params, history: History) -> float:
        # print(len(history.rows))
        if (len(history.rows) == 1) and self.LP:
            self.p, self.q = self.calculate_pq(bidding_input_params)
        p_q = max(self.p + self.q, 1e-4)
        CTR, CVR = bidding_input_params['prev_ctr'], bidding_input_params['prev_cr']
        bid = (CVR + self.q * self.C * CTR) / p_q  # CTR * CVR -> CVR
        # print(bid)
        return bid

    def calculate_pq(self, bidding_input_params: Dict[str, Any]) -> Tuple[float, float]:
        # Section Problem formulation, linear programming problem
        wp = bidding_input_params['wp_for_lp']
        ctr = bidding_input_params['ctr_for_lp']
        cvr = bidding_input_params['cr_for_lp']
        B = bidding_input_params['balance']

        C, N = self.C, cvr.shape[0]
        c = np.ones(N + 2)
        c[-2], c[-1] = B, 0
        A_up = -1 * np.vstack((np.vstack((np.eye(N), wp)), wp - ctr * C)).T
        res = linprog(c=c, A_ub=A_up, b_ub=-np.array(cvr), bounds=(0, None))
        p, q = res.x[-2], res.x[-1]
        # print(f'simple p: {p}, q: {q}')
        return max(p, 0.1), max(q, 0.1)
