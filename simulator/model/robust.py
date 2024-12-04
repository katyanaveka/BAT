import numpy as np
import cvxpy as cp

from typing import Tuple, Dict, Any
from math import sqrt

from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History


class RobustBid(_Bidder):
    default_params = {
        'traffic_path': '../data/traffic_share.csv',
        'eps': 0.01,
        'p': 1.,
        'q': 1.,
        'LP': False
    }

    def __init__(self, params: dict = None):
        super().__init__()

        params = params or {}
        self.traffic = Traffic(path=params.get("traffic_path", self.default_params['traffic_path']))
        self.LP = params.get('LP', self.default_params['LP'])
        self.p = params.get('p', self.default_params['p'])  # gamma
        self.q = params.get('q', self.default_params['q'])  # u_0
        self.C = 10
        eps = params.get('eps', self.default_params['eps'])
        self.alpha = sqrt(2 * eps)

    def place_bid(self, bidding_input_params, history: History) -> float:
        # print(len(history.rows))
        if (len(history.rows) == 1) and self.LP:
            self.p, self.q = self.calculate_pq(bidding_input_params)
        p_q = max(self.p + self.q, 1e-4)
        T = bidding_input_params['T']
        CTR, CVR = bidding_input_params['prev_ctr'], bidding_input_params['prev_cr']
        bid = (CVR + self.q * self.C * CTR) / p_q  # CTR*CVR -> CVR
        cvr_list = bidding_input_params['cvr_list']
        # print('T: ', T)
        # print(f'win: {bidding_input_params["winning"]}')
        if bidding_input_params['winning'] and (len(cvr_list) == 0):
            bid += -self.alpha / p_q * (self.q / sqrt(T))
        elif bidding_input_params['winning']:
            cvr_norm = np.linalg.norm(cvr_list)
            bid += -self.alpha / p_q * (CVR ** 2 / cvr_norm)
        return bid

    def solve_robust_primal(self, CTR, CVR, wp, B, alpha, C, is_win, cvr_list, T):
        n = len(CTR)

        delta = cp.Variable(n, value=-CTR)
        u = cp.Variable(n)
        gamma = cp.Variable(nonneg=True)
        u_0 = cp.Variable(nonneg=True)

        if is_win:
            cvr_norm = np.linalg.norm(cvr_list)
            if cvr_norm > 0:
                delta += alpha * CVR / cvr_norm
            else:
                delta += self.alpha * CVR
            u.value = 1 / np.sqrt(T) * np.ones(n)
        else:
            u.value = np.zeros(n)
        exp = -cp.multiply(delta, CVR) - gamma * wp - alpha * u - u_0 * wp + C * u_0 * CTR
        objective = (gamma * B + cp.sum(cp.maximum(0, exp)))

        constraints = [
            cp.norm(delta + CTR, 2) <= alpha,
            cp.norm(u, 2) <= u_0
        ]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        gamma_opt = gamma.value
        u_0_opt = u_0.value
        return gamma_opt, u_0_opt

    def calculate_pq(self, bidding_input_params: Dict[str, Any]) -> Tuple[float, float]:
        is_win = bidding_input_params['winning']
        T = bidding_input_params['T']
        wp = np.array(bidding_input_params['wp_for_lp'])
        CTR = np.array(bidding_input_params['ctr_for_lp'])
        CVR = np.array(bidding_input_params['cr_for_lp'])
        B = bidding_input_params['balance']
        cvr_list = bidding_input_params['cvr_list']
        cvr_list = np.array(cvr_list).flatten()
        T = bidding_input_params['T']
        # print(CTR, CVR, wp, B, self.alpha, self.C, is_win, cvr_list, T)
        p, q = self.solve_robust_primal(CTR, CVR, wp, B, self.alpha, self.C, is_win, cvr_list, T)
        # print(p, q)
        # print(f'robust p: {p}, q: {q}')
        return p, q
