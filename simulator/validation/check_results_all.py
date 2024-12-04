import pandas as pd
from time import time
from collections import defaultdict
# from tqdm import tqdm_notebook as tqdm
from typing import Type, Dict, Any, List
from simulator.simulation.simulate_all import simulate_campaign
from simulator.simulation.modules import Campaign, CampaignInstance, History
from simulator.model.bidder import _Bidder
from simulator.validation.metrics import compile_metrics


def autobidder_check(
    bidder: Type,
    params: Dict[str, Any],
    auction_mode: str = 'VCG',
    mean_click_price: float = 5.0
) -> Dict[str, Any]:
    """
    Perform an automated check of a bidding strategy across multiple campaigns.

    This function simulates campaigns using the provided bidder and compiles performance metrics.

    Args:
        bidder: The bidder class to be used in simulations.
        params: Dictionary containing simulation parameters.
        auction_mode: The auction mode to use. Defaults to 'VCG'.
        category: The category of campaigns to simulate. Defaults to 'Transport'.

    Returns: A dictionary containing simulation results and performance metrics.
    """
    status = "OK!"
    status_msg = ""
    time_all_start = time()

    data_campaigns = pd.read_csv(params["input_campaigns"]).reset_index()  # .sample(3)
    data_campaigns['start_hour'] = pd.to_datetime(data_campaigns.campaign_start, unit='s').dt.hour
    data_stats = pd.read_csv(params["input_stats"])

    time_inf_start = time()
    hist_data_list: List[pd.DataFrame] = []

    campaign_instances = defaultdict(list)  # start_hour : CampaignInstance

    for _, campaign in data_campaigns.iterrows():
        campaign_instance = create_campaign_instance(campaign, mean_click_price, bidder, params)
        campaign_instances[campaign["start_hour"]].append(campaign_instance)
        # bidder_instances.append(bidder_instance)

    min_hour = min(campaign_instances.keys())
    max_hour = max(campaign_instances.keys()) + 24

    for hour in range(min_hour, max_hour + 1):
        T = 0  # active indexes cardinality
        cvr_list = []  # cvr list for winners
        for start_hour, campaign_instances_one_hour in campaign_instances.items():
            # on each step recreate all campaign instances in the current start_hour list
            if (start_hour <= hour) and (start_hour + 24 > hour):
                temp_campaigns = []
                while campaign_instances_one_hour:
                    campaign_inst = campaign_instances_one_hour.pop()
                    # print(campaign_inst.campaign.campaign_id)
                    # < hour + 24 as the campaigns lifetime is 24 hours for any of them
                    campaign_inst = simulate_campaign(
                        campaign_inst=campaign_inst,
                        stats_file=data_stats[data_stats.campaign_id == campaign_inst.campaign_id].copy(),
                        auction_mode=auction_mode,
                        T=T,
                        cvr_list=cvr_list
                    )
                    temp_campaigns.append(campaign_inst)
                T = 0
                cvr_list = []
                for campaign_inst in temp_campaigns:
                    if campaign_inst.campaign.winning:
                        T += 1
                        cvr_list.append(campaign_inst.campaign.campaign_cr)
                campaign_instances[start_hour] = temp_campaigns

    time_inf_end = time()

    for start_hour, campaign_instances_one_hour in campaign_instances.items():
        for campaign_inst in campaign_instances_one_hour:
            hist_data_list.append(campaign_inst.history.to_data_frame())

    metrics, data = compile_metrics(pd.concat(hist_data_list, axis=0, ignore_index=True))

    time_all_end = time()
    return {
        "status": status,
        "status_msg": status_msg,
        "time_overall_sec": time_all_end - time_all_start,
        "time_inference_sec": time_inf_end - time_inf_start,
        "score": metrics,
        'data': data
    }


def create_campaign_instance(campaign: pd.Series,
                             mean_click_price: float,
                             bidder: _Bidder,
                             params: dict) -> CampaignInstance:
    """
    Create a Campaign instance from a pandas Series.
    """
    campaign_id = int(campaign['campaign_id'])
    campaign = Campaign(
        item_id=campaign['item_id'],
        campaign_id=int(campaign['campaign_id']),
        loc_id=int(campaign["loc_id"]),
        region_id=int(campaign["region_id"]),
        logical_category=campaign["logical_category"],
        microcat_ext=int(campaign["microcat_ext"]),
        campaign_start=int(campaign["campaign_start"]),
        campaign_end=int(campaign["campaign_end"]),
        initial_balance=campaign['auction_budget'],
        balance=campaign['auction_budget'],
        curr_time=int(campaign["campaign_start"]),
        prev_time=int(campaign["campaign_start"]),
        prev_balance=campaign['auction_budget'],
        prev_bid=0,
        prev_clicks=0,
        desired_clicks=max(1, campaign['auction_budget'] // mean_click_price),
        desired_time=(int(campaign["campaign_end"]) - int(campaign["campaign_start"])) // 3600
    )

    campaign.curr_time = campaign.campaign_start // 3600 * 3600
    bidder_instance = bidder(params)
    return CampaignInstance(
        campaign_id=int(campaign_id),
        campaign=campaign,
        bidder=bidder_instance,
        history=History()
    )
