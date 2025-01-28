# BAT: Benchmark for Auto-bidding Task
Anonymized submission


├── data/
│ ├── subsample_campaigns.csv
│ └── subsample_stats.csv
├── notebooks/
│ ├── baseline_bidders.ipynb
│ ├── bidder_example.ipynb
│ ├── pictures_rebuttal.ipynb
│ ├── filter-fpa.ipynb
│ └── filter-vcg.ipynb
├── src/
│ ├── bidders/
│ │ ├── bidder.py
│ │ ├── broi_bidder.py
│ │ ├── linear_bidder.py
│ │ ├── m_pid.py
│ │ ├── mystique.py
│ │ ├── slivkins_bidder.py
│ │ └── ta_pid.py
│ ├── simulation/
│ │ ├── modules.py
│ │ ├── simulate.py
│ │ └── traffic.py
│ └── utils/
│ ├── check_results.py
│ ├── metrics.py
│ ├── utils.py
│ └── utils_visualization.py
├── .flake8
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── traffic_share.csv
