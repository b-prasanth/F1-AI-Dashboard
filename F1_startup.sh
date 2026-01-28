#!/bin/bash

# Script to start or kill F1 processes

action="$1"

if [ "$action" == "start" ]; then

  # Create logs directory if it doesn't exist
  mkdir -p logs

  # Activate venv
  source venv/bin/activate

  # Start F1 data update and Plot Generation
  cd backend
  python update_f1_stats.py >> ../logs/F1_startup_f1-data-update.log 2>&1 &
  python plot_generation.py --all >> ../logs/F1_startup-plot-generation.log 2>&1 &

  # Start API servers
  cd ../api
  python api1.py >> ../logs/F1_startup-api1.log 2>&1 &
  python api2.py >> ../logs/F1_startup-api2.log 2>&1 &
  python api3.py >> ../logs/F1_startup-api3.log 2>&1 &



  # Start React development server
  cd ../f1-ui && npm start > ../logs/F1_startup-ui.log 2>&1 &
elif [ "$action" == "kill" ]; then
  # Kill API servers
  pkill -f api1.py
  pkill -f api2.py
  pkill -f api3.py

  # Kill React development server
  pkill -f "npm start"
else
  echo "Usage: $0 [start|kill]"
  exit 1
fi

exit 0
