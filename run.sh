#!/bin/bash
# Activate virtual environment and run the DQN multi-agent program

cd "$(dirname "$0")"
source .venv/bin/activate

echo "========================================="
echo "DQN Multi-Agent RL - Car Game"
echo "========================================="
echo ""
echo "Choose implementation:"
echo "1) Independent DQN (2 separate agents)"
echo "2) Nash-Q Learning (joint Q-values)"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Running Independent DQN..."
        python car_game_multiagent.py "$@"
        ;;
    2)
        echo "Running Nash-Q Learning..."
        python car_game_nashq.py "$@"
        ;;
    *)
        echo "Invalid choice. Use: ./run.sh or specify directly:"
        echo "  source .venv/bin/activate && python car_game_multiagent.py"
        echo "  source .venv/bin/activate && python car_game_nashq.py"
        ;;
esac
