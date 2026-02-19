#!/bin/bash
# Activate virtual environment and run the DQN multi-agent program

cd "$(dirname "$0")"
source venv/bin/activate

echo "========================================="
echo "DQN Multi-Agent RL - PyTorch Version"
echo "========================================="
echo ""
echo "Choose an environment:"
echo "1) Predators and Prey"
echo "2) Agents and Landmarks"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Running Predators and Prey environment..."
        python predators_prey_multiagent.py "$@"
        ;;
    2)
        echo "Running Agents and Landmarks environment..."
        python agents_landmarks_multiagent.py "$@"
        ;;
    *)
        echo "Invalid choice. Use: ./run.sh or specify directly:"
        echo "  source venv/bin/activate && python predators_prey_multiagent.py"
        echo "  source venv/bin/activate && python agents_landmarks_multiagent.py"
        ;;
esac
