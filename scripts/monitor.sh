#!/bin/bash
# PhaseGPT Production Monitor
# Usage: ./scripts/monitor.sh

LOG_FILE="training.log"
PID_FILE="training.pid"

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${BLUE}===============================================================${NC}"
    echo -e "${BLUE}   PHASEGPT v1.4 - ORACLE TRAINING DASHBOARD (Mac Studio)      ${NC}"
    echo -e "${BLUE}===============================================================${NC}"
    
    # 1. Process Status
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            # Get resource usage specific to the python process
            RESOURCE_USAGE=$(ps -o %cpu,%mem,time -p $PID | tail -n 1)
            STATUS="${GREEN}RUNNING${NC}"
            CPU=$(echo $RESOURCE_USAGE | awk '{print $1}')
            MEM=$(echo $RESOURCE_USAGE | awk '{print $2}')
            TIME=$(echo $RESOURCE_USAGE | awk '{print $3}')
        else
            STATUS="${RED}STOPPED${NC} (Process $PID not found)"
            CPU="0.0"
            MEM="0.0"
            TIME="00:00:00"
        fi
    else
        STATUS="${YELLOW}UNKNOWN${NC} (No PID file)"
    fi

    echo -e "STATUS: $STATUS   PID: $PID"
    echo -e "CPU: ${YELLOW}${CPU}%${NC}   MEM: ${YELLOW}${MEM}%${NC}   TIME: ${YELLOW}${TIME}${NC}"
    echo -e "${BLUE}---------------------------------------------------------------${NC}"
    
    # 2. Training Metrics Parsing
    if [ -f "$LOG_FILE" ]; then
        # Extract latest loss line: "  Epoch 1 | Step 20 | Loss: 1.5778"
        LATEST_LINE=$(grep "Loss:" "$LOG_FILE" | tail -n 1)
        
        if [ ! -z "$LATEST_LINE" ]; then
            EPOCH=$(echo "$LATEST_LINE" | awk '{print $2}')
            STEP=$(echo "$LATEST_LINE" | awk '{print $5}')
            LOSS=$(echo "$LATEST_LINE" | awk '{print $8}')
            
            echo -e "CURRENT EPOCH: ${GREEN}$EPOCH${NC}"
            echo -e "CURRENT STEP:  ${GREEN}$STEP${NC}"
            echo -e "CURRENT LOSS:  ${GREEN}$LOSS${NC}"
            
            # Simple Textual Progress Bar based on Loss (Lower is better)
            # Assuming starting loss ~10, target ~0.5
            # Inverting logic for visualization
            echo ""
            echo "Convergence Visualizer:"
            if (( $(echo "$LOSS > 10" | bc -l) )); then
                echo -e "[${RED}====================${NC}] High Loss (Early Training)"
            elif (( $(echo "$LOSS > 5" | bc -l) )); then
                 echo -e "[${YELLOW}==========          ${NC}] Learning..."
            elif (( $(echo "$LOSS > 1" | bc -l) )); then
                 echo -e "[${GREEN}=====               ${NC}] Converging..."
            else
                 echo -e "[${BLUE}=                   ${NC}] OPTIMAL (< 1.0)"
            fi
        else
            echo "Waiting for first training step..."
            echo "(Dataset generation or model loading in progress)"
        fi
    else
        echo "Log file not found."
    fi
    
    echo -e "${BLUE}---------------------------------------------------------------${NC}"
    echo "RECENT LOGS:"
    tail -n 5 "$LOG_FILE"
    
    echo -e "${BLUE}===============================================================${NC}"
    echo "Press [CTRL+C] to exit dashboard (Training continues)"
    
    sleep 2
done
