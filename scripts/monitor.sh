#!/bin/bash
# PhaseGPT Production Monitor v1.5
# Usage: ./scripts/monitor.sh

LOG_FILE="training.log"
PID_FILE="training.pid"

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper to format seconds
format_time() {
    ((h=${1}/3600))
    ((m=(${1}%3600)/60))
    ((s=${1}%60))
    printf "%02d:%02d:%02d\n" $h $m $s
}

while true; do
    clear
    echo -e "${BLUE}===============================================================${NC}"
    echo -e "${BLUE}   PHASEGPT v1.4 - ORACLE TRAINING DASHBOARD (Mac Studio)      ${NC}"
    echo -e "${BLUE}===============================================================${NC}"
    
    # --- 1. System & Process Health ---
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            # Get start time and resource usage
            # ps output format: %cpu, %mem, etime (elapsed time)
            STATS=$(ps -o %cpu,%mem,etime -p $PID | tail -n 1)
            CPU=$(echo $STATS | awk '{print $1}')
            MEM=$(echo $STATS | awk '{print $2}')
            ELAPSED=$(echo $STATS | awk '{print $3}')
            
            STATUS="${GREEN}RUNNING${NC}"
            
            # Get System Load Average
            LOAD=$(uptime | awk -F'load averages:' '{ print $2 }')
        else
            STATUS="${RED}STOPPED${NC} (Process $PID not found)"
            CPU="0.0"
            MEM="0.0"
            ELAPSED="--"
            LOAD="--"
        fi
    else
        STATUS="${YELLOW}UNKNOWN${NC} (No PID file)"
    fi

    echo -e "STATUS: $STATUS   PID: $PID   UPTIME: $ELAPSED"
    echo -e "CPU: ${YELLOW}${CPU}%${NC}   MEM: ${YELLOW}${MEM}%${NC}   LOAD: ${CYAN}${LOAD}${NC}"
    echo -e "${BLUE}---------------------------------------------------------------${NC}"
    
    # --- 2. Training Phase Detection ---
    PHASE="Initializing..."
    if [ -f "$LOG_FILE" ]; then
        if grep -q "Generating Golden Dataset" "$LOG_FILE"; then
            PHASE="${YELLOW}Building Dataset (10k samples)${NC}"
        fi
        if grep -q "Initializing VolitionalTrainer" "$LOG_FILE"; then
            PHASE="${CYAN}Initializing Trainer & Model${NC}"
        fi
        if grep -q "Starting training" "$LOG_FILE"; then
            PHASE="${GREEN}Training Loop Active${NC}"
        fi
        if grep -q "Saving Artifacts" "$LOG_FILE"; then
            PHASE="${BLUE}Saving Final Model${NC}"
        fi
    fi
    echo -e "CURRENT PHASE: $PHASE"
    
    # --- 3. Training Metrics ---
    if [ -f "$LOG_FILE" ]; then
        # Grep all loss lines
        LOSS_LINES=$(grep "Loss:" "$LOG_FILE" | tail -n 5)
        
        if [ ! -z "$LOSS_LINES" ]; then
            # Get latest values
            LATEST=$(echo "$LOSS_LINES" | tail -n 1)
            EPOCH=$(echo "$LATEST" | awk '{print $2}')
            STEP=$(echo "$LATEST" | awk '{print $5}')
            LOSS=$(echo "$LATEST" | awk '{print $8}')
            
            echo -e "EPOCH: ${GREEN}$EPOCH${NC}   STEP: ${GREEN}$STEP${NC}   LOSS: ${GREEN}$LOSS${NC}"
            
            echo -e "\n${CYAN}Loss Trend (Last 5 Updates):${NC}"
            echo "$LOSS_LINES" | awk '{print "  Step " $5 ": " $8}'
            
            # Progress Bar logic
            echo -e "\n${CYAN}Convergence Status:${NC}"
            MAX_LOSS=10.0
            WIDTH=20
            # Simple bash math for progress bar
            BAR_LEN=$(echo "scale=0; ($MAX_LOSS - $LOSS) * $WIDTH / $MAX_LOSS" | bc -l 2>/dev/null)
            if [ -z "$BAR_LEN" ] || (( $(echo "$BAR_LEN < 0" | bc -l) )); then BAR_LEN=0; fi
            if (( $(echo "$BAR_LEN > $WIDTH" | bc -l) )); then BAR_LEN=$WIDTH; fi
            
            printf "High Loss ["
            for ((i=0; i<$BAR_LEN; i++)); do printf "#"; done
            for ((i=$BAR_LEN; i<$WIDTH; i++)); do printf "."; done
            printf "] Low Loss\n"
            
        else
            echo -e "\n${YELLOW}Waiting for first gradient update...${NC}"
        fi
        
        # Check for OOM
        if grep -q "out of memory" "$LOG_FILE"; then
            echo -e "\n${RED}!!! CRITICAL WARNING: OOM DETECTED IN LOGS !!!${NC}"
        fi
    fi
    
    echo -e "${BLUE}---------------------------------------------------------------${NC}"
    echo "LATEST LOG OUTPUT:"
    tail -n 3 "$LOG_FILE"
    
    echo -e "${BLUE}===============================================================${NC}"
    echo "Press [CTRL+C] to exit dashboard (Training continues)"
    
    sleep 2
done