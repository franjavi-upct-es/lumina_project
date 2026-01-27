#!/bin/bash
# =============================================================================
# Setup NVIDIA Container Toolkit for Docker GPU support
# Enables Docker containers to access the host NVIDIA GPU
# =============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== NVIDIA Container Toolkit Setup ===${NC}"

# 1. Check host GPU
echo -e "\n${YELLOW}[1/5] Checking host NVIDIA GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Install NVIDIA drivers first.${NC}"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo -e "${GREEN}Host GPU detected.${NC}"

# 2. Check if toolkit is already installed
echo -e "\n${YELLOW}[2/5] Checking nvidia-container-toolkit...${NC}"
if command -v nvidia-ctk &> /dev/null; then
    echo -e "${GREEN}nvidia-container-toolkit is already installed.${NC}"
    nvidia-ctk --version
else
    echo "Installing nvidia-container-toolkit..."

    # Add NVIDIA GPG key and repo
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    echo -e "${GREEN}nvidia-container-toolkit installed.${NC}"
fi

# 3. Configure Docker runtime
echo -e "\n${YELLOW}[3/5] Configuring Docker NVIDIA runtime...${NC}"
sudo nvidia-ctk runtime configure --runtime=docker
echo -e "${GREEN}Docker runtime configured.${NC}"

# 4. Restart Docker
echo -e "\n${YELLOW}[4/5] Restarting Docker daemon...${NC}"
sudo systemctl restart docker
echo -e "${GREEN}Docker restarted.${NC}"

# 5. Verify
echo -e "\n${YELLOW}[5/5] Verifying GPU access from Docker...${NC}"
if docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi; then
    echo -e "${GREEN}Docker GPU access verified successfully.${NC}"
else
    echo -e "${RED}Verification failed. Check the output above for errors.${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== Setup complete. You can now run: make up-gpu ===${NC}"
