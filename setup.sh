# setup.sh

# Update system packages
sudo apt-get update

# Install Tesseract OCR and dependencies
sudo apt-get install -y tesseract-ocr
sudo apt-get install -y libleptonica-dev

# Verify the installation
which tesseract  # Check if the tesseract binary is in the PATH
tesseract --version  # Verify the installed version
