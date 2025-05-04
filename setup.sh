# setup.sh

# Update system packages
sudo apt-get update

# Install Tesseract OCR
sudo apt-get install -y tesseract-ocr

# Verify Tesseract installation
tesseract --version