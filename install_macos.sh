# Visual Analytics System - macOS Installation Script
# Run this with: bash install_macos.sh

echo "=========================================="
echo "Visual Analytics System - macOS Installer"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

echo ""
echo "Step 1: Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 2: Installing OpenCV (this must come FIRST)..."
pip install opencv-python==4.8.1.78

echo ""
echo "Step 3: Installing Core Framework..."
pip install fastapi==0.109.0
pip install uvicorn[standard]==0.27.0
pip install python-multipart==0.0.6
pip install pydantic==2.5.3
pip install pydantic-settings==2.1.0

echo ""
echo "Step 4: Installing Computer Vision packages..."
pip install numpy==1.24.3
pip install pillow==10.2.0

echo ""
echo "Step 5: Installing PyTorch..."
pip install torch==2.1.2
pip install torchvision==0.16.2

echo ""
echo "Step 6: Installing YOLOv8..."
pip install ultralytics==8.1.0

echo ""
echo "Step 7: Installing Tracking packages..."
pip install filterpy==1.4.5
pip install scikit-learn==1.3.2
pip install scipy==1.11.4

echo ""
echo "Step 8: Installing Database..."
pip install sqlalchemy==2.0.25
pip install aiosqlite==0.19.0

echo ""
echo "Step 9: Installing Utilities..."
pip install python-jose[cryptography]==3.3.0
pip install python-dotenv==1.0.0
pip install aiofiles==23.2.1

echo ""
echo "Step 10: Installing EasyOCR (may take a while, opencv already installed so should skip opencv-python-headless)..."
pip install easyocr==1.7.1 --no-deps
pip install easyocr==1.7.1

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python3 << 'EOF'
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except:
    print("✗ OpenCV failed")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except:
    print("✗ PyTorch failed")

try:
    import ultralytics
    print("✓ Ultralytics: OK")
except:
    print("✗ Ultralytics failed")

try:
    import fastapi
    print("✓ FastAPI: OK")
except:
    print("✗ FastAPI failed")

try:
    import easyocr
    print("✓ EasyOCR: OK")
except:
    print("✗ EasyOCR failed")

print("\nCore packages installed!")
EOF

echo ""
echo "Next steps:"
echo "1. cd frontend"
echo "2. pip install -r requirements.txt"
echo "3. cd ../backend"
echo "4. python -m app.main"
