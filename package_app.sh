#!/bin/bash
# Package the Trunkline ML application for macOS

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Packaging Trunkline ML Application ===${NC}"

# Check for Xcode command line tools
if ! xcode-select -p &>/dev/null; then
    echo -e "${YELLOW}Xcode command line tools are not installed. Installing...${NC}"
    xcode-select --install
    echo -e "${GREEN}Please complete the Xcode installation and run this script again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install pyinstaller==5.13.0 pillow pyobjc-core pyobjc
    deactivate
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Convert icon to ICNS if needed
if [ ! -f "trunkline.icns" ]; then
    echo "Converting icon to ICNS format..."
    python convert_to_icns.py trunkline.png
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ Trunkline.app Trunkline_ML.dmg

# Ensure the Python library is accessible
PYTHON_LIB=$(python -c "import sys; print(sys.executable)")
if [ ! -f "$PYTHON_LIB" ]; then
    echo -e "${RED}Error: Python library not found at $PYTHON_LIB${NC}"
    exit 1
fi

# Build the application
echo "Building application..."
pyinstaller --clean --noconfirm trunkline.spec

# Verify the app bundle
echo -e "${GREEN}Verifying app bundle...${NC}"
if [ -d "dist/Trunkline.app" ]; then
    echo -e "${GREEN}Successfully built Trunkline.app${NC}"
    echo "Location: $(pwd)/dist/Trunkline.app"
    
    # Create Application Support directory
    mkdir -p "$HOME/Library/Application Support/Trunkline/Logs"
    
    # Set proper permissions
    chmod -R 755 "dist/Trunkline.app"
    
    # Remove any extended attributes that might cause issues
    xattr -cr "dist/Trunkline.app"
    
    # Sign the application (ad-hoc signing for development)
    echo "Signing application..."
    codesign --force --deep --sign - "dist/Trunkline.app"
    
    # Verify the signature
    echo "Verifying code signature..."
    codesign -dv --verbose=4 "dist/Trunkline.app"
    
    # Ensure Frameworks directory exists
    mkdir -p "dist/Trunkline.app/Contents/Frameworks"
    
    # Copy Python library to Frameworks
    echo -e "${YELLOW}Copying Python library to app bundle...${NC}"
    PYTHON_LIB_PATH="/Users/jcroyle/anaconda3/lib/libpython3.11.dylib"
    if [ -f "$PYTHON_LIB_PATH" ]; then
        echo "Found Python library at $PYTHON_LIB_PATH"
        cp "$PYTHON_LIB_PATH" "dist/Trunkline.app/Contents/Frameworks/"
        
        # Update the Python library's install name
        echo "Updating library install name..."
        install_name_tool -id "@executable_path/../Frameworks/libpython3.11.dylib" \
                         "dist/Trunkline.app/Contents/Frameworks/libpython3.11.dylib"
        
        # Update the executable's reference to the Python library
        echo "Updating executable's library references..."
        install_name_tool -change "$PYTHON_LIB_PATH" \
                         "@executable_path/../Frameworks/libpython3.11.dylib" \
                         "dist/Trunkline.app/Contents/MacOS/trunkline"
        
        # Verify the changes
        echo "Verifying library references..."
        otool -L "dist/Trunkline.app/Contents/MacOS/trunkline" | grep python
    else
        echo -e "${RED}Error: Python library not found at $PYTHON_LIB_PATH${NC}"
        exit 1
    fi
    
    # Create a DMG for distribution
    read -p "Create DMG for distribution? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating DMG..."
        
        # Create a temporary directory for the DMG
        DMG_TEMP="dist/Trunkline_temp"
        mkdir -p "$DMG_TEMP"
        
        # Copy the app to the temp directory
        cp -R "dist/Trunkline.app" "$DMG_TEMP/"
        
        # Create a symbolic link to Applications
        ln -s /Applications "$DMG_TEMP/Applications"
        
        # Create the DMG
        hdiutil create -volname "Trunkline ML" \
                      -srcfolder "$DMG_TEMP" \
                      -ov \
                      -format UDZO \
                      "Trunkline_ML.dmg"
        
        # Clean up
        rm -rf "$DMG_TEMP"
        
        echo -e "${GREEN}DMG created: $(pwd)/Trunkline_ML.dmg${NC}"
    fi
    
    # Offer to install to Applications
    read -p "Install to Applications folder? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing to Applications..."
        rm -rf "/Applications/Trunkline ML.app"
        cp -R "dist/Trunkline.app" "/Applications/Trunkline ML.app"
        echo -e "${GREEN}Installed to /Applications/Trunkline ML.app${NC}"
        
        # Fix permissions on the installed app
        sudo chown -R $(whoami) "/Applications/Trunkline ML.app"
        xattr -cr "/Applications/Trunkline ML.app"
    fi
    
    # Open the app directly to see any error messages
    read -p "Launch the application now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -d "/Applications/Trunkline ML.app" ]; then
            echo -e "${GREEN}Launching Trunkline ML...${NC}"
            open -a "Trunkline ML" --args --debug
        else
            echo -e "${GREEN}Launching from build directory...${NC}"
            open "dist/Trunkline.app" --args --debug
        fi
        
        # Show the log file location
        LOG_FILE="$HOME/Library/Logs/Trunkline/trunkline.log"
        echo -e "${YELLOW}Check the log file for any errors: $LOG_FILE${NC}"
        
        # Tail the log file
        if [ -f "$LOG_FILE" ]; then
            echo -e "${YELLOW}=== Tail of log file ===${NC}"
            tail -n 20 "$LOG_FILE"
        fi
    fi
else
    echo -e "${RED}Error: Failed to build Trunkline.app${NC}"
    exit 1
fi

echo -e "${GREEN}=== Packaging complete! ===${NC}"
echo -e "${YELLOW}Note: For the best experience, drag Trunkline.app to your Applications folder.${NC}"
