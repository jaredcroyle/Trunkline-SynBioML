#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

APP_BUNDLE="dist/Trunkline.app"
PYTHON_LIB_PATH="/Users/jcroyle/anaconda3/lib/libpython3.11.dylib"

# Check if app bundle exists
if [ ! -d "$APP_BUNDLE" ]; then
    echo -e "${RED}Error: App bundle not found at $APP_BUNDLE${NC}"
    exit 1
fi

echo -e "${GREEN}=== Fixing Trunkline App Bundle ===${NC}"

# Create Frameworks directory if it doesn't exist
mkdir -p "$APP_BUNDLE/Contents/Frameworks"

# Copy Python library to Frameworks
echo -e "${YELLOW}Copying Python library to app bundle...${NC}"
if [ -f "$PYTHON_LIB_PATH" ]; then
    echo "Found Python library at $PYTHON_LIB_PATH"
    cp "$PYTHON_LIB_PATH" "$APP_BUNDLE/Contents/Frameworks/"
    
    # Update the Python library's install name
    echo "Updating library install name..."
    install_name_tool -id "@executable_path/../Frameworks/libpython3.11.dylib" \
                     "$APP_BUNDLE/Contents/Frameworks/libpython3.11.dylib"
    
    # Update the executable's reference to the Python library
    echo "Updating executable's library references..."
    install_name_tool -change "$PYTHON_LIB_PATH" \
                     "@executable_path/../Frameworks/libpython3.11.dylib" \
                     "$APP_BUNDLE/Contents/MacOS/trunkline"
    
    # Verify the changes
    echo -e "${YELLOW}Verifying library references...${NC}"
    echo "Executable dependencies:"
    otool -L "$APP_BUNDLE/Contents/MacOS/trunkline" | grep -E 'python|Python'
    
    echo -e "\nPython library dependencies:"
    otool -L "$APP_BUNDLE/Contents/Frameworks/libpython3.11.dylib" | head -n 5
else
    echo -e "${RED}Error: Python library not found at $PYTHON_LIB_PATH${NC}"
    exit 1
fi

# Remove any quarantine attributes
echo -e "${YELLOW}Removing quarantine attributes...${NC}"
xattr -cr "$APP_BUNDLE"

# Sign the application
echo -e "${YELLOW}Signing application...${NC}"
codesign --force --deep --sign - "$APP_BUNDLE"

# Verify the signature
echo -e "${YELLOW}Verifying code signature...${NC}"
codesign -dv --verbose=4 "$APP_BUNDLE"

echo -e "${GREEN}=== App bundle fixed successfully! ===${NC}"
echo "You can now run the application with: open $APP_BUNDLE"
