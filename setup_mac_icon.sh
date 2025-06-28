#!/bin/bash

# Create iconset directory
mkdir Trunkline.iconset

# Generate icons at different sizes
sips -z 16 16     trunkline.png --out Trunkline.iconset/icon_16x16.png
sips -z 32 32     trunkline.png --out Trunkline.iconset/icon_16x16@2x.png
sips -z 32 32     trunkline.png --out Trunkline.iconset/icon_32x32.png
sips -z 64 64     trunkline.png --out Trunkline.iconset/icon_32x32@2x.png
sips -z 128 128   trunkline.png --out Trunkline.iconset/icon_128x128.png
sips -z 256 256   trunkline.png --out Trunkline.iconset/icon_128x128@2x.png
sips -z 256 256   trunkline.png --out Trunkline.iconset/icon_256x256.png
sips -z 512 512   trunkline.png --out Trunkline.iconset/icon_256x256@2x.png
sips -z 512 512   trunkline.png --out Trunkline.iconset/icon_512x512.png
sips -z 1024 1024 trunkline.png --out Trunkline.iconset/icon_512x512@2x.png

# Create .icns file
echo "Creating .icns file..."
iconutil -c icns Trunkline.iconset -o Trunkline.icns

# Clean up
rm -R Trunkline.iconset

echo "Icon file created: Trunkline.icns"

# Create app bundle structure if it doesn't exist
if [ ! -d "Trunkline.app" ]; then
    echo "Creating app bundle..."
    mkdir -p "Trunkline.app/Contents/MacOS"
    mkdir -p "Trunkline.app/Contents/Resources"
    
    # Create Info.plist
    cat > "Trunkline.app/Contents/Info.plist" <<EOL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>trunkline</string>
    <key>CFBundleIconFile</key>
    <string>Trunkline.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.trunkline.ml</string>
    <key>CFBundleName</key>
    <string>Trunkline</string>
    <key>CFBundleDisplayName</key>
    <string>Trunkline ML</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOL
    
    # Create launcher script
    cat > "Trunkline.app/Contents/MacOS/trunkline" << 'EOL'
#!/bin/bash
# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

# Activate virtual environment if it exists
if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
fi

# Run the application
python -m src.gui.gui
EOL
    
    chmod +x "Trunkline.app/Contents/MacOS/trunkline"
    
    echo "App bundle created at Trunkline.app"
fi

# Copy the icon to the app bundle
cp Trunkline.icns "Trunkline.app/Contents/Resources/"

echo "Setup complete! You can now run the app by double-clicking Trunkline.app"
