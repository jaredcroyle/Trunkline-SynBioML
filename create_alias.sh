#!/bin/bash

# Remove any existing alias
rm -f "$HOME/Applications/TrunklineML"

# Create the alias using osascript
osascript <<EOD
tell application "Finder"
    set appPath to (POSIX file "/Applications/TrunklineML.app") as text
    set appAlias to make new alias file at (path to applications folder from user domain) to appPath with properties {name:"TrunklineML"}
    set the icon of appAlias to file "icon.icns" of folder "Resources" of application file appPath
end tell
EOD

# Set the custom icon
touch "$HOME/Applications/TrunklineML"
/usr/bin/SetFile -a C "$HOME/Applications/TrunklineML"
