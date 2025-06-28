import base64
import os

# A simple icon in base64 format (16x16 transparent PNG)
ICON_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAbwAAAG8B8aLcQwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADUSURBVDiNY2AYWLAwMDAwMDExMTAzMh5hYGDgY2BgYGBkYGBgYmRgYGBgYGBgYGJkYGBiZGBgYGRgYGRgYGBkYGBgYGRgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGD4D8QAAA6gB0m5Mv3rAAAAAElFTkSuQmCC
"""

# Create the icon file
def create_icon():
    icon_path = os.path.join("assets", "icon.icns")
    os.makedirs("assets", exist_ok=True)
    
    # For now, we'll create a .png file as .icns requires special tools
    png_path = os.path.join("assets", "icon.png")
    with open(png_path, "wb") as f:
        f.write(base64.b64decode(ICON_BASE64))
    
    print(f"Created icon at: {png_path}")
    print("Note: For a proper .icns file, you'll need to use the 'iconutil' tool on macOS.")
    print("For now, we'll proceed with the .png file.")
    
    return png_path

if __name__ == "__main__":
    create_icon()
