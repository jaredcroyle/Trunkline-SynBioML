#!/usr/bin/env python3
"""
Convert PNG to ICNS format for macOS app icons.
"""
import os
import sys
from PIL import Image

def convert_png_to_icns(png_path, output_dir=None):
    """Convert a PNG file to ICNS format."""
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"Input file not found: {png_path}")
    
    if output_dir is None:
        output_dir = os.path.dirname(png_path)
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(png_path))[0]
    icns_path = os.path.join(output_dir, f"{base_name}.icns")
    
    # Create iconset directory
    iconset_dir = os.path.join(output_dir, f"{base_name}.iconset")
    os.makedirs(iconset_dir, exist_ok=True)
    
    try:
        # Open the image
        img = Image.open(png_path)
        
        # Required icon sizes for ICNS
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        
        # Generate icons at different sizes
        for size in sizes:
            # Standard size
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(f"{iconset_dir}/icon_{size}x{size}.png")
            
            # Double size for retina displays
            if size < 1024:  # Don't create 2048x2048 icon
                resized_2x = img.resize((size*2, size*2), Image.Resampling.LANCZOS)
                resized_2x.save(f"{iconset_dir}/icon_{size}x{size}@2x.png")
        
        # Convert to ICNS using sips and iconutil
        os.system(f'iconutil -c icns "{iconset_dir}" -o "{icns_path}"')
        print(f"Successfully created {icns_path}")
        return icns_path
        
    except Exception as e:
        print(f"Error converting icon: {e}", file=sys.stderr)
        return None
    finally:
        # Clean up iconset directory
        if os.path.exists(iconset_dir):
            import shutil
            shutil.rmtree(iconset_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.png> [output_dir]", file=sys.stderr)
        sys.exit(1)
    
    png_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = convert_png_to_icns(png_path, output_dir)
    if not result:
        sys.exit(1)
