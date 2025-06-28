#!/usr/bin/env python3
"""
Convert a PNG file to macOS .icns format
"""
import os
import subprocess
import sys
from PIL import Image

def convert_to_icns(png_path, output_path=None):
    """Convert a PNG file to ICNS format using sips and iconutil"""
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"Input file not found: {png_path}")
    
    if not output_path:
        output_path = os.path.splitext(png_path)[0] + '.icns'
    
    # Create temporary iconset directory
    iconset_dir = output_path.replace('.icns', '.iconset')
    os.makedirs(iconset_dir, exist_ok=True)
    
    # Define the icon sizes needed for macOS
    icon_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    try:
        # Generate icons at different sizes
        for size in icon_sizes:
            # Normal size
            size_name = f'icon_{size}x{size}.png'
            size_2x = size * 2
            size_name_2x = f'icon_{size}x{size}@2x.png'
            
            # Generate normal size
            subprocess.run([
                'sips', '-z', str(size), str(size),
                '--out', os.path.join(iconset_dir, size_name),
                png_path
            ], check=True)
            
            # Generate @2x size
            subprocess.run([
                'sips', '-z', str(size_2x), str(size_2x),
                '--out', os.path.join(iconset_dir, size_name_2x),
                png_path
            ], check=True)
        
        # Create the .icns file
        subprocess.run([
            'iconutil', '-c', 'icns',
            '-o', output_path,
            iconset_dir
        ], check=True)
        
        print(f"Successfully created {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating ICNS file: {e}", file=sys.stderr)
        return None
    finally:
        # Clean up the iconset directory
        if os.path.exists(iconset_dir):
            import shutil
            shutil.rmtree(iconset_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.png [output.icns]")
        sys.exit(1)
    
    input_png = sys.argv[1]
    output_icns = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_to_icns(input_png, output_icns)
