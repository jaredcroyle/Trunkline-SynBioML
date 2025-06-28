#!/usr/bin/env python3
"""
Create a simple app icon for Trunkline ML
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    # Create a 512x512 image with a blue gradient background
    size = 512
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a circular background with gradient
    center = size // 2
    max_radius = center
    steps = 100
    
    for i in range(steps):
        # Calculate current radius and alpha
        ratio = i / steps
        radius = int(max_radius * (1 - ratio))
        alpha = int(255 * (1 - ratio * 0.5))  # Fade out towards the edges
        
        # Blue color (Material Blue 700)
        r, g, b = 25, 118, 210
        
        # Draw circle
        bbox = [
            center - radius,
            center - radius,
            center + radius,
            center + radius
        ]
        draw.ellipse(bbox, fill=(r, g, b, alpha), outline=None)
    
    # Add text
    try:
        # Try to use a nice font if available
        font = ImageFont.truetype("Arial Bold", 100)
    except:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Draw the text
    text = "TL"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((size - text_width) // 2, (size - text_height) // 2)
    
    # Draw text with a slight shadow
    draw.text((position[0]+3, position[1]+3), text, fill=(0, 0, 0, 100), font=font)
    draw.text(position, text, fill=(255, 255, 255), font=font)
    
    # Save the icon
    output_path = os.path.join(os.path.dirname(__file__), 'app_icon.png')
    img.save(output_path, 'PNG')
    print(f"Icon saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_icon()
