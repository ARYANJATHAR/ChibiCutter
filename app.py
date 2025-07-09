import streamlit as st
import os
import zipfile
import io
from PIL import Image, ImageFilter, ImageEnhance
import tempfile
import numpy as np
import cv2

def cut_stickers(image, cols, rows, margin_x, margin_y):
    """
    Cut an image into a grid of stickers
    """
    img_width, img_height = image.size
    
    # Calculate cell dimensions
    cell_width = img_width // cols
    cell_height = img_height // rows
    
    stickers = []
    
    # Crop each cell
    for row in range(rows):
        for col in range(cols):
            left = col * cell_width + margin_x
            top = row * cell_height + margin_y
            right = (col + 1) * cell_width - margin_x
            bottom = (row + 1) * cell_height - margin_y
            
            # Make sure we don't go outside image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)
            
            # Crop the region
            cropped = image.crop((left, top, right, bottom))
            
            # Store sticker info
            sticker_num = row * cols + col + 1
            stickers.append({
                'image': cropped,
                'name': f"sticker_{sticker_num}.png",
                'number': sticker_num
            })
    
    return stickers

def create_zip_file(stickers):
    """
    Create a ZIP file containing all stickers
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for sticker in stickers:
            # Save image to bytes
            img_buffer = io.BytesIO()
            sticker['image'].save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Add to zip
            zip_file.writestr(sticker['name'], img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

def show_home_page():
    """
    Display the home page with instructions and examples
    """
    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem; color: white;'>
        <h1 style='margin: 0; font-size: 3rem;'>🎨 Chibi Sticker Creator</h1>
        <p style='font-size: 1.2rem; margin: 1rem 0 0 0;'>Cut your AI-generated character grids into individual stickers!</p>
        <p style='font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.9;'>✂️ Smart Margin Detection • Perfect Sticker Cutting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Capabilities section
    st.markdown("## 🤖 What Our Tool Can Do")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **✂️ Perfect Sticker Cutting**
        - Simple 3×4 grid layout
        - Customizable rows and columns
        - Individual sticker download
        """)
    
    with col2:
        st.markdown("""
        **📏 Smart Margin Calculation**
        - Automatically finds optimal margins
        - Avoids cutting into characters
        - Removes grid borders perfectly
        """)
    
    with col3:
        st.markdown("""
        **🎁 Easy Download Options**
        - Download all stickers as ZIP
        - Individual sticker downloads
        - High quality PNG format
        """)
    
    # Simple Step-by-Step Guide
    st.markdown("## 🚀 How to Use This Tool")
    
    # Step 1
    st.markdown("""
    ### Step 1: Go to ChatGPT 🤖
    Visit [ChatGPT](https://chat.openai.com) or any AI image generator
    """)
    
    # Step 2
    st.markdown("""
    ### Step 2: Use This Prompt 📝
    Copy and paste this prompt:
    """)
    
    prompt = """Convert the image into a set of 12 chibi sticker (4x4 grid) with outfit similar to this one, 
    including expression of laughing being angry, crying, sulking, thinking, being sleepy, blowing a kiss, winking, being surprise"""
    
    st.code(prompt, language=None)
    
    # Step 3
    st.markdown("""
    ### Step 3: Generate & Download 🖼
    - Generate the image in ChatGPT
    - Right-click and save the image to your computer
    """)
    
    # Step 4
    st.markdown("""
    ### Step 4: Upload Here ⬆️
    Click the button below and upload your saved image. Our tool will automatically:
    - 📐 **Calculate optimal margins** to avoid cutting into characters
    - ✂️ **Cut perfect stickers** with clean edges
    - 🎁 **Create downloadable assets** ready to use
    """)

def show_sticker_cutter():
    """
    Display the sticker cutting interface
    """
    st.markdown("## ✂️ Cut Your Sticker Grid")
    
    # Back to home button
    if st.button("🏠 Back to Home", key="back_home"):
        st.session_state.page = "home"
        st.experimental_rerun()
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload your grid image",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload an image containing a grid of stickers/characters"
    )
    
    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📸 Your Image")
            st.image(image, caption="Original Grid Image", width=None)
        
        with col2:
            st.subheader("⚙️ Grid Settings")
            
            # Default grid size of 3x4
            default_cols = 3
            default_rows = 4
            
            # Calculate optimal margins for the default grid
            with st.spinner("🔍 Calculating optimal margins..."):
                auto_margin_x, auto_margin_y = detect_optimal_margins(image, default_cols, default_rows)
            
            # Display image dimensions and aspect ratio
            aspect_ratio = image.size[0] / image.size[1]
            st.info(f"📏 Image size: {image.size[0]} × {image.size[1]} pixels")
            
            # Grid size settings
            st.subheader("🔲 Grid Size")
            cols = st.slider("Columns", min_value=1, max_value=10, value=default_cols, help="Number of columns in your grid")
            rows = st.slider("Rows", min_value=1, max_value=10, value=default_rows, help="Number of rows in your grid")
            
            # Recalculate margins if grid size changes
            if cols != default_cols or rows != default_rows:
                with st.spinner("🔄 Recalculating margins for new grid size..."):
                    auto_margin_x, auto_margin_y = detect_optimal_margins(image, cols, rows)
            
            # Margin settings
            st.subheader("📐 Margin Settings")
            st.info(f"🤖 Suggested margins: {auto_margin_x}px × {auto_margin_y}px")
            
            adjust_margins = st.checkbox("🔧 Manually adjust margins", value=False, help="Check this to override automatic margins")
            
            if adjust_margins:
                st.warning("⚠️ Using manual margins")
                margin_x = st.slider("Horizontal Margin", min_value=0, max_value=100, value=auto_margin_x, help="Pixels to trim from left/right of each cell")
                margin_y = st.slider("Vertical Margin", min_value=0, max_value=100, value=auto_margin_y, help="Pixels to trim from top/bottom of each cell")
            else:
                # Use auto-detected margins
                margin_x, margin_y = auto_margin_x, auto_margin_y
            
            # Show final grid info
            st.subheader("📊 Final Settings")
            total_stickers = cols * rows
            st.info(f"🎯 Grid: {cols} × {rows} = {total_stickers} stickers")
            st.info(f"📐 Margins: {margin_x}px × {margin_y}px")
            
            # Process button
            if st.button("✂️ Cut Stickers"):
                with st.spinner("🔄 Processing your stickers..."):
                    try:
                        # Cut the stickers
                        stickers = cut_stickers(image, cols, rows, margin_x, margin_y)
                        
                        # Store in session state
                        st.session_state.stickers = stickers
                        st.session_state.processed = True
                        
                        st.success(f"✅ Successfully created {len(stickers)} stickers!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
    
    # Show results if processed
    if hasattr(st.session_state, 'processed') and st.session_state.processed:
        st.subheader("🎉 Your Stickers")
        
        # Create download button
        zip_file = create_zip_file(st.session_state.stickers)
        st.download_button(
            label="📥 Download All Stickers (ZIP)",
            data=zip_file,
            file_name="stickers.zip",
            mime="application/zip"
        )
        
        # Display preview of stickers
        st.subheader("👀 Preview")
        
        # Calculate grid layout for preview
        cols_preview = min(6, len(st.session_state.stickers))  # Max 6 columns for preview
        
        sticker_cols = st.columns(cols_preview)
        
        for i, sticker in enumerate(st.session_state.stickers):
            with sticker_cols[i % cols_preview]:
                st.image(
                    sticker['image'], 
                    caption=f"Sticker {sticker['number']}", 
                    width=None
                )
                
                # Individual download button
                img_buffer = io.BytesIO()
                sticker['image'].save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"⬇️ {sticker['number']}",
                    data=img_buffer,
                    file_name=sticker['name'],
                    mime="image/png",
                    key=f"download_{sticker['number']}"
                )

def detect_grid_lines_improved(image):
    """
    Improved grid line detection using Hough Line Transform
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply stronger blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive threshold for better edge detection
    edges = cv2.Canny(blurred, 30, 80, apertureSize=3)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(image.size) * 0.3))
    
    h_lines = []
    v_lines = []
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            
            # Check if line is horizontal (theta close to 0 or pi)
            if abs(theta) < 0.1 or abs(theta - np.pi) < 0.1:
                y = int(rho / np.sin(theta)) if abs(np.sin(theta)) > 0.1 else int(rho)
                if 10 < y < image.size[1] - 10:  # Avoid edges
                    h_lines.append(y)
            
            # Check if line is vertical (theta close to pi/2)
            elif abs(theta - np.pi/2) < 0.1:
                x = int(rho / np.cos(theta)) if abs(np.cos(theta)) > 0.1 else int(rho)
                if 10 < x < image.size[0] - 10:  # Avoid edges
                    v_lines.append(x)
    
    return h_lines, v_lines

def analyze_grid_structure(image):
    """
    Dynamic grid analysis using multiple detection methods
    Returns: cols, rows, margin_x, margin_y
    """
    width, height = image.size
    
    # Method 1: Pattern repetition analysis
    try:
        pattern_cols, pattern_rows = detect_repeating_patterns(image)
    except Exception:
        pattern_cols, pattern_rows = 3, 3
    
    # Method 2: Content distribution analysis
    try:
        content_cols, content_rows = analyze_content_distribution(image)
    except Exception:
        content_cols, content_rows = 3, 3
    
    # Method 3: Line detection
    try:
        h_lines, v_lines = detect_grid_lines_improved(image)
        h_lines = [line for line in h_lines if 30 < line < height - 30]
        v_lines = [line for line in v_lines if 30 < line < width - 30]
        h_lines = sorted(list(set(h_lines)))
        v_lines = sorted(list(set(v_lines)))
        
        line_rows = len(h_lines) + 1 if 1 <= len(h_lines) <= 6 else 0
        line_cols = len(v_lines) + 1 if 1 <= len(v_lines) <= 6 else 0
    except Exception:
        line_rows, line_cols = 0, 0
        h_lines, v_lines = [], []
    
    # Combine results with voting system
    candidates = []
    if 2 <= pattern_cols <= 6 and 2 <= pattern_rows <= 6:
        candidates.append((pattern_cols, pattern_rows, "pattern"))
    if 2 <= content_cols <= 6 and 2 <= content_rows <= 6:
        candidates.append((content_cols, content_rows, "content"))
    if line_cols > 0 and line_rows > 0:
        candidates.append((line_cols, line_rows, "lines"))
    
    # Choose the most common result, or best one if no consensus
    if candidates:
        # Count votes for each grid size
        vote_count = {}
        for cols, rows, method in candidates:
            key = (cols, rows)
            if key not in vote_count:
                vote_count[key] = []
            vote_count[key].append(method)
        
        # Find the grid size with most votes
        best_key = max(vote_count.keys(), key=lambda k: len(vote_count[k]))
        cols, rows = best_key
        detection_methods = vote_count[best_key]
    else:
        # Fallback to aspect ratio if all methods fail
        cols, rows = fallback_aspect_ratio_analysis(width, height)
        detection_methods = ["fallback"]
    
    # Calculate margins
    if h_lines and v_lines:
        margin_y = min(h_lines[0] // 2, (height - h_lines[-1]) // 2, 50)
        margin_x = min(v_lines[0] // 2, (width - v_lines[-1]) // 2, 50)
    else:
        margin_x = max(5, min(30, width // (cols * 6)))
        margin_y = max(5, min(30, height // (rows * 6)))
    
    return cols, rows, int(margin_x), int(margin_y), detection_methods

def fallback_aspect_ratio_analysis(width, height):
    """
    Fallback method that's more dynamic than hardcoded ranges
    """
    aspect_ratio = width / height
    
    # Calculate likely dimensions based on typical cell aspect ratios
    # Assume cells are roughly square (0.8 to 1.2 aspect ratio)
    
    if aspect_ratio > 1:  # Landscape
        # Start with square cells and adjust
        base_cols = int(round(aspect_ratio))
        base_rows = 1
        
        # Try to find a reasonable factorization
        for rows in range(2, 7):
            cols = round(aspect_ratio * rows)
            if 2 <= cols <= 6:
                cell_aspect = (width / cols) / (height / rows)
                if 0.5 <= cell_aspect <= 2.0:  # Reasonable cell proportions
                    return cols, rows
        
        return max(2, min(6, base_cols)), max(2, min(6, base_rows))
    
    else:  # Portrait
        # Start with square cells and adjust
        base_rows = int(round(1 / aspect_ratio))
        base_cols = 1
        
        # Try to find a reasonable factorization
        for cols in range(2, 7):
            rows = round(cols / aspect_ratio)
            if 2 <= rows <= 6:
                cell_aspect = (width / cols) / (height / rows)
                if 0.5 <= cell_aspect <= 2.0:  # Reasonable cell proportions
                    return cols, rows
        
        return max(2, min(6, base_cols)), max(2, min(6, base_rows))

def detect_repeating_patterns(image):
    """
    Analyze image to detect repeating patterns that indicate grid structure
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    # Analyze horizontal patterns (to find rows)
    row_means = []
    for y in range(height):
        row_mean = np.mean(gray[y, :])
        row_means.append(row_mean)
    
    # Analyze vertical patterns (to find columns)  
    col_means = []
    for x in range(width):
        col_mean = np.mean(gray[:, x])
        col_means.append(col_mean)
    
    # Find periodic patterns in the means
    rows = find_periodic_pattern(row_means, height)
    cols = find_periodic_pattern(col_means, width)
    
    return cols, rows

def find_periodic_pattern(signal, dimension):
    """
    Find the most likely number of repetitions in a signal
    """
    signal = np.array(signal)
    
    # Try different numbers of divisions
    best_score = 0
    best_divisions = 2
    
    for divisions in range(2, min(8, dimension // 50)):  # Test 2-7 divisions, but ensure each cell is at least 50px
        segment_size = len(signal) // divisions
        if segment_size < 20:  # Each segment should be at least 20 pixels
            continue
            
        segments = []
        for i in range(divisions):
            start = i * segment_size
            end = start + segment_size
            if end <= len(signal):
                segments.append(signal[start:end])
        
        if len(segments) < 2:
            continue
            
        # Calculate similarity between segments
        score = 0
        comparisons = 0
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                # Compute correlation between segments
                corr = np.corrcoef(segments[i], segments[j])[0, 1]
                if not np.isnan(corr):
                    score += abs(corr)
                    comparisons += 1
        
        if comparisons > 0:
            avg_score = score / comparisons
            if avg_score > best_score:
                best_score = avg_score
                best_divisions = divisions
    
    return best_divisions

def analyze_content_distribution(image):
    """
    Analyze content distribution to estimate grid size
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find content regions (non-empty areas)
    height, width = edges.shape
    
    # Divide image into potential grid cells and analyze content density
    best_cols, best_rows = 3, 3  # fallback
    best_score = 0
    
    # Test different grid configurations
    for test_cols in range(2, 7):
        for test_rows in range(2, 7):
            cell_width = width // test_cols
            cell_height = height // test_rows
            
            # Skip if cells would be too small
            if cell_width < 50 or cell_height < 50:
                continue
            
            content_densities = []
            
            # Calculate content density for each cell
            for row in range(test_rows):
                for col in range(test_cols):
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height
                    
                    cell = edges[y1:y2, x1:x2]
                    density = np.sum(cell) / (cell_width * cell_height)
                    content_densities.append(density)
            
            # Good grids should have similar content densities across cells
            if len(content_densities) > 1:
                density_std = np.std(content_densities)
                density_mean = np.mean(content_densities)
                
                # Score based on consistency (low std dev) and sufficient content
                if density_mean > 0:
                    consistency_score = 1 / (1 + density_std / density_mean)
                    content_score = min(1.0, density_mean / 50)  # Normalize content score
                    total_score = consistency_score * content_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_cols = test_cols
                        best_rows = test_rows
    
    return best_cols, best_rows

def estimate_grid_by_aspect_ratio(image):
    """Fallback method using dynamic aspect ratio analysis"""
    width, height = image.size
    
    cols, rows = fallback_aspect_ratio_analysis(width, height)
    
    # Calculate margins as percentage of cell size
    margin_x = width // (cols * 8)  # ~12.5% of cell width
    margin_y = height // (rows * 8)  # ~12.5% of cell height
    
    return cols, rows, max(5, margin_x), max(5, margin_y)

def detect_optimal_margins(image, cols=3, rows=4):
    """
    Detect optimal margins for cutting stickers with a known grid size.
    This function focuses on finding margins that ensure each sticker is properly centered
    and doesn't overlap with others.
    
    Args:
        image: PIL Image object
        cols: Number of columns in grid (default 3)
        rows: Number of rows in grid (default 4)
    
    Returns:
        margin_x, margin_y: Optimal horizontal and vertical margins
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Calculate cell dimensions
    cell_width = width // cols
    cell_height = height // rows
    
    # Convert to grayscale for processing
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply edge detection to find content boundaries
    edges = cv2.Canny(gray, 30, 100)
    
    # Variables to track margins
    best_margin_x = cell_width // 8  # Default is 12.5% of cell width
    best_margin_y = cell_height // 8  # Default is 12.5% of cell height
    
    # Sample a few cells to determine optimal margins
    cell_samples = []
    for row in range(rows):
        for col in range(cols):
            # Only check some cells for efficiency
            if len(cell_samples) >= 4:  # Sample at most 4 cells
                break
                
            # Calculate cell coordinates
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            cell = edges[y1:y2, x1:x2]
            
            # Only use cells with significant content
            if np.sum(cell) > (cell_width * cell_height * 0.01):
                cell_samples.append(cell)
    
    # If we found cells with content, analyze them
    if cell_samples:
        # Find horizontal margins
        margin_x_candidates = []
        for cell in cell_samples:
            # Find left margin - look for first column with significant content
            for x in range(0, cell_width // 2):
                col_sum = np.sum(cell[:, x])
                if col_sum > (cell_height * 5):  # Threshold for content
                    margin_x_candidates.append(max(0, x - 5))  # Add some buffer
                    break
            
            # Find right margin - look for last column with significant content
            for x in range(cell_width - 1, cell_width // 2, -1):
                if x >= cell.shape[1]:
                    continue
                col_sum = np.sum(cell[:, x])
                if col_sum > (cell_height * 5):
                    right_margin = max(0, cell_width - x - 5)  # Add buffer
                    margin_x_candidates.append(right_margin)
                    break
        
        # Find vertical margins
        margin_y_candidates = []
        for cell in cell_samples:
            # Find top margin - look for first row with significant content
            for y in range(0, cell_height // 2):
                if y >= cell.shape[0]:
                    continue
                row_sum = np.sum(cell[y, :])
                if row_sum > (cell_width * 5):  # Threshold for content
                    margin_y_candidates.append(max(0, y - 5))  # Add buffer
                    break
            
            # Find bottom margin - look for last row with significant content
            for y in range(cell_height - 1, cell_height // 2, -1):
                if y >= cell.shape[0]:
                    continue
                row_sum = np.sum(cell[y, :])
                if row_sum > (cell_width * 5):
                    bottom_margin = max(0, cell_height - y - 5)  # Add buffer
                    margin_y_candidates.append(bottom_margin)
                    break
        
        # Calculate final margins
        if margin_x_candidates:
            # Use median to avoid outliers
            best_margin_x = int(np.median(margin_x_candidates))
        if margin_y_candidates:
            best_margin_y = int(np.median(margin_y_candidates))
    
    # Ensure margins are reasonable (not too small or large)
    best_margin_x = max(5, min(cell_width // 4, best_margin_x))
    best_margin_y = max(5, min(cell_height // 4, best_margin_y))
    
    return best_margin_x, best_margin_y

# Main App
def main():
    st.set_page_config(
        page_title="🎨 Chibi Sticker Creator",
        page_icon="✂️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Navigation
    if st.session_state.page == "home":
        show_home_page()
        
        # Start button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Creating Stickers!"):
                st.session_state.page = "cutter"
                st.experimental_rerun()
    
    elif st.session_state.page == "cutter":
        show_sticker_cutter()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;'>
        <p style='margin: 0; color: #6c757d;'>
            🎨 <strong>Chibi Sticker Creator</strong> | Made with ❤️ using Streamlit<br>
            Smart margin detection for perfect sticker cutting from ChatGPT, DALL-E, and other AI generators!
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
