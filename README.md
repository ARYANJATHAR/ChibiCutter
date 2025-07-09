# ChibiCutter

ChibiCutter is a tool that helps you cut AI-generated character grids into individual stickers with perfect margins. Whether you've created a grid of chibi characters or emoji expressions using ChatGPT, DALL-E, or other AI image generators, this tool makes it easy to split them into individual stickers.

## Features

- **Simple Grid Cutting** - Cut any image into a grid of individual stickers
- **Smart Margin Detection** - Automatically calculates optimal margins to avoid cutting into characters
- **Customizable Grid Size** - Set your own row and column count (default 3×4)
- **Bulk Download** - Get all your stickers in a single ZIP file
- **Individual Downloads** - Download specific stickers separately
- **High Quality Output** - All stickers are saved as PNG files with transparency preserved

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ChibiCutter.git
   cd ChibiCutter
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```

2. Open the provided URL in your web browser (typically http://localhost:8501)

3. Follow these steps in the application:
   - Upload your grid image
   - Adjust grid dimensions if needed (default is 3×4)
   - The app will automatically calculate optimal margins
   - Click "Cut Stickers" to process your image
   - Download all stickers as a ZIP file or download individual stickers

## Example

The app works best with grid-style images like these:
- Character expression sheets
- Emoji collections
- Sticker packs with uniform layouts

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- Pillow (PIL)
- NumPy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 