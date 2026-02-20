# Virtual Try-On Application

A simple AI-powered virtual try-on system that allows users to try on different clothing items virtually.

## Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- Windows/Mac/Linux

### Installation & Running (3 Simple Steps)

#### Step 1: Install Dependencies
```bash
pip install fastapi uvicorn opencv-python numpy pillow mediapipe aiofiles python-multipart pydantic-settings
```

#### Step 2: Run the Application
```bash
python run.py
```

#### Step 3: Open in Browser
Open your web browser and go to:
```
http://localhost:5500
```

That's it! The application is now running.

## How to Use

1. **Upload Your Photo**: Click "Upload Photo" or use the camera to take a picture
2. **Select Clothing**: Choose any clothing item from the gallery
3. **Try It On**: Click the "Try On" button
4. **View Result**: See yourself wearing the selected clothing!

## Features

- 📸 Upload photo or use camera
- 👕 24 different clothing items (dresses, shirts, jackets, etc.)
- 🎨 Real-time virtual try-on
- 💾 Download results
- 📱 Responsive design

## Adding More Clothing Items

To add your own clothing items:

1. Add image files (PNG format) to `frontend/assets/clothes/`
2. Edit `frontend/assets/clothes/clothes.json` and add entries like:

```json
{
  "id": 25,
  "name": "Your Clothing Name",
  "category": "shirt",
  "image": "your-image.png",
  "description": "Description here",
  "gender": "unisex",
  "sizes": {
    "S": { "chest": "36-38", "length": "28" },
    "M": { "chest": "40-42", "length": "29" },
    "L": { "chest": "44-46", "length": "30" },
    "XL": { "chest": "48-50", "length": "31" }
  }
}
```

3. Refresh the browser

## Project Structure

```
Try-on/
├── app/                    # Backend application
│   ├── api/               # API endpoints
│   ├── core/              # Configuration
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   └── utils/             # Utilities
├── frontend/              # Frontend files
│   ├── assets/           # Images and data
│   ├── index.html        # Main page
│   ├── script.js         # JavaScript
│   └── style.css         # Styles
├── storage/              # Uploaded files and results
├── run.py                # Application starter
└── requirements.txt      # Python dependencies

```

## Troubleshooting

### Port Already in Use
If port 5500 is already in use, edit `app/core/config.py` and change:
```python
PORT: int = 5500  # Change to another port like 8000
```

### Missing Dependencies
If you get import errors, reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Camera Not Working
- Make sure your browser has camera permissions
- Try using HTTPS or localhost
- Use the upload option instead

## API Documentation

Once running, visit:
```
http://localhost:5500/docs
```

## Technology Stack

- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: OpenCV, MediaPipe
- **Image Processing**: NumPy, Pillow

## License

Free to use for educational purposes.

## Support

For issues or questions, check the code comments or create an issue in the repository.

---

**Note**: This is a basic implementation for learning purposes. For production use, consider adding:
- User authentication
- Database integration
- Advanced AI models
- Cloud storage
- Better error handling