# ACAS API Testing with Postman

## API Endpoints Overview

The ACAS backend provides the following endpoints:

### 1. Health Check
- **GET** `http://localhost:8000/`
- Description: Check if the backend is running
- Response: `{"status": "Backend running"}`

### 2. Main Verification Endpoint
- **POST** `http://localhost:8000/verify/verify`
- Description: Main fake news verification pipeline
- Requires: JSON payload with user and tweet data

### 3. Image-Text Similarity Analysis
- **POST** `http://localhost:8000/api/similarity`
- Description: Analyze similarity between image and text
- Supports: File upload or image URL

### 4. XAI Explanation
- **POST** `http://localhost:8000/api/xai-explain`
- Description: Get detailed XAI explanations with visualizations
- Supports: File upload or image URL

## Postman Setup Instructions

### 1. Create a New Collection
1. Open Postman
2. Click "New" → "Collection"
3. Name it "ACAS Fake News Detection"
4. Create requests for each endpoint

### 2. Health Check Request
- **Method**: GET
- **URL**: `http://localhost:8000/`
- **Headers**: None required
- **Expected Response**: 
```json
{
    "status": "Backend running"
}
```

### 3. Main Verification Request
- **Method**: POST
- **URL**: `http://localhost:8000/verify/verify`
- **Headers**: 
  - Content-Type: application/json
- **Body** (raw JSON):
```json
{
    "user": {
        "name": "John Doe",
        "username": "johndoe",
        "verified": false
    },
    "tweet": {
        "text": "Breaking: Scientists discover cure for all diseases! #health #miracle",
        "image_url": "https://example.com/image.jpg"
    }
}
```

### 4. Image-Text Similarity Request (File Upload)
- **Method**: POST
- **URL**: `http://localhost:8000/api/similarity`
- **Headers**: None (multipart/form-data automatically set)
- **Body** (form-data):
  - Key: `image`, Type: File, Value: [Select image file]
  - Key: `text`, Type: Text, Value: "A beautiful sunset over the ocean"

### 5. Image-Text Similarity Request (URL)
- **Method**: POST
- **URL**: `http://localhost:8000/api/similarity`
- **Headers**: None (multipart/form-data automatically set)
- **Body** (form-data):
  - Key: `image_url`, Type: Text, Value: `https://example.com/image.jpg`
  - Key: `text`, Type: Text, Value: "A beautiful sunset over the ocean"

### 6. XAI Explanation Request (File Upload)
- **Method**: POST
- **URL**: `http://localhost:8000/api/xai-explain`
- **Headers**: None (multipart/form-data automatically set)
- **Body** (form-data):
  - Key: `image`, Type: File, Value: [Select image file]
  - Key: `text`, Type: Text, Value: "A beautiful sunset over the ocean"

### 7. XAI Explanation Request (URL)
- **Method**: POST
- **URL**: `http://localhost:8000/api/xai-explain`
- **Headers**: None (multipart/form-data automatically set)
- **Body** (form-data):
  - Key: `image_url`, Type: Text, Value: `https://example.com/image.jpg`
  - Key: `text`, Type: Text, Value: "A beautiful sunset over the ocean"

## Expected Response Examples

### Main Verification Response:
```json
{
    "masked_user_id": "ANON_a1b2c3d4",
    "credibility": {
        "verdict": "FAKE",
        "confidence": 85,
        "reason": "Cross-check complete. Text flagged as FAKE but Image flagged as REAL. Consistency mismatch detected."
    }
}
```

### Image-Text Similarity Response:
```json
{
    "similarity_score": 0.78,
    "decision": "MATCH",
    "ocr_text": "Ocean Sunset Resort",
    "input_text": "A beautiful sunset over the ocean",
    "from_url": false
}
```

### XAI Explanation Response:
```json
{
    "similarity_score": 0.78,
    "decision": "MATCH",
    "ocr_text": "Ocean Sunset Resort",
    "input_text": "A beautiful sunset over the ocean",
    "explanation": "This matches because the visual content and text have a similarity score of 0.7800 compared to the threshold of 0.24.",
    "confidence_level": "high",
    "visualizations": {
        "gradcam": "base64_encoded_image_data...",
        "text_attention": "base64_encoded_image_data...",
        "decomposition": "base64_encoded_image_data..."
    },
    "decomposition_details": {
        "overall_similarity": 0.78,
        "concept_decomposition": {...},
        "top_concepts": [...]
    }
}
```

## Testing Tips

1. **Start the backend first**: Make sure the backend is running on `http://localhost:8000`
2. **Test health check**: Always start with the health check endpoint to ensure the server is running
3. **Use sample images**: For file uploads, use small image files (< 5MB) for faster testing
4. **Check response times**: Image processing can take 10-30 seconds depending on image size
5. **Monitor console logs**: The backend will print detailed logs for debugging

## Common Issues and Solutions

1. **Connection refused**: Make sure the backend is running
2. **422 Unprocessable Entity**: Check that all required fields are provided
3. **500 Internal Server Error**: Check backend console logs for detailed error messages
4. **File upload issues**: Ensure the image file is not corrupted and is in a supported format (jpg, png, etc.)

## Sample Test Data

You can use these sample texts for testing:
- "A beautiful sunset over the ocean"
- "Breaking: Scientists discover cure for all diseases!"
- "Breaking news: Earthquake hits major city"
- "Health tip: Drink 8 glasses of water daily"

For images, you can use:
- Nature photos (landscapes, sunsets)
- News images
- Health-related images
- Any image that relates to your test text