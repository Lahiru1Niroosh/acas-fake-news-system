from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
# Import PipelineController dynamically inside the functions to avoid circular imports

router = APIRouter()

@router.post("/api/similarity")
async def api_similarity(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    text: str = Form(...)
):
    """API endpoint for image-text similarity analysis"""
    try:
        # Check if image and text are provided
        if not image and not image_url:
            return JSONResponse(
                status_code=400,
                content={'error': 'Image is required'}
            )
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={'error': 'Text is required'}
            )
        
        # Create controller instance
        from ..pipeline.controller import PipelineController
        controller = PipelineController()
        
        # Handle image input - either file upload or URL
        if image:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
                temp_path = temp_file.name
                content = await image.read()
                temp_file.write(content)
            
            try:
                # Calculate similarity
                result = controller.analyze_image_text_similarity(temp_path, text, from_url=False)
                result['from_url'] = False
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        else:
            # Handle image URL
            result = controller.analyze_image_text_similarity(image_url, text, from_url=True)
            result['from_url'] = True

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@router.post("/api/xai-explain")
async def api_xai_explain(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    text: str = Form(...)
):
    """API endpoint for XAI explanation"""
    try:
        if not image and not image_url:
            return JSONResponse(
                status_code=400,
                content={'error': 'Image is required'}
            )
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={'error': 'Text is required'}
            )
        
        # Create controller instance
        from ..pipeline.controller import PipelineController
        controller = PipelineController()
        
        # Handle image input - either file upload or URL
        if image:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
                temp_path = temp_file.name
                content = await image.read()
                temp_file.write(content)
            
            try:
                # Get comprehensive XAI explanation
                explanation = controller.get_xai_explanation(temp_path, text, from_url=False)
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        else:
            # Handle image URL
            explanation = controller.get_xai_explanation(image_url, text, from_url=True)

        return explanation

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )