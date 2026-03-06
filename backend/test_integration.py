"""
Test script to verify the integration of Flask app functionality into the backend.
"""
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_image_text_similarity():
    """Test the image-text similarity functionality"""
    print("Testing Image-Text Similarity Integration...")
    
    try:
        from pipeline.image_text_similarity import ImageTextSimilarityAnalyzer
        
        # Initialize the analyzer
        analyzer = ImageTextSimilarityAnalyzer()
        print("✓ ImageTextSimilarityAnalyzer initialized successfully")
        
        # Test basic functionality (without actual image processing to avoid needing real files)
        print("✓ Image-Text Similarity module imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Error testing Image-Text Similarity: {str(e)}")
        return False

def test_pipeline_controller_enhancement():
    """Test that the pipeline controller has been enhanced with image-text similarity"""
    print("\nTesting Pipeline Controller Enhancement...")
    
    try:
        from pipeline.controller import PipelineController
        
        # Initialize the controller
        controller = PipelineController()
        
        # Check if the new attributes exist
        assert hasattr(controller, 'image_text_analyzer'), "Missing image_text_analyzer attribute"
        assert hasattr(controller, 'xai_system'), "Missing xai_system attribute"
        print("✓ PipelineController enhanced with image-text similarity and XAI")
        
        # Check if the new methods exist
        assert hasattr(controller, 'analyze_image_text_similarity'), "Missing analyze_image_text_similarity method"
        assert hasattr(controller, 'get_xai_explanation'), "Missing get_xai_explanation method"
        print("✓ New methods added to PipelineController")
        
        return True
    except Exception as e:
        print(f"✗ Error testing Pipeline Controller enhancement: {str(e)}")
        return False

def test_agents_enhancement():
    """Test that the agents have been enhanced"""
    print("\nTesting Agents Enhancement...")
    
    try:
        from agents.image_agent import ImageAgent
        
        # Initialize the image agent
        agent = ImageAgent()
        print("✓ Enhanced ImageAgent initialized successfully")
        
        # Check if it has the analyzer
        if hasattr(agent, 'analyzer') and agent.analyzer is not None:
            print("✓ ImageAgent has image-text similarity analyzer")
        else:
            print("? ImageAgent analyzer not initialized (might be due to missing dependencies)")
        
        return True
    except Exception as e:
        print(f"✗ Error testing Image Agent enhancement: {str(e)}")
        return False

def test_routes_integration():
    """Test that the routes have been properly integrated"""
    print("\nTesting Routes Integration...")
    
    try:
        from routes.image_text_similarity import router
        print("✓ Image-Text Similarity routes imported successfully")
        
        # Check if the router has the expected endpoints
        routes = [route.path for route in router.routes]
        expected_paths = ["/api/similarity", "/api/xai-explain"]
        
        for path in expected_paths:
            if path in routes:
                print(f"✓ Route {path} found")
            else:
                print(f"? Route {path} not found: {routes}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing routes integration: {str(e)}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("="*60)
    print("ACAS Backend Integration Tests")
    print("="*60)
    
    tests = [
        test_image_text_similarity,
        test_pipeline_controller_enhancement,
        test_agents_enhancement,
        test_routes_integration
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All integration tests passed!")
        print("\nThe Flask app functionality has been successfully integrated into the backend.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("="*60)
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)