"""
API Testing Script
Tests the FastAPI endpoints
"""

import requests
import json
import time
from datetime import datetime


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)


def test_health_check(base_url):
    """Test health check endpoint"""
    print_header("Testing Health Check")
    
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        
        data = response.json()
        print("[OK] Health check successful")
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Training date: {data.get('model_training_date', 'N/A')}")
        print(f"  Features count: {data['features_count']}")
        return True
    except Exception as e:
        print(f"[X] Health check failed: {e}")
        return False


def test_model_info(base_url):
    """Test model info endpoint"""
    print_header("Testing Model Info")
    
    try:
        response = requests.get(f"{base_url}/model/info")
        response.raise_for_status()
        
        data = response.json()
        print("[OK] Model info retrieved successfully")
        print(f"  Model type: {data['model_type']}")
        print(f"  Training date: {data['training_date']}")
        print(f"  Features count: {data['features_count']}")
        print(f"  Hyperparameters: {json.dumps(data['hyperparameters'], indent=4)}")
        return True
    except Exception as e:
        print(f"[X] Model info failed: {e}")
        return False


def test_features_list(base_url):
    """Test features list endpoint"""
    print_header("Testing Features List")
    
    try:
        response = requests.get(f"{base_url}/features")
        response.raise_for_status()
        
        data = response.json()
        print("[OK] Features list retrieved successfully")
        print(f"  Total features: {data['features_count']}")
        print(f"  First 10 features: {data['features'][:10]}")
        return True
    except Exception as e:
        print(f"[X] Features list failed: {e}")
        return False


def test_prediction_latest(base_url):
    """Test latest prediction endpoint"""
    print_header("Testing Latest Prediction")
    
    try:
        print("Fetching latest BTC data and making prediction...")
        print("(This may take 10-30 seconds)")
        
        start_time = time.time()
        response = requests.get(f"{base_url}/predict/latest", timeout=60)
        elapsed = time.time() - start_time
        
        response.raise_for_status()
        
        data = response.json()
        print(f"[OK] Prediction successful (took {elapsed:.1f}s)")
        print(f"\n  Prediction: {data['prediction_label']}")
        print(f"  Confidence: {data['confidence']:.2%}")
        print(f"  Probability UP: {data['probability_up']:.2%}")
        print(f"  Probability DOWN: {data['probability_down']:.2%}")
        print(f"  Timestamp: {data['timestamp']}")
        
        # Interpretation
        print(f"\n  Interpretation:")
        if data['confidence'] > 0.6:
            confidence_level = "High"
        elif data['confidence'] > 0.55:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        print(f"  The model predicts BTC will go {data['prediction_label']}")
        print(f"  in the next 24 hours with {confidence_level} confidence.")
        
        return True
    except requests.exceptions.Timeout:
        print("[X] Prediction timed out (>60s)")
        return False
    except Exception as e:
        print(f"[X] Prediction failed: {e}")
        return False


def test_root_endpoint(base_url):
    """Test root endpoint"""
    print_header("Testing Root Endpoint")
    
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        
        data = response.json()
        print("[OK] Root endpoint successful")
        print(f"  Message: {data['message']}")
        print(f"  Version: {data['version']}")
        print(f"  Available endpoints: {list(data['endpoints'].keys())}")
        return True
    except Exception as e:
        print(f"[X] Root endpoint failed: {e}")
        return False


def main():
    """Run all API tests"""
    base_url = "http://localhost:8000"
    
    print("="*60)
    print(" "*15 + "API TESTING SUITE")
    print("="*60)
    print(f"\nBase URL: {base_url}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    print_header("Checking API Connection")
    try:
        response = requests.get(base_url, timeout=5)
        print("[OK] API is running and accessible")
    except requests.exceptions.ConnectionError:
        print("[X] Cannot connect to API")
        print("\nPlease start the API first:")
        print("  python src/api.py")
        print("or")
        print("  uvicorn src.api:app --reload")
        return
    except Exception as e:
        print(f"[X] Connection error: {e}")
        return
    
    # Run tests
    results = []
    
    results.append(("Root Endpoint", test_root_endpoint(base_url)))
    results.append(("Health Check", test_health_check(base_url)))
    results.append(("Model Info", test_model_info(base_url)))
    results.append(("Features List", test_features_list(base_url)))
    results.append(("Latest Prediction", test_prediction_latest(base_url)))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status:8} | {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Additional info
    print("\n" + "="*60)
    print("ADDITIONAL TESTING OPTIONS")
    print("="*60)
    print("\n1. Interactive Swagger UI:")
    print(f"   {base_url}/docs")
    print("\n2. Alternative API docs:")
    print(f"   {base_url}/redoc")
    print("\n3. Manual curl commands:")
    print(f"   curl {base_url}/health")
    print(f"   curl {base_url}/predict/latest")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: Testing failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

