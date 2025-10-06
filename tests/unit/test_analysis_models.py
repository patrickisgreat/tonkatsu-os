from tonkatsu_os.api.models import AnalysisRequest, AnalysisResult


def test_analysis_request_accepts_optional_config():
    payload = {
        "spectrum_data": [0.1, 0.2, 0.3],
        "preprocess": True,
        "config": {"models": {"svm": True}},
    }

    request = AnalysisRequest(**payload)
    assert request.config is not None
    assert request.config["models"]["svm"] is True


def test_analysis_result_allows_extended_predictions_and_components():
    payload = {
        "predicted_compound": "TestCompound",
        "confidence": 0.85,
        "uncertainty": 0.15,
        "model_agreement": 0.9,
        "top_predictions": [
            {"compound": "TestCompound", "probability": 0.85},
            {"compound": "AltCompound", "probability": 0.1},
        ],
        "individual_predictions": {
            "random_forest": {"compound": "TestCompound", "confidence": 0.84},
            "svm": {"compound": "TestCompound", "confidence": 0.82},
            "neural_network": {"compound": "TestCompound", "confidence": 0.81},
            "database_match": {"compound": "TestCompound", "confidence": 0.85},
        },
        "confidence_analysis": {
            "overall_confidence": 0.85,
            "confidence_components": {
                "probability_score": 0.85,
                "entropy_score": 0.15,
                "peak_match_score": 0.9,
                "model_agreement_score": 0.95,
                "spectral_quality_score": 0.88,
                "similarity_score": 0.87,
            },
            "risk_level": "low",
            "recommendation": "Looks good",
        },
        "processing_time": 0.42,
        "method": "database_similarity",
    }

    result = AnalysisResult(**payload)

    assert result.individual_predictions.random_forest is not None
    assert result.individual_predictions.database_match is not None
    components = result.confidence_analysis.confidence_components.model_dump()
    assert components["similarity_score"] == 0.87
